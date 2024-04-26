import requests
from bs4 import BeautifulSoup
import pandas as pd
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import yfinance as yf
import pypfopt
from pypfopt import EfficientFrontier
from pypfopt import risk_models
import pandas as pd

resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
soup = BeautifulSoup(resp.text, 'lxml')
table=soup.find('table',{'class':'wikitable sortable'})
tickers=[]
securities=[]
gics_industries=[]
gics_sub_industries=[]

options = []


for row in table.findAll('tr')[1:]:
    ticker = row.findAll('td')[0].text
    security = row.findAll('td')[1].text
    gics_industry = row.findAll('td')[3].text
    gics_sub_industry = row.findAll('td')[4].text

    tickers.append(ticker.lower().replace(r"\n", " "))
    securities.append(security)
    gics_industries.append(gics_industry.lower())
    gics_sub_industries.append(gics_sub_industry.lower())
        
df = list(zip(tickers,securities))
df=pd.DataFrame(df, columns=['tickers','securities'])
df.set_index('tickers', inplace=True)
for tic in df.index:
    options.append({'label':'{} {}'.format(tic,df.loc[tic]['securities']), 'value':tic})
fyear=[{'label':"1",'value':1},{'label':"2",'value':2},{'label':"3",'value':3}]  
risk=[{'label':"risk averse",'value':1},{'label':"moderate",'value':2},{'label':"aggressive",'value':3}]  

app = dash.Dash()
server=app.server
app.layout = html.Div([
    html.H1('Financial Dashboard(by Shaantanu)'),
    html.Div([
        html.H3('Select stock symbols you want to add to your portfolio:', style={'paddingRight':'30px'}),
        dcc.Dropdown(
            id='my_ticker_symbol',
            options=options,
            value=['TSLA'],
            multi=True
        )
    ], style={'display':'inline-block', 'verticalAlign':'top', 'width':'30%'}),
    html.Div([
        html.H3('Select look forward period'),
        dcc.Dropdown(
            id='my_date_picker',
            options=fyear,
            value=["1"],
            multi=False
        )
    ], style={'display':'inline-block','verticalAlign':'top', 'width':'30%'}),
        html.Div([
        html.H3('Select risk appetite'),
        dcc.Dropdown(
            id='my_risk',
            options=risk,
            value=["aggressive"],
            multi=False
        )
    ], style={'display':'inline-block'}),
    html.Div([
        html.Button(
            id='submit-button',
            n_clicks=0,
            children='Prices Submit',
            style={'fontSize':24, 'marginLeft':'30px'}
        ),
    ], style={'display':'inline-block'}),
            html.Div([
        html.Button(
            id='submit-button2',
            n_clicks=0,
            children='Portfolio Button',
            style={'fontSize':24, 'marginLeft':'30px'}
        ),
    ], style={'display':'inline-block'}),
    dcc.Graph(
        id='my_graph',
        figure={
            'data': [
                {'x': [1,2], 'y': [3,1]}
            ]
        }
    )
    ,dcc.Graph(
        id='my_graph2',
        figure={
            'data': [
                {'x': [1,2], 'y': [3,1]}
            ]
        }
    )
    
])
    
@app.callback(
    Output('my_graph', 'figure'),
    [Input('submit-button', 'n_clicks')],
    [State('my_ticker_symbol', 'value'),
    State('my_date_picker', 'value'),
    State('my_risk','value')])
def update_graph(n_clicks, stock_ticker, lookforward_period,risk):
    content=[]
    content=stock_ticker
    data=pd.DataFrame()
    traces = []
    for contents in content:
        data=pd.concat([data,yf.download(f"{contents}", start="2015-01-01", end="2020-01-01").iloc[:,4]],axis=1, sort=False)
        
    data.columns = content
    
    data1=data.dropna(how="all")
    data1=data1.dropna(axis='columns',how="any")
    traces = []
    for tic in stock_ticker:
        traces.append({'x':data1.index, 'y': data1[f"{tic}"], 'name':tic})

    fig = {
        'data': traces,
        'layout': {'title':', '.join(stock_ticker)+' Closing Prices for past 5 years'}
    }
    return fig


@app.callback(
    Output('my_graph2', 'figure'),
    [Input('submit-button2', 'n_clicks')],
    [State('my_ticker_symbol', 'value'),
    State('my_date_picker', 'value'),
    State('my_risk','value')])

def update_graph1(n_clicks, stock_ticker, lookforward_period,risk):
    exp_ret=pd.Series()
    content1=list(stock_ticker)
    data3=pd.DataFrame()
    for contents in content1:
        data3=pd.concat([data3,yf.download(f"{contents}", start="2015-01-01", end="2020-01-01").iloc[:,4]],axis=1, sort=False)
    data3.columns = content1
    
    data4=data3.dropna(how="all")
    data4=data4.dropna(axis='columns',how="any")
    
    cumulative_ret_data=pd.DataFrame()
    for contents in content1:
        cumulative_ret_data[f"{contents}"] = (1+(data4[f"{contents}"]).pct_change()).cumprod() 
    cumulative_ret_data=cumulative_ret_data.fillna(1)
    S = risk_models.sample_cov(data4)

    # Portfolio optimization based on lookforward period and risk preference
    if lookforward_period == 2:
        frequency = 500
    elif lookforward_period == 3:
        frequency = 750
    else:
        frequency = 250  # Default frequency for other cases

    # Calculate expected returns with the specified frequency
    exp_ret = pypfopt.expected_returns.mean_historical_return(data4, frequency=frequency)

    # Initialize Efficient Frontier with the expected returns and a given covariance matrix
    ef = EfficientFrontier(exp_ret, S, weight_bounds=(-1, 1))

    # Optimize portfolio based on risk preference
    if risk == 2:
        weights_ef = ef.max_sharpe(risk_free_rate=0.02)
    elif risk == 3:
        weights_ef = ef.max_quadratic_utility(risk_aversion=0.00001, market_neutral=False)
    else:
        weights_ef = ef.min_volatility()
        
    dictlist=[]
    for key, value in weights_ef.items():
        temp = [key,value]
        dictlist.append(temp)
            
    weights=pd.DataFrame(dictlist,columns=['ticker','weight']).set_index('ticker') 
    weights=weights.iloc[:,0]
    weights=weights.sort_index()
    weight_list=weights.tolist()
        
    
    HRP_df=weights.multiply(weights,weight_list) 
    HRP_cumu=cumulative_ret_data.reindex(sorted(cumulative_ret_data.columns), axis=1) #sort column by company names  
    HRP_df=HRP_cumu.mul(weight_list, axis=1)
    HRP_df['final_returns'] = HRP_df.sum(axis=1)

    fig2 = {
        # set data equal to traces
        'data':[{'x':HRP_df.index, 'y':HRP_df["final_returns"] , 'name':tic}] ,
        # use string formatting to include all symbols in the chart title
        'layout': {'title':', '.join(stock_ticker)+' portfolio'}
    }
    return fig2


if __name__ == '__main__':
    app.run_server()
