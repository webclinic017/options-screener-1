import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import datetime
from wallstreet import Stock
import requests
import time
import streamlit.components.v1 as components
import SessionState  # Assuming SessionState.py lives on this folder
import os
import json
import config

@st.cache
def load_data():
    df = pd.read_pickle('master_df.pkl')
    return df

@st.cache(suppress_st_warning=True)
def load_price_data():
    with open('prices_df.pkl', 'rb') as handle:
        prices_df = pickle.load(handle)
    return prices_df

@st.cache
def load_opt_data():
    master_options_all = pd.read_pickle('master_options.pkl')
    return master_options_all

@st.cache            
def get_sentiment(ticker):
    
    r = requests.get('https://finnhub.io/api/v1/news-sentiment?symbol={}&token={}'.format(ticker, config.TOKEN))
    
    buzz = pd.DataFrame()
    sentiment_df = pd.DataFrame()
    news = pd.DataFrame()
    
    buzz.at['Media attention values', 'Media attention score'] = r.json()['buzz']['buzz']
    buzz.at['Media attention values', 'No. of articles last week'] = r.json()['buzz']['articlesInLastWeek']
    buzz.at['Media attention values', 'No. of articles per week on average'] = r.json()['buzz']['weeklyAverage']

    sentiment_df.at['Sentiment values', 'Bullish score'] = r.json()['sentiment']['bullishPercent']
    sentiment_df.at['Sentiment values', 'Bearish score'] = r.json()['sentiment']['bearishPercent']
    sentiment_df.at['Sentiment values', 'Sector average'] = r.json()['sectorAverageBullishPercent']

    news.at['News values', 'Company news score'] = r.json()['companyNewsScore']
    news.at['News values', 'Sector average'] = r.json()['sectorAverageNewsScore']

    return buzz, sentiment_df, news
    

def show_graph(ticker):

    ticker_price_data  = prices_df[ticker]

    set1 = { 'x': ticker_price_data.Date, 'open': ticker_price_data.Open, 'close': ticker_price_data.Close, 'high': ticker_price_data.High, 'low': ticker_price_data.Low, 'type': 'candlestick','name':ticker}
    set2 = { 'x': ticker_price_data.Date, 'y': ticker_price_data['avg_20'], 'type': 'scatter', 'mode': 'lines', 'line': { 'width': 1.5, 'color': 'blue' },'name': 'MA 20 periods'}
    set3 = { 'x': ticker_price_data.Date, 'y': ticker_price_data['avg_50'], 'type': 'scatter', 'mode': 'lines', 'line': { 'width': 1.5, 'color': 'yellow' },'name': 'MA 50 periods'}
    set4 = { 'x': ticker_price_data.Date, 'y': ticker_price_data['avg_200'], 'type': 'scatter', 'mode': 'lines', 'line': { 'width': 1.5, 'color': 'black' },'name': 'MA 200 periods'}
    
    data = [set1, set2, set3, set4]
    
    fig = go.Figure(data=data)
    
    st.plotly_chart(fig)
             

session = SessionState.get(run_id=0)


st.write("""
         # Options Screener Pipeline
         ## Stocks
 
         **Step 1**: Use the Stock Parameters tools in the sidebar to screen stocks.
 
         The most actively traded stocks are shown below:
 
         """)

    
st.sidebar.title('Step 1: Stock Parameters')

df = load_data()

master_options_all = load_opt_data()

if st.sidebar.button("Reset Parameters"):
      session.run_id += 1

volume = st.sidebar.slider('20-Day Average Volume (default: 1,000,000)', 
                           int(df['Vol_20D'].min()), 
                           int(df['Vol_20D'].max()), 
                           10000000, 
                           key = session.run_id)

min_price = st.sidebar.number_input('Minimum price',value=10,key = session.run_id)

max_price = st.sidebar.number_input('Maximum price', value=100.,key = session.run_id)

eps = st.sidebar.radio('Earnings Per Share (EPS)', ['All', 'Positive', 'Negative'], key = session.run_id)

eps_mapper = {'Positive' : 'Yes', 
              'Negative' : 'No' ,
              'All' : ''}

div = st.sidebar.radio('Forward Dividend Yield', ['All', 'Positive yield','No information'],key = session.run_id)

div_mapper = {'All' : '', 
              'No information' : 'No',
              'Positive yield' : 'Yes'}

rsi = st.sidebar.slider('20-Day Relative Strength Index (RSI)', 
                        int(df['RSI_20D'].min()), 
                        int(df['RSI_20D'].max()), 
                        (int(df['RSI_20D'].min()),int(df['RSI_20D'].max())),
                        key = session.run_id)

beta = st.sidebar.slider('Beta', df['beta'].min(), df['beta'].max(), (float(df['beta'].min()),2.), key = session.run_id)

sp500 = st.sidebar.radio('S&P500', ['All','In S&P500','Not in S&P500'],index=0, key = session.run_id)

sp500_mapper = {'In S&P500': (not True), 
                'Not in S&P500': (not False), 
                'All': 'nan'}

st.sidebar.write('Beta Features')

incl_targetprice = st.sidebar.radio('Filter for only stocks having a price below "1y Target Est" on Yahoo Finance?',
                                   ['No', 'Yes'], key = session.run_id)

if incl_targetprice == 'Yes':
    target_price_filter = df['Price'] <=  df['target_est'] 
else:
    target_price_filter = df['Price'] >=  df['Price']

incl_myprice = st.sidebar.radio("Filter for only stocks having a price below 'myprice'? ('myprice' is calculated using the stock's sustainable growth rate and a dividend discount model)", 
                                ['No', 'Yes'], key = session.run_id)

if incl_myprice == 'Yes':
    my_price_filter = df['Price'] <=  df['myprice'] 
else:
    my_price_filter = df['Price'] >=  df['Price']

cols = ['Name', 'Price', 'RSI_20D', 'Std_Dev_20D', 'Vol_20D',
       'Low_52W', 'High_52W', 'beta','EPS',
       'forward dividend', 'earnings_date',
       'sp500', 'target_est', 'myprice']

df_filtered = df[cols][
                       (df['RSI_20D'] >= rsi[0]) & (df['RSI_20D'] <= rsi[1]) &
                       (df['Vol_20D'] >= volume) &
                       (df['Price'] >= min_price) & (df['Price'] <= max_price) & 
                       (df['beta'] >= beta[0]) & (df['beta'] <= beta[1]) &
                       (df['pos_eps'].str.contains(eps_mapper[eps])) &
                       (df['pos_div'].str.contains(div_mapper[div])) &
                       (df['sp500'] != sp500_mapper[sp500]) &
                       my_price_filter &
                       target_price_filter
                       ]

st.dataframe(df_filtered)

st.write("Stocks loaded:", len(df_filtered), "/", len(df))
#st.write('Last updated:', updated_time,' GMT')


st.header('Options')

selected_tickers = st.multiselect('Selected Tickers', 
                                  df_filtered.index.tolist(), 
                                  df_filtered.index.tolist(), 
                                  key = session.run_id)

st.sidebar.title('Step 2: Options Parameters')

st.write("**Step 2**: In the table below you will find the options of the stocks screened in Step 1. Use the Options Parameters tools in the sidebar to screen the options.")

selected_tickers_ = [t for t in selected_tickers if t in master_options_all.index.unique()]

master_options = master_options_all.loc[selected_tickers_,:]

exp_dates = master_options['Expiration Date'].unique()

exp_date = st.selectbox('Choose expiration date', exp_dates, key = session.run_id)

iv = st.sidebar.selectbox('Implied Volatility over: ',(0.0, 0.2, 0.3, 0.5), key = session.run_id)

delta = st.sidebar.slider('Delta', 
                          master_options['Delta'].min(), 
                          master_options['Delta'].max(), 
                          (-.3, 0.3),
                          key = session.run_id)

vol = st.sidebar.number_input('Volume over:', 0, value=1, key = session.run_id) 

premium = st.sidebar.number_input('Premium over:', 0.,value=0., key = session.run_id)

price_strike = st.sidebar.selectbox('Price to Strike over: ', (0.0, 0.01, 0.02, 0.03, 0.04, 0.05), key = session.run_id)

price = st.sidebar.slider('Strike', 
                          master_options['Strike'].min(), 
                          master_options['Strike'].max(), 
                          (float(master_options['Strike'].min()), float(master_options['Strike'].max())),
                          key = session.run_id)

opt_type = st.selectbox('Choose option type', ['put', 'call'], key = session.run_id)

opts_filtered = master_options[(master_options['Expiration Date'] == exp_date) &
                                (master_options['IV'] >= iv) &
                                (master_options['Vol'] >= vol) &
                                (master_options['Delta'] >= delta[0]) &
                                (master_options['Delta'] <= delta[1]) &
                                (master_options['PriceToStrike'] >= price_strike) &
                                (master_options['Last Sale'] >= premium) &
                                (master_options['Strike'] >= price[0]) &
                                (master_options['Strike'] <= price[1]) &
                                (master_options['type'] == opt_type)
                               ]
cols2 = ['Strike', 'Last Sale', 'Vol', 
         'Open Int','IV', 'Delta', 'Gamma', 
         'PriceToStrike', 'VolOI', 'Net', 
         'Bid', 'Ask', 'Contract', 
         'Expiration Date','type']

st.dataframe(opts_filtered[cols2])

st.write('Options shown:', len(opts_filtered))


st.header('Price Chart')

st.write('**Step 3**: Refer to the Price Chart below as well as the news sentiment values shown in the sidebar.')

ticker = st.selectbox('Ticker', opts_filtered.index.sort_values().unique().tolist(),key = session.run_id)

if ticker is not None:
    
    prices_df = load_price_data()
    
    show_graph(ticker)
    
    try:
        st.sidebar.title('Step 3: News Sentiment')
        
        buzz, sentiment_df, news = get_sentiment(ticker)
        
        st.sidebar.table(buzz.T)
        
        st.sidebar.table(sentiment_df.T)
        
        st.sidebar.table(news.T)

        
        st.subheader("Recommendation")
        
        r = requests.get('https://finnhub.io/api/v1/stock/recommendation?symbol={}&token={}'.format(ticker, config.TOKEN))
        
        st.json(r.json()[0])
        

        st.subheader("Price Target")
        
        r = requests.get('https://finnhub.io/api/v1/stock/price-target?symbol={}&token={}'.format(ticker, config.TOKEN))
        
        st.json(r.json())
    
    except:
        st.sidebar.write("Not available at the moment")

else:
    "No ticker"