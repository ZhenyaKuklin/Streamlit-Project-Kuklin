import streamlit as st
import yfinance as yf

st.sidebar.title('Apple cotirovki')

st.sidebar.write("""
# Simple Stock Price App
         
Shown are the stock **closing** and ***volume*** of Apple!
         
""")


tickerSymbol = 'AAPL'


tickerData = yf.Ticker(tickerSymbol)

#1 sposob
tickerDf = tickerData.history(period = '1d',
                              start = '2010-05-31', 
                              end = '2020-05-31'
                              )
#2 sposob
# tickerDf = yf.download('AAPL', start = '2010-05-31',
#                       end = '2020-05-31')

st.write("""
## Closing Price
""")

st.line_chart(tickerDf.Close)

st.write("""
## Volume Price
""")

st.line_chart(tickerDf.Volume)