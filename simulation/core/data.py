"""Data fetching utilities."""
import yfinance as yf
import pandas as pd
from curl_cffi import requests
import streamlit as st


@st.cache_data(ttl=3600)
def get_data(tickers, start_date, end_date, buffer_days=365):
    """Download data using curl_cffi session.
    
    Args:
        tickers: List of ticker symbols or single ticker string
        start_date: Start date for data
        end_date: End date for data
        buffer_days: Additional days to fetch before start_date for indicator calculation
        
    Returns:
        DataFrame with OHLCV data
    """
    session = requests.Session(impersonate="chrome110", verify=False)
    fetch_start_date = (pd.to_datetime(start_date) - pd.DateOffset(days=buffer_days)).strftime('%Y-%m-%d')
    data = yf.download(
        tickers, 
        start=fetch_start_date, 
        end=end_date, 
        session=session,
        auto_adjust=False, 
        group_by='ticker', 
        progress=False
    )
    return data
