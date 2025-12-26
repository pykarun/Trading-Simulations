"""Data fetching utilities."""
import yfinance as yf
import pandas as pd
from curl_cffi import requests


def get_data(tickers, start_date, end_date, buffer_days=365, interval='1d'):
    """Download data using curl_cffi session.

    This function avoids importing Streamlit at module import time so it can be
    used by non-UI scripts without triggering Streamlit runtime warnings. If
    Streamlit is available at call-time, we use its `st.cache_data` decorator
    to cache results; otherwise the function fetches data directly.

    Args:
        tickers: List of ticker symbols or single ticker string
        start_date: Start date for data
        end_date: End date for data
        buffer_days: Additional days to fetch before start_date for indicator calculation

    Returns:
        DataFrame with OHLCV data
    """

    def _fetch(tickers_inner, start_inner, end_inner, buffer_days_inner, interval_inner):
        session = requests.Session(impersonate="chrome110", verify=False)
        fetch_start_date = (pd.to_datetime(start_inner) - pd.DateOffset(days=buffer_days_inner)).strftime('%Y-%m-%d')
        data = yf.download(
            tickers_inner,
            start=fetch_start_date,
            end=end_inner,
            interval=interval_inner,
            session=session,
            auto_adjust=False,
            group_by='ticker',
            progress=False
        )
        return data

    # If Streamlit is available at runtime, use its cache decorator; otherwise
    # call the fetch function directly. Importing Streamlit here (inside the
    # function) avoids emitting Streamlit-specific warnings during module import
    # when non-UI scripts import this module.
    try:
        import streamlit as st
        cached = st.cache_data(ttl=3600)(_fetch)
        return cached(tickers, start_date, end_date, buffer_days, interval)
    except Exception:
        return _fetch(tickers, start_date, end_date, buffer_days, interval)
