"""Manage persistent local 30m intraday data repository."""
import pandas as pd
import yfinance as yf
from pathlib import Path
from curl_cffi import requests
import os


DATA_DIR = Path(__file__).parent.parent / "data_30m"


def ensure_data_dir():
    """Create data directory if it doesn't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_data_file_path(ticker):
    """Get local CSV path for a ticker's 30m data."""
    ensure_data_dir()
    return DATA_DIR / f"{ticker}_30m.csv"


def load_local_data(ticker):
    """Load 30m data from local CSV, return empty DataFrame if not found."""
    filepath = get_data_file_path(ticker)
    if filepath.exists():
        df = pd.read_csv(filepath, index_col='Date', parse_dates=['Date'])
        df.index = pd.to_datetime(df.index, utc=True)
        return df
    return pd.DataFrame()


def save_local_data(ticker, df):
    """Save 30m data to local CSV."""
    ensure_data_dir()
    filepath = get_data_file_path(ticker)
    df_copy = df.copy()
    df_copy.index.name = 'Date'
    df_copy.to_csv(filepath)


def fetch_30m_from_yahoo(ticker, start_date='2010-01-01'):
    """Fetch 30m data from yfinance (limited to recent ~60 days)."""
    try:
        session = requests.Session(impersonate="chrome110", verify=False)
        data = yf.download(
            ticker,
            start=start_date,
            interval='30m',
            session=session,
            auto_adjust=False,
            progress=False
        )
        if isinstance(data, pd.DataFrame) and not data.empty:
            if not isinstance(data.index, pd.DatetimeIndex):
                data.index = pd.to_datetime(data.index)
            if data.index.tz is None:
                data.index = data.index.tz_localize('UTC')
            elif data.index.tz.zone != 'UTC':
                data.index = data.index.tz_convert('UTC')
        return data
    except Exception as e:
        print(f"Error fetching {ticker} from yfinance: {e}")
        return pd.DataFrame()


def append_new_data(ticker):
    """Append new 30m bars to local file since last stored timestamp."""
    local_df = load_local_data(ticker)
    
    # Fetch latest data from yfinance (covers last ~60 days)
    fresh_df = fetch_30m_from_yahoo(ticker)
    
    if fresh_df.empty:
        print(f"No fresh data from yfinance for {ticker}")
        return local_df
    
    if local_df.empty:
        # First time; save all fresh data
        save_local_data(ticker, fresh_df)
        return fresh_df
    
    # Merge: keep local data, append any new bars from fresh_df
    last_local_timestamp = local_df.index.max()
    new_bars = fresh_df[fresh_df.index > last_local_timestamp]
    
    if not new_bars.empty:
        merged = pd.concat([local_df, new_bars]).drop_duplicates()
        merged = merged.sort_index()
        save_local_data(ticker, merged)
        print(f"Appended {len(new_bars)} new 30m bars for {ticker}")
        return merged
    else:
        print(f"No new 30m bars for {ticker}")
        return local_df


def get_30m_data_range(ticker, start_date, end_date):
    """
    Get 30m data for a date range, updating local file first.
    Returns DataFrame with all 30m bars in [start_date, end_date].
    """
    # Append new data to local file
    df = append_new_data(ticker)
    
    if df.empty:
        raise ValueError(f"No 30m data available for {ticker}")
    
    # Slice to requested range (convert dates to UTC datetimes)
    start_ts = pd.Timestamp(start_date, tz='UTC')
    end_ts = pd.Timestamp(end_date, tz='UTC') + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    
    sliced = df.loc[start_ts:end_ts]
    if sliced.empty:
        raise ValueError(f"No 30m data for {ticker} in range {start_date} to {end_date}")
    
    return sliced
