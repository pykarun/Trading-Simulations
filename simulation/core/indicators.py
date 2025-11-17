"""Technical indicator calculations."""
import pandas as pd


def calculate_ema(data, period):
    """Calculate EMA for given period.
    
    Args:
        data: DataFrame with 'Close' column
        period: EMA period
        
    Returns:
        DataFrame with added 'EMA' column
    """
    df = data.copy()
    df['EMA'] = df['Close'].ewm(span=period, adjust=False).mean()
    return df


def calculate_double_ema(data, fast_period, slow_period):
    """Calculate two EMAs for crossover strategy.
    
    Args:
        data: DataFrame with 'Close' column
        fast_period: Fast EMA period
        slow_period: Slow EMA period
        
    Returns:
        DataFrame with added 'EMA_Fast' and 'EMA_Slow' columns
    """
    df = data.copy()
    df['EMA_Fast'] = df['Close'].ewm(span=fast_period, adjust=False).mean()
    df['EMA_Slow'] = df['Close'].ewm(span=slow_period, adjust=False).mean()
    return df


def calculate_rsi(data, period=14):
    """Calculate RSI.
    
    Args:
        data: DataFrame with 'Close' column
        period: RSI period (default 14)
        
    Returns:
        DataFrame with added 'RSI' column
    """
    df = data.copy()
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df


def calculate_bollinger_bands(data, period=20, std_dev=2.0):
    """Calculate Bollinger Bands.
    
    Args:
        data: DataFrame with 'Close' column
        period: Moving average period
        std_dev: Number of standard deviations for bands
        
    Returns:
        DataFrame with added BB columns
    """
    df = data.copy()
    df['BB_Middle'] = df['Close'].rolling(window=period).mean()
    df['BB_Std'] = df['Close'].rolling(window=period).std()
    df['BB_Upper'] = df['BB_Middle'] + (std_dev * df['BB_Std'])
    df['BB_Lower'] = df['BB_Middle'] - (std_dev * df['BB_Std'])
    # Calculate position within bands (0 = lower band, 1 = upper band)
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    return df


def calculate_atr(data, period=14):
    """Calculate Average True Range (ATR).
    
    ATR measures market volatility.
    
    Args:
        data: DataFrame with 'High', 'Low', 'Close' columns
        period: ATR period
        
    Returns:
        DataFrame with added 'ATR' column
    """
    df = data.copy()
    
    # Calculate True Range
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    
    # Calculate ATR as moving average of True Range
    df['ATR'] = df['TR'].rolling(window=period).mean()
    
    # Clean up temporary columns
    df.drop(['H-L', 'H-PC', 'L-PC', 'TR'], axis=1, inplace=True)
    
    return df


def calculate_msl_msh(data, msl_period=20, msh_period=20, msl_lookback=5, msh_lookback=5):
    """Calculate MSL (Multi-timeframe Stop Low) and MSH (Multi-timeframe Stop High).
    
    MSL: Lowest low over lookback period, smoothed with moving average
    MSH: Highest high over lookback period, smoothed with moving average
    
    These provide dynamic support/resistance levels for stop loss.
    
    Args:
        data: DataFrame with 'High' and 'Low' columns
        msl_period: Smoothing period for MSL
        msh_period: Smoothing period for MSH
        msl_lookback: Lookback period for lowest low
        msh_lookback: Lookback period for highest high
        
    Returns:
        DataFrame with added MSL/MSH columns
    """
    df = data.copy()
    
    # Calculate rolling lowest low and highest high
    df['Lowest_Low'] = df['Low'].rolling(window=msl_lookback).min()
    df['Highest_High'] = df['High'].rolling(window=msh_lookback).max()
    
    # Smooth with moving average to create MSL and MSH
    df['MSL'] = df['Lowest_Low'].rolling(window=msl_period).mean()
    df['MSH'] = df['Highest_High'].rolling(window=msh_period).mean()
    
    # Calculate distance from current price to MSL/MSH (as percentage)
    df['MSL_Distance'] = ((df['Close'] - df['MSL']) / df['Close']) * 100
    df['MSH_Distance'] = ((df['MSH'] - df['Close']) / df['Close']) * 100
    
    return df
