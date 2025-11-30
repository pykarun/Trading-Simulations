"""Alpaca live trader wrapper for the existing strategy logic.

This script re-uses the project's indicator functions (from `simulation.core`) to
generate trading signals and executes market orders on Alpaca Paper trading API.

How it works:
- Historical data: fetched from Yahoo (yfinance) as in the project
- Current price / orders / account info: fetched via Alpaca REST API
- Strategy parameters are defined in the `STRATEGY` constant (JSON-like dict)
- The script runs in a loop every `LOOP_INTERVAL_MINUTES` and will retry failed
  orders up to `ORDER_RETRY_COUNT` times.

Notes / prerequisites:
- You must set `ALPACA_API_KEY`, `ALPACA_API_SECRET` and `ALPACA_BASE_URL` below.
- The script uses the public Alpaca market-data endpoint to fetch latest trade.
- This file is a lightweight runner; it does not attempt to handle every edge
  case of a production trading system. Use with caution and test in paper mode.
"""

import sys
import os
import time
import math
import json
import datetime
from typing import Dict, Any

import requests
import yfinance as yf
import pandas as pd
import argparse
import logging

# Temporarily silence noisy third-party logger messages (e.g., Streamlit cache warning)
# by raising the root logger level to ERROR early. `setup_logging()` below will
# configure handlers and set the intended level for the application.
_OLD_ROOT_LOG_LEVEL = logging.getLogger().level
logging.getLogger().setLevel(logging.ERROR)

# Ensure local package imports work (project's `simulation` package)
sys.path.insert(0, os.path.abspath('simulation'))
from core.indicators import (
    calculate_ema, calculate_double_ema, calculate_rsi,
    calculate_bollinger_bands, calculate_atr, calculate_msl_msh,
    calculate_macd, calculate_adx, calculate_supertrend
)
from core.stoploss import compute_stop_state

# Logging
sys.path.insert(0, os.path.abspath('.'))
from utils.logging import setup_logging, get_logger
setup_logging()
logger = get_logger(__name__)

# Silence Streamlit cache-data runtime warning when running non-UI scripts
# (some project modules import `streamlit` for caching; in non-Streamlit runtime
# this emits a warning `No runtime found, using MemoryCacheStorageManager`).
try:
    logging.getLogger('streamlit.runtime.caching.cache_data_api').setLevel(logging.ERROR)
except Exception:
    pass


# -------------------------- USER-CONFIGURABLE CONSTANTS ---------------------
# Load secrets and configuration from a local file.
# You must create `alpaca_secrets.py` in the same directory as this script.
# Example `alpaca_secrets.py`:
# ALPACA_API_KEY_ID = "YOUR_ALPACA_API_KEY_ID"
# ALPACA_SECRET_KEY = "YOUR_ALPACA_SECRET_KEY"
# ALPACA_PAPER_TRADING = True
# ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
# ALPACA_DATA_URL = "https://data.alpaca.markets"

import alpaca_secrets
_secrets_config = {
    'API_KEY': getattr(alpaca_secrets, 'ALPACA_API_KEY_ID', None),
    'SECRET_KEY': getattr(alpaca_secrets, 'ALPACA_SECRET_KEY', None),
    'PAPER_TRADING': getattr(alpaca_secrets, 'ALPACA_PAPER_TRADING', None),
    'BASE_URL': getattr(alpaca_secrets, 'ALPACA_BASE_URL', None),
    'DATA_URL': getattr(alpaca_secrets, 'ALPACA_DATA_URL', None),
}

# Alpaca credentials (paper trading) - loaded from alpaca_secrets.py
ALPACA_API_KEY = _secrets_config.get('API_KEY')
ALPACA_API_SECRET = _secrets_config.get('SECRET_KEY')

ALPACA_PAPER_TRADING = _secrets_config.get('PAPER_TRADING')
if ALPACA_PAPER_TRADING is None:
    ALPACA_PAPER_TRADING = True # Default to True if not provided.

ALPACA_BASE_URL = _secrets_config.get('BASE_URL', 'https://paper-api.alpaca.markets')
ALPACA_DATA_URL = _secrets_config.get('DATA_URL', 'https://data.alpaca.markets')


# Trading loop interval (minutes)
LOOP_INTERVAL_MINUTES = 1

# Order retry policy
ORDER_RETRY_COUNT = 3
ORDER_RETRY_DELAY = 5  # seconds between retries

# Symbol configuration
SYMBOL_QQQ = "QQQ"
SYMBOL_TQQQ = "TQQQ"

# Allocation: fraction of available cash to use when buying (0.0-1.0)
ALLOCATION_FRACTION = 1.0

# Final strategy JSON (example). Modify these parameters to tune trading.
STRATEGY: Dict[str, Any] = None

# Attempt to load strategy from external JSON file
_STRATEGY_CONFIG_PATH = 'simulation/config/strategy_config.json'
if os.path.exists(_STRATEGY_CONFIG_PATH):
    try:
        with open(_STRATEGY_CONFIG_PATH, 'r') as f:
            external_strategy = json.load(f)

        # Validate loaded JSON is a mapping/dict
        if not isinstance(external_strategy, dict):
            logger.warning('Strategy file %s did not contain a JSON object (got %s); ignoring external config.', _STRATEGY_CONFIG_PATH, type(external_strategy).__name__)
        else:
            # Assign external strategy directly — the user manages the config file.
            STRATEGY = external_strategy
            logger.info('Strategy loaded from %s', _STRATEGY_CONFIG_PATH)

            # Log enabled boolean flags and key params as compact JSON
            try:
                enabled_details = {}
                for k, v in STRATEGY.items():
                    if isinstance(v, bool) and v:
                        details = {'enabled': True, 'params': {}}
                        if k in ('use_ema', 'use_double_ema'):
                            details['params']['ema_period'] = STRATEGY.get('ema_period')
                            details['params']['ema_fast'] = STRATEGY.get('ema_fast')
                            details['params']['ema_slow'] = STRATEGY.get('ema_slow')
                        # Only attach stop-loss details when the stop-loss feature
                        # itself is the enabled boolean being described.
                        if k == 'use_stop_loss':
                            details['params']['stop_loss_pct'] = STRATEGY.get('stop_loss_pct')
                        # ATR-specific details only when ATR flag is the enabled key
                        if k == 'use_atr':
                            details['params']['atr_multiplier'] = STRATEGY.get('atr_multiplier')
                            details['params']['atr_period'] = STRATEGY.get('atr_period')
                        # MSL/MSH-specific details only when that flag is enabled
                        if k == 'use_msl_msh':
                            details['params']['msl_period'] = STRATEGY.get('msl_period')
                            details['params']['msl_lookback'] = STRATEGY.get('msl_lookback')
                        if k == 'use_bb':
                            details['params']['bb_period'] = STRATEGY.get('bb_period')
                            details['params']['bb_std_dev'] = STRATEGY.get('bb_std_dev')
                        if k == 'use_macd':
                            details['params']['macd_fast'] = STRATEGY.get('macd_fast')
                            details['params']['macd_slow'] = STRATEGY.get('macd_slow')
                            details['params']['macd_signal_period'] = STRATEGY.get('macd_signal_period')
                        if k == 'use_adx':
                            details['params']['adx_period'] = STRATEGY.get('adx_period')
                            details['params']['adx_threshold'] = STRATEGY.get('adx_threshold')
                        if k == 'use_supertrend':
                            details['params']['st_period'] = STRATEGY.get('st_period')
                            details['params']['st_multiplier'] = STRATEGY.get('st_multiplier')

                        enabled_details[k] = details

                if enabled_details:
                    compact = json.dumps(enabled_details, separators=(',', ':'), sort_keys=True)
                    logger.info('Enabled strategy flags and key params: %s', compact)
                else:
                    logger.info('No boolean strategy flags enabled in loaded config.')
            except Exception as _e:
                logger.debug('Could not compute enabled strategy flags/details: %s', _e)
    except Exception as e:
        logger.warning('Could not load strategy from %s. Error: %s', _STRATEGY_CONFIG_PATH, e)
else:
    logger.info('No external strategy config found at %s, using hardcoded defaults.', _STRATEGY_CONFIG_PATH)


# -------------------------- Alpaca API helpers -------------------------------
def _alpaca_headers():
    return {
        'APCA-API-KEY-ID': ALPACA_API_KEY,
        'APCA-API-SECRET-KEY': ALPACA_API_SECRET
    }


def get_latest_trade_from_alpaca(symbol: str) -> float:
    """Fetch latest trade price for `symbol` from Alpaca data endpoint.

    Falls back to yfinance if Alpaca data is unavailable.
    """
    try:
        url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/trades/latest"
        resp = requests.get(url, headers=_alpaca_headers(), timeout=10)
        resp.raise_for_status()
        j = resp.json()
        # Response shape: {"trade": {"p": price, ...}}
        price = j.get('trade', {}).get('p')
        if price is None:
            raise ValueError('No price in response')
        return float(price)
    except Exception:
        # Last-resort: use yfinance for current price
        try:
            ticker = yf.Ticker(symbol)
            last = ticker.history(period='1d')
            if not last.empty:
                return float(last['Close'].iloc[-1])
        except Exception:
            pass
        raise


def get_account() -> Dict[str, Any]:
    url = f"{ALPACA_BASE_URL}/v2/account"
    resp = requests.get(url, headers=_alpaca_headers(), timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_position(symbol: str) -> Dict[str, Any] | None:
    url = f"{ALPACA_BASE_URL}/v2/positions/{symbol}"
    resp = requests.get(url, headers=_alpaca_headers(), timeout=10)
    if resp.status_code == 404:
        return None
    resp.raise_for_status()
    return resp.json()


def submit_order(symbol: str, qty: int, side: str = 'buy') -> Dict[str, Any]:
    url = f"{ALPACA_BASE_URL}/v2/orders"
    payload = {
        'symbol': symbol,
        'qty': qty,
        'side': side,
        'type': 'market',
        'time_in_force': 'gtc'
    }
    resp = requests.post(url, headers=_alpaca_headers(), json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


# Runtime flags
DRY_RUN = False


def validate_alpaca_credentials(timeout: int = 10) -> tuple[bool, str]:
    """Validate Alpaca credentials by calling the account endpoint.

    Returns (True, account_json_str) on success or (False, error_message) on failure.
    """
    try:
        acct = get_account()
        return True, json.dumps({'account_id': acct.get('id'), 'status': acct.get('status')})
    except requests.exceptions.HTTPError as e:
        # Likely 401 or similar
        return False, f'HTTPError: {e} ({getattr(e.response, "status_code", "no-code")})'
    except Exception as e:
        return False, str(e)


# -------------------------- Strategy signal computation ---------------------
def compute_signal(params: Dict[str, Any], qqq: pd.DataFrame, tqqq: pd.DataFrame, current_price: float, qqq_live_price: float | None = None) -> tuple[str, pd.Series, bool]:
    """Compute BUY/SELL signal for the latest available bar using same logic as strategy.

    qqq and tqqq should be DataFrames indexed by date and contain High/Low/Close.
    current_price is the latest TQQQ market price (from Alpaca) and will be appended
    as the last row to both qqq and tqqq (same timestamp) for indicator computation.
    """
    df_q = qqq.copy()
    df_t = tqqq.copy()

    # Append current price row with today's date/time if latest index is older
    now = pd.Timestamp(datetime.datetime.now()).normalize()
    live_injected = False
    if df_q.index[-1] < now:
        last_row_q = df_q.iloc[-1].copy()
        # If a live QQQ price was provided, use it for the last row so EMA
        # calculations reflect the most up-to-date market price. Otherwise,
        # fall back to the previous behavior: use current_price when
        # `use_supertrend` is enabled, else retain the Yahoo close.
        if qqq_live_price is not None:
            last_row_q['Close'] = qqq_live_price
            live_injected = True
        else:
            last_row_q['Close'] = current_price if params.get('use_supertrend', False) else last_row_q['Close']
        df_q.loc[now] = last_row_q
        last_row_t = df_t.iloc[-1].copy()
        last_row_t['Close'] = current_price
        df_t.loc[now] = last_row_t

    # Indicators
    if params.get('use_double_ema', False):
        df_q = calculate_double_ema(df_q, params['ema_fast'], params['ema_slow'])
    elif params.get('use_ema', True):
        df_q = calculate_ema(df_q, params['ema_period'])

    df_q = calculate_rsi(df_q, period=14)
    if params.get('use_bb', False):
        df_q = calculate_bollinger_bands(df_q, params.get('bb_period', 20), params.get('bb_std_dev', 2.0))
    if params.get('use_atr', False):
        df_q = calculate_atr(df_q, params.get('atr_period', 14))
        df_t = calculate_atr(df_t, params.get('atr_period', 14))
    if params.get('use_msl_msh', False):
        df_q = calculate_msl_msh(df_q, params.get('msl_period', 20), params.get('msl_period', 20), params.get('msl_lookback', 5), params.get('msl_lookback', 5))
        df_t = calculate_msl_msh(df_t, params.get('msl_period', 20), params.get('msl_period', 20), params.get('msl_lookback', 5), params.get('msl_lookback', 5))
    if params.get('use_macd', False):
        df_q = calculate_macd(df_q, params.get('macd_fast', 12), params.get('macd_slow', 26), params.get('macd_signal_period', 9))
    if params.get('use_adx', False):
        df_q = calculate_adx(df_q, params.get('adx_period', 14))
    if params.get('use_supertrend', False):
        df_q = calculate_supertrend(df_q, period=params.get('st_period', 10), multiplier=params.get('st_multiplier', 3.0))
        df_t = calculate_supertrend(df_t, period=params.get('st_period', 10), multiplier=params.get('st_multiplier', 3.0))

    sim = df_q.iloc[-1]

    # Base signal (EMA-based)
    if params.get('use_ema', True):
        if params.get('use_double_ema', False):
            ema_fast_val = sim.get('EMA_Fast')
            ema_slow_val = sim.get('EMA_Slow')
            base_signal = 'BUY' if ema_fast_val > ema_slow_val else 'SELL'
        else:
            qqq_close = sim['Close']
            qqq_ema = sim.get('EMA')
            base_signal = 'BUY' if qqq_close > qqq_ema else 'SELL'
    else:
        base_signal = 'BUY'

    signal = base_signal

    # RSI filters
    if params.get('use_rsi', False):
        rsi = sim.get('RSI')
        if pd.notna(rsi):
            rsi_buy_signal = (rsi < params.get('rsi_oversold', 30)) or (rsi > params.get('rsi_threshold', 50) and base_signal == 'BUY')
            rsi_sell_signal = rsi > params.get('rsi_overbought', 70)
            if rsi_sell_signal:
                signal = 'SELL'
            elif base_signal == 'BUY' and rsi_buy_signal:
                signal = 'BUY'
            else:
                signal = 'SELL'

    # BB, MACD, ADX filters
    if signal == 'BUY':
        if params.get('use_bb', False):
            bb_pos = sim.get('BB_Position')
            if pd.notna(bb_pos) and bb_pos > params.get('bb_buy_threshold', 0.2):
                signal = 'SELL'

        if params.get('use_macd', False) and signal == 'BUY':
            macd_hist = sim.get('MACD_Hist')
            if pd.notna(macd_hist) and macd_hist <= 0:
                signal = 'SELL'

        if params.get('use_adx', False) and signal == 'BUY':
            adx = sim.get('ADX')
            plus_di = sim.get('+DI')
            minus_di = sim.get('-DI')
            if pd.notna(adx) and pd.notna(plus_di) and pd.notna(minus_di):
                if adx < params.get('adx_threshold', 25) or plus_di < minus_di:
                    signal = 'SELL'

    # Supertrend filter: require ST_dir == 1
    if params.get('use_supertrend', False) and signal == 'BUY':
        st_dir = sim.get('ST_dir')
        if pd.isna(st_dir) or st_dir != 1:
            signal = 'SELL'

    # Sell overrides
    if params.get('use_bb', False):
        bb_pos = sim.get('BB_Position')
        if pd.notna(bb_pos) and bb_pos >= params.get('bb_sell_threshold', 0.8):
            signal = 'SELL'

    return signal, sim, live_injected


# -------------------------- Trading loop -----------------------------------
def run_trading_loop(run_once: bool = False):
    logger.info('Starting Alpaca trading loop.')
    logger.debug('Runtime info: Python %s, pandas %s, yfinance %s', sys.version.split()[0], pd.__version__, getattr(yf, '__version__', 'unknown'))
    session = requests.Session()
    # Track peak price observed while a position is held (for trailing stop calculation)
    peak_price: float | None = None

    def _format_indicators(indicators: pd.Series, strategy_params: Dict[str, Any]) -> str:
        """Formats relevant indicator values for logging based on active strategy parameters."""
        log_parts = []
        if strategy_params.get('use_ema', True):
            if strategy_params.get('use_double_ema', False):
                ema_fast = indicators.get('EMA_Fast')
                ema_slow = indicators.get('EMA_Slow')
                if pd.notna(ema_fast) and pd.notna(ema_slow):
                    log_parts.append(f"DEMA({strategy_params['ema_fast']},{strategy_params['ema_slow']}): {ema_fast:.2f}/{ema_slow:.2f}")
            else:
                ema = indicators.get('EMA')
                if pd.notna(ema):
                    log_parts.append(f"EMA({strategy_params['ema_period']}): {ema:.2f}")
        
        rsi = indicators.get('RSI')
        if pd.notna(rsi):
            log_parts.append(f"RSI: {rsi:.2f}")
        
        if strategy_params.get('use_bb', False):
            bb_upper = indicators.get('BB_Upper')
            bb_lower = indicators.get('BB_Lower')
            bb_mid = indicators.get('BB_Mid')
            bb_pos = indicators.get('BB_Position')
            if all(pd.notna(v) for v in [bb_upper, bb_lower, bb_mid, bb_pos]):
                log_parts.append(f"BB({strategy_params['bb_period']},{strategy_params['bb_std_dev']}): U:{bb_upper:.2f} M:{bb_mid:.2f} L:{bb_lower:.2f} Pos:{bb_pos:.2f}")
        
        if strategy_params.get('use_macd', False):
            macd = indicators.get('MACD')
            signal_line = indicators.get('MACD_Signal')
            hist = indicators.get('MACD_Hist')
            if all(pd.notna(v) for v in [macd, signal_line, hist]):
                log_parts.append(f"MACD({strategy_params['macd_fast']},{strategy_params['macd_slow']},{strategy_params['macd_signal_period']}): MACD:{macd:.2f} Sig:{signal_line:.2f} Hist:{hist:.2f}")

        if strategy_params.get('use_adx', False):
            adx = indicators.get('ADX')
            plus_di = indicators.get('+DI')
            minus_di = indicators.get('-DI')
            if all(pd.notna(v) for v in [adx, plus_di, minus_di]):
                log_parts.append(f"ADX({strategy_params['adx_period']}): {adx:.2f} +DI:{plus_di:.2f} -DI:{minus_di:.2f}")
        
        if strategy_params.get('use_supertrend', False):
            st_dir = indicators.get('ST_dir')
            st_final_upper = indicators.get('ST_final_upper')
            st_final_lower = indicators.get('ST_final_lower')
            if pd.notna(st_dir) and pd.notna(st_final_upper) and pd.notna(st_final_lower):
                log_parts.append(f"Supertrend({strategy_params.get('st_period', 10)},{strategy_params.get('st_multiplier', 3.0)}): Dir:{int(st_dir)} Upper:{st_final_upper:.2f} Lower:{st_final_lower:.2f}")

        return "; ".join(log_parts)


    def _build_signal_reason(params: Dict[str, Any], indicators: pd.Series, signal: str, current_price: float) -> str:
        """Build a human-readable reason string explaining why a BUY/SELL/SELL override was chosen.

        The string will include: which strategy produced the base signal, the price used,
        the primary indicator values and comparisons, and any filters that overrode the base
        decision (e.g., RSI, BB, MACD, ADX, Supertrend).
        """
        parts = []
        # Price context
        parts.append(f"Price=${current_price:.2f}")

        # EMA logic
        try:
            if params.get('use_double_ema', False):
                ef = indicators.get('EMA_Fast')
                es = indicators.get('EMA_Slow')
                if pd.notna(ef) and pd.notna(es):
                    parts.append(f"DEMA: Fast={ef:.2f} {'>' if ef>es else '<='} Slow={es:.2f}")
            elif params.get('use_ema', True):
                ema = indicators.get('EMA')
                close = indicators.get('Close')
                if pd.notna(ema) and pd.notna(close):
                    parts.append(f"EMA({params.get('ema_period')}): EMA={ema:.2f} Close={close:.2f} ({'>' if close>ema else '<='})")
        except Exception:
            pass

        # RSI
        try:
            if params.get('use_rsi', False):
                rsi = indicators.get('RSI')
                if pd.notna(rsi):
                    parts.append(f"RSI={rsi:.2f} (oversold<{params.get('rsi_oversold',30)} / thresh>{params.get('rsi_threshold',50)} / overbought>{params.get('rsi_overbought',70)})")
        except Exception:
            pass

        # Bollinger Bands
        try:
            if params.get('use_bb', False):
                bbp = indicators.get('BB_Position')
                if pd.notna(bbp):
                    parts.append(f"BB_Pos={bbp:.2f} (buy_th={params.get('bb_buy_threshold',0.2)} sell_th={params.get('bb_sell_threshold',0.8)})")
        except Exception:
            pass

        # MACD
        try:
            if params.get('use_macd', False):
                macd_hist = indicators.get('MACD_Hist')
                if pd.notna(macd_hist):
                    parts.append(f"MACD_Hist={macd_hist:.2f}")
        except Exception:
            pass

        # ADX
        try:
            if params.get('use_adx', False):
                adx = indicators.get('ADX')
                pdi = indicators.get('+DI')
                mdi = indicators.get('-DI')
                if pd.notna(adx):
                    parts.append(f"ADX={adx:.2f} (+DI={pdi:.2f} -DI={mdi:.2f}) thresh={params.get('adx_threshold',25)}")
        except Exception:
            pass

        # Supertrend
        try:
            if params.get('use_supertrend', False):
                st_dir = indicators.get('ST_dir')
                parts.append(f"Supertrend_dir={int(st_dir) if not pd.isna(st_dir) else 'NA'}")
        except Exception:
            pass

        # Stop-loss info when applicable
        try:
            if any([params.get('use_stop_loss', False), params.get('use_atr', False), params.get('use_msl_msh', False)]):
                if params.get('use_stop_loss', False):
                    parts.append(f"StopLoss: type=percentage pct={params.get('stop_loss_pct',0)}")
                if params.get('use_atr', False):
                    atr = indicators.get('ATR')
                    parts.append(f"StopLoss: type=ATR atr={atr:.2f} mult={params.get('atr_multiplier',2.0)}" if pd.notna(atr) else "StopLoss: type=ATR atr=NA")
                if params.get('use_msl_msh', False):
                    parts.append("StopLoss: type=MSL/MSH")
        except Exception:
            pass

        # Final action summary
        parts.append(f"Action={signal}")

        return ' | '.join(parts)


    def _format_params_used(params: Dict[str, Any], indicators: pd.Series, entry_price: float | None = None) -> str:
        """Return a compact string describing key parameter values used for the decision.

        Includes EMA configuration/values and Stop-Loss configuration (percentage / ATR multiplier / MSL/MSH).
        This string is intended for inclusion in action logs so the operator can quickly see
        which numeric values were applied.
        """
        parts = []
        try:
            if params.get('use_double_ema', False):
                ef = indicators.get('EMA_Fast')
                es = indicators.get('EMA_Slow')
                parts.append(f"DEMA_periods={params.get('ema_fast')}/{params.get('ema_slow')}")
                if pd.notna(ef) and pd.notna(es):
                    parts.append(f"DEMA_vals={ef:.2f}/{es:.2f}")
            elif params.get('use_ema', True):
                parts.append(f"EMA_period={params.get('ema_period')}")
                ema_val = indicators.get('EMA')
                if pd.notna(ema_val):
                    parts.append(f"EMA_val={ema_val:.2f}")
        except Exception:
            pass

        try:
            # Stop-loss configuration
            if params.get('use_stop_loss', False):
                parts.append(f"SL_pct={params.get('stop_loss_pct', 0):.2f}%")
            if params.get('use_atr', False):
                atr_val = indicators.get('ATR')
                parts.append(f"SL_ATR_mult={params.get('atr_multiplier', 2.0)}")
                if pd.notna(atr_val):
                    parts.append(f"ATR_val={atr_val:.2f}")
                else:
                    parts.append("ATR_val=NA")
            if params.get('use_msl_msh', False):
                parts.append("SL=MSL/MSH")
        except Exception:
            pass

        if entry_price is not None:
            try:
                parts.append(f"Entry=${entry_price:.2f}")
            except Exception:
                pass

        return '; '.join(parts) if parts else 'Params:N/A'

    while True:
        try:
            logger.debug('Beginning loop iteration at %s', datetime.datetime.now().isoformat())
            # Fetch historical price history (1+ year to ensure indicators)
            end = datetime.date.today()
            start = end - datetime.timedelta(days=730)
            data = yf.download([SYMBOL_QQQ, SYMBOL_TQQQ], start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), group_by='ticker', progress=False)

            if data.empty:
                logger.warning('Failed to download historical data; retrying next interval')
                time.sleep(60)
                continue

            # Build DataFrames
            qqq = data[SYMBOL_QQQ].rename_axis('Date')
            tqqq = data[SYMBOL_TQQQ].rename_axis('Date')

            # Get current market price from Alpaca
            try:
                current_price = get_latest_trade_from_alpaca(SYMBOL_TQQQ)
            except Exception as e:
                logger.warning('Failed to fetch current price from Alpaca, using last close; error: %s', e)
                current_price = float(tqqq['Close'].iloc[-1])

            # Also fetch live QQQ price so we can inject it into the QQQ series
            # for EMA calculations (keeps EMA up-to-date with market).
            try:
                qqq_live_price = get_latest_trade_from_alpaca(SYMBOL_QQQ)
            except Exception:
                qqq_live_price = float(qqq['Close'].iloc[-1])

            # Get account and position info
            account = get_account()
            cash = float(account.get('cash', 0))
            pos = get_position(SYMBOL_TQQQ)
            has_position = pos is not None
            current_shares = int(float(pos.get('qty', 0))) if has_position else 0
            position_value = current_shares * current_price if has_position else 0.0
            equity = float(account.get('equity', 0))

            logger.info('*** Portfolio Snapshot ***')
            logger.info('Account Equity: $%.2f | Cash: $%.2f', equity, cash)
            if has_position:
                logger.info('Position in %s: %d shares | Current Value: $%.2f', SYMBOL_TQQQ, current_shares, position_value)
            else:
                logger.info('No current position in %s', SYMBOL_TQQQ)


            signal, indicators, qqq_live_used = compute_signal(STRATEGY, qqq, tqqq, current_price, qqq_live_price=qqq_live_price)
            indicator_summary = _format_indicators(indicators, STRATEGY)

            # Compute current stop-loss state (if we have a position) so we can
            # show the SL value in the Analysis output even when we are not
            # executing a sell. We'll compute once and reuse for enforcement.
            stop_state = None
            sl_summary = 'No position'
            if has_position and any([STRATEGY.get('use_stop_loss', False), STRATEGY.get('use_atr', False), STRATEGY.get('use_msl_msh', False)]):
                try:
                    entry_price_local = float(pos.get('avg_entry_price', current_price)) if has_position else None
                    try:
                        tqqq_last_row = tqqq.iloc[-1]
                    except Exception:
                        tqqq_last_row = None
                    stop_state = compute_stop_state(STRATEGY, indicators, entry_price_local, peak_price, current_price, tqqq_row=tqqq_last_row)
                    sp = stop_state.get('stop_price')
                    method = stop_state.get('method')
                    if sp is not None:
                        sl_summary = f"{method.upper()}:${sp:.2f}"
                    else:
                        sl_summary = f"{method.upper() if method else 'SL'}:N/A"
                except Exception:
                    stop_state = None
                    sl_summary = 'SL:Error'

            logger.info('Analysis (%s): Current Price: $%.2f | Indicators: %s | SL: %s', datetime.datetime.now().isoformat(), current_price, indicator_summary, sl_summary)
            logger.info('Suggested Action (Signal): %s', signal)

            # Log whether we injected a live QQQ price into the indicator calculation
            try:
                if STRATEGY.get('use_double_ema', False):
                    ef = indicators.get('EMA_Fast')
                    es = indicators.get('EMA_Slow')
                    ema_str = f"DEMA_vals={ef:.2f}/{es:.2f}" if pd.notna(ef) and pd.notna(es) else 'DEMA_vals=NA'
                elif STRATEGY.get('use_ema', True):
                    ema_val = indicators.get('EMA')
                    ema_str = f"EMA_val={ema_val:.2f}" if pd.notna(ema_val) else 'EMA_val=NA'
                else:
                    ema_str = 'EMA:disabled'
                logger.info('QQQ Live Injected: %s | %s', bool(qqq_live_used), ema_str)
            except Exception:
                logger.debug('Failed to log EMA/live-injection info')

            # Build and log a human-readable decision reason that captures the
            # strategy check, relevant indicator values, the comparisons made, and
            # the resulting action suggestion (BUY/SELL/HOLD).
            try:
                reason = _build_signal_reason(STRATEGY, indicators, signal, current_price)
                logger.info('Decision Reason: %s', reason)
            except Exception as _e:
                reason = '<reason unavailable>'
                logger.debug('Failed to build detailed decision reason: %s', _e)

            # Prepare a compact params summary string (EMA values, SL config, ATR, entry price)
            entry_price_local = float(pos.get('avg_entry_price', current_price)) if has_position else None
            params_str = _format_params_used(STRATEGY, indicators, entry_price_local)

            # Use previously computed stop_state (if any) for logging/enforcement
            if stop_state is not None:
                # Update tracked peak price
                peak_price = stop_state.get('peak_price', peak_price)

                # Log the computed stop info
                if stop_state.get('method') == 'percentage' and stop_state.get('stop_price') is not None:
                    pct = STRATEGY.get('stop_loss_pct', 0.0)
                    logger.info('Calculated Trailing Stop Loss (Percentage-based): Entry: $%.2f, Peak: $%.2f, SL %%: %.2f%%, SL Price: $%.2f', entry_price_local, peak_price, pct, stop_state['stop_price'])
                elif stop_state.get('method') == 'atr' and stop_state.get('stop_price') is not None:
                    logger.info('Calculated Stop Loss (ATR-based): %s', stop_state.get('reason', 'ATR-based'))
                elif stop_state.get('method') == 'msl':
                    logger.info('MSL/MSH Stop Loss is active; actual stop price determined by indicator during trading.')
                else:
                    logger.debug('Stop loss state: %s', stop_state.get('reason'))

                # If a stop price was calculated, check for breach and force SELL
                try:
                    slp = stop_state.get('stop_price')
                    if slp is not None and current_price <= slp:
                        logger.info('Stop-loss triggered: current_price $%.2f <= stop_loss_price $%.2f. Forcing SELL.', current_price, slp)
                        signal = 'SELL'
                        reason = (reason if reason else '') + f' | StopLossTriggered(curr={current_price:.2f},sl={slp:.2f})'
                except Exception:
                    logger.debug('Error while evaluating stop-loss trigger.')
            elif not any([STRATEGY.get('use_stop_loss', False), STRATEGY.get('use_atr', False), STRATEGY.get('use_msl_msh', False)]) and has_position:
                logger.info('Stop Loss calculation skipped: No stop-loss strategy enabled.')
            elif has_position:
                logger.info('Stop Loss calculation skipped: No current position in %s.', SYMBOL_TQQQ)

            # When position has been closed, reset peak_price so it starts fresh on next entry
            if not has_position:
                peak_price = None

            if signal == 'BUY':
                budget = cash * ALLOCATION_FRACTION
                if budget < 1:
                    logger.info('Actual Action: No BUY (reason: Insufficient cash, budget < $1.00) | Params: %s | Reason: %s', params_str, reason)
                else:
                    qty = math.floor(budget / current_price)
                    if qty <= 0:
                        logger.info('Actual Action: No BUY (reason: Calculated quantity is 0) | Params: %s | Reason: %s', params_str, reason)
                    else:
                        action_type = "initial BUY" if not has_position else "additional BUY"
                        logger.info('Actual Action: Attempting %s order for %d shares of %s (Using $%.2f of cash) | Params: %s | Reason: %s', action_type, qty, SYMBOL_TQQQ, budget, params_str, reason)

                        if DRY_RUN:
                            logger.info('Dry-run: Would have placed BUY order for %d shares of %s at market | Params: %s', qty, SYMBOL_TQQQ, params_str)
                            # success = True # Already implied by dry-run
                        else:
                            success = False
                            for attempt in range(1, ORDER_RETRY_COUNT + 1):
                                try:
                                    order = submit_order(SYMBOL_TQQQ, qty, side='buy')
                                    logger.info('Order submitted: %s (Order ID: %s)', action_type, order.get('id'))
                                    success = True
                                    break
                                except Exception as e:
                                    logger.error('Order attempt %d failed: %s', attempt, e)
                                    time.sleep(ORDER_RETRY_DELAY)
                            if not success:
                                logger.error('All BUY order attempts failed; will retry next loop')

            elif signal == 'SELL' and has_position:
                try:
                    shares_to_sell = int(float(pos.get('qty', 0)))
                except Exception:
                    shares_to_sell = 0 # Should not happen if has_position is True and pos is not None

                if shares_to_sell <= 0:
                    logger.warning('Actual Action: No SELL (reason: Position reported but quantity <= 0, has_position: %s) | Params: %s', has_position, params_str)
                else:
                    logger.info('Actual Action: Attempting SELL order for %d shares of %s | Params: %s | Reason: %s', shares_to_sell, SYMBOL_TQQQ, params_str, reason)
                    if DRY_RUN:
                        logger.info('Dry-run: Would have placed SELL order for %d shares of %s at market | Params: %s', shares_to_sell, SYMBOL_TQQQ, params_str)
                        # success = True
                    else:
                        success = False
                        for attempt in range(1, ORDER_RETRY_COUNT + 1):
                            try:
                                order = submit_order(SYMBOL_TQQQ, shares_to_sell, side='sell')
                                logger.info('Sell order submitted (Order ID: %s)', order.get('id'))
                                success = True
                                break
                            except Exception as e:
                                logger.error('Sell attempt %d failed: %s', attempt, e)
                                time.sleep(ORDER_RETRY_DELAY)
                        if not success:
                            logger.error('All SELL order attempts failed; will retry next loop')

            elif signal == 'SELL' and not has_position:
                logger.info('Actual Action: No SELL (reason: Signal is SELL but no existing position in %s) | Params: %s | Reason: %s', SYMBOL_TQQQ, params_str, reason)
            else: # This covers cases where signal is not BUY/SELL or other conditions aren't met
                logger.info('Actual Action: No action (reason: Signal %s or other conditions not met) | Params: %s | Reason: %s', signal, params_str, reason)


        except Exception as e:
            logger.exception('Unexpected error in trading loop: %s', e)

        # End of normal iteration
        logger.info('Loop iteration complete.')
        if run_once:
            logger.info('Run-once mode enabled: exiting after single iteration.')
            return

        # Sleep until next interval
        logger.info('Waiting %s minutes until next trading check...', LOOP_INTERVAL_MINUTES)
        time.sleep(LOOP_INTERVAL_MINUTES * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Alpaca live trader runner')
    parser.add_argument('--dry-run', action='store_true', help='Run without submitting orders')
    parser.add_argument('--validate-only', action='store_true', help='Validate Alpaca credentials and exit')
    parser.add_argument('--once', action='store_true', help='Run a single loop iteration and exit')
    args = parser.parse_args()

    DRY_RUN = bool(args.dry_run)

    # Basic presence check for credentials
    if not ALPACA_API_KEY or not ALPACA_API_SECRET:
        logger.error('Alpaca API key/secret not found. Set environment variables and retry.')
        raise SystemExit(1)

    # If user only wants to validate credentials, perform check and exit
    if args.validate_only:
        ok, msg = validate_alpaca_credentials()
        if ok:
            logger.info('Alpaca credential validation succeeded: %s', msg)
            print('OK', msg)
            raise SystemExit(0)
        else:
            logger.error('Alpaca credential validation failed: %s', msg)
            print('ERROR', msg)
            raise SystemExit(2)

    # Validate credentials upfront (unless dry-run)
    if not DRY_RUN:
        ok, msg = validate_alpaca_credentials()
        if not ok:
            logger.error('Alpaca credential validation failed at startup: %s', msg)
            raise SystemExit(1)

    run_trading_loop(run_once=bool(args.once))
