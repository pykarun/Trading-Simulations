"""Minimal trader demo (double-EMA + trailing stop).

Simple, self-contained script that:
 - Uses QQQ double-EMA to generate a signal
 - Trades TQQQ (simulated or via Alpaca paper API)
 - Prints clear logs so you can verify behavior

Run examples:
    python scripts/alpaca_trader_minimal.py --once
    python scripts/alpaca_trader_minimal.py --once --live --dry-run

Note: credentials are hardcoded placeholders in this file for quick testing.
"""
from __future__ import annotations
import datetime
import logging
import math
import time
from typing import Optional
import os

import pandas as pd
import yfinance as yf
import requests

# Alpaca credentials: prefer environment variables for safety.
# If env vars are not set, the defaults below (placeholders) are used.
ALPACA_API_KEY = os.getenv('ALPACA_API_KEY', "PKKPLP36QVRYGIXPWJ4H6I2BG2")
ALPACA_API_SECRET = os.getenv('ALPACA_API_SECRET', "CVskWgEoGjbi17M1GYs9FYB3EWwipGZNBBzeaVHmGJQX")
# By default the script was using the paper endpoint. To submit real/live
# orders set ALPACA_BASE_URL to 'https://api.alpaca.markets' via env var.
ALPACA_BASE_URL = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
ALPACA_DATA_URL = os.getenv('ALPACA_DATA_URL', 'https://data.alpaca.markets')


# Configuration — change these values to customize behavior
# Symbols
SYMBOL_QQQ = 'QQQ'
SYMBOL_TQQQ = 'TQQQ'

# Strategy defaults (easy to edit)
EMA_FAST = 9
EMA_SLOW = 21
STOP_LOSS_PCT = 15.0  # trailing stop as percent of peak price

# Simulation / runtime defaults
INITIAL_CAPITAL = 10000.0
ALLOCATION_FRACTION = 1.0  # fraction of cash to allocate on buy

# Data-fetch lookback windows (days)
HISTORY_DAYS_QQQ = 365
HISTORY_DAYS_TQQQ = 7

# Loop defaults
DEFAULT_LOOP_INTERVAL = 60

# Fixed dollars to spend on BUY signal
FIXED_BUY_DOLLARS = 100.0


def _alpaca_headers():
    return {
        'APCA-API-KEY-ID': ALPACA_API_KEY,
        'APCA-API-SECRET-KEY': ALPACA_API_SECRET
    }


def get_latest_trade_from_alpaca(symbol: str) -> float:
    """Fetch latest trade price for `symbol` from Alpaca data endpoint.

    Falls back to yfinance if Alpaca data is unavailable or credentials missing.
    """
    if not ALPACA_API_KEY or not ALPACA_API_SECRET:
        # fallback
        ticker = yf.Ticker(symbol)
        last = ticker.history(period='1d')
        if not last.empty:
            return float(last['Close'].iloc[-1])
        raise RuntimeError('No price available for ' + symbol)

    try:
        url = f"{ALPACA_DATA_URL}/v2/stocks/{symbol}/trades/latest"
        resp = requests.get(url, headers=_alpaca_headers(), timeout=10)
        resp.raise_for_status()
        j = resp.json()
        price = j.get('trade', {}).get('p')
        if price is None:
            raise ValueError('No price in response')
        return float(price)
    except Exception:
        # fallback to yfinance
        ticker = yf.Ticker(symbol)
        last = ticker.history(period='1d')
        if not last.empty:
            return float(last['Close'].iloc[-1])
        raise


def submit_order_alpaca(symbol: str, notional: float, side: str = 'buy') -> dict:
    qty_int = int(round(notional))
    if qty_int <= 0:
        raise ValueError(f'Notional {notional} rounds to {qty_int}, cannot submit')
    if not ALPACA_API_KEY or not ALPACA_API_SECRET:
        raise RuntimeError('Alpaca credentials not configured')
    url = f"{ALPACA_BASE_URL}/v2/orders"
    payload = {
        'symbol': symbol,
        'notional': notional,
        'side': side,
        'type': 'market',
        'time_in_force': 'gtc'
    }
    resp = requests.post(url, headers=_alpaca_headers(), json=payload, timeout=10)
    resp.raise_for_status()
    return resp.json()


def get_alpaca_account_cash() -> float:
    """Return available cash (or buying_power) from Alpaca account as float.

    Raises on HTTP/network errors. Caller should handle exceptions and fall
    back to simulated capital if necessary.
    """
    if not ALPACA_API_KEY or not ALPACA_API_SECRET:
        raise RuntimeError('Alpaca credentials not configured')
    url = f"{ALPACA_BASE_URL}/v2/account"
    resp = requests.get(url, headers=_alpaca_headers(), timeout=10)
    resp.raise_for_status()
    j = resp.json()
    # prefer explicit cash; fallback to buying_power
    cash = j.get('cash')
    bp = j.get('buying_power')
    try:
        return float(cash) if cash is not None else float(bp)
    except Exception:
        # If parsing failed, raise to signal caller to fallback
        raise


# Note: configuration values were moved to the top of this file in the
# "# Configuration" block so they are easy to find and edit.


def setup_logger() -> logging.Logger:
    fmt = '%(asctime)s | %(levelname)-5s | %(message)s'
    logger = logging.getLogger('minimal-trader')
    logger.setLevel(logging.INFO)
    # Ensure idempotent setup
    if logger.handlers:
        logger.handlers.clear()
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt))
    logger.addHandler(ch)
    # File handler with hourly rotation
    try:
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, 'app.log')
        from logging.handlers import TimedRotatingFileHandler
        fh = TimedRotatingFileHandler(log_path, when='H', interval=1, backupCount=48, encoding='utf-8')
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(fmt))
        logger.addHandler(fh)
    except Exception as e:
        # If file handler cannot be created, fall back to console only
        logger.warning('file_logging_disabled=%s', e)
    logger.propagate = False
    return logger


logger = setup_logger()


def fetch_history(symbol: str, days: int = 365) -> pd.DataFrame:
    """Fetch daily historical OHLCV for a symbol using yfinance."""
    end = datetime.date.today()
    start = end - datetime.timedelta(days=days)
    # Explicitly set `auto_adjust=False` to preserve historical OHLC semantics
    # and silence the yfinance FutureWarning about the default changing.
    df = yf.download(symbol, start=start.strftime('%Y-%m-%d'), end=end.strftime('%Y-%m-%d'), progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError(f'Failed to download data for {symbol}')
    return df


def compute_double_ema(df: pd.DataFrame, fast: int, slow: int) -> pd.DataFrame:
    # Simple EMA computation: expect a 'Close' column and compute two EMAs.
    df = df.copy()
    if 'Close' not in df.columns:
        # Try to coerce the first numeric column to 'Close'
        num = df.select_dtypes(include='number')
        if num.empty:
            raise RuntimeError('No numeric column available to compute EMAs')
        df['Close'] = num.iloc[:, 0]
    df['EMA_Fast'] = df['Close'].ewm(span=fast, adjust=False).mean()
    df['EMA_Slow'] = df['Close'].ewm(span=slow, adjust=False).mean()
    return df


def latest_price(df: pd.DataFrame) -> float:
    """Return the most recent close price from a DataFrame or Series.

    This function is defensive: it accepts either a Series of close values
    or a DataFrame with a 'Close' or 'Adj Close' column, and falls back to
    the first numeric column if necessary.
    """
    try:
        if isinstance(df, pd.Series):
            return float(df.iloc[-1])

        last = df.tail(1)
        if isinstance(last, pd.Series):
            return float(last.iat[0])

        # last is a one-row DataFrame
        if 'Close' in last.columns:
            return float(last['Close'].to_numpy().item())
        if 'Adj Close' in last.columns:
            return float(last['Adj Close'].to_numpy().item())

        num = last.select_dtypes(include='number')
        if not num.empty:
            return float(num.to_numpy().item())

        # fallback to first cell of the row
        return float(last.iloc[0, 0])
    except Exception as e:
        raise RuntimeError('Failed to extract latest price: ' + str(e))


def safe_float(x) -> float:
    """Convert x to float without triggering pandas' single-element Series warning.

    If x is a pandas Series (single-element), extract via .iat[0]. Otherwise
    just call float(x).
    """
    try:
        if isinstance(x, pd.Series):
            return float(x.iat[0])
        return float(x)
    except Exception:
        # Last-resort: coerce using pandas and extract
        try:
            return float(pd.to_numeric(x).item())
        except Exception:
            return float(pd.to_numeric(x).iloc[0])


def compute_signal_from_qqq(df_qqq: pd.DataFrame) -> str:
    """Return 'BUY' if EMA_Fast > EMA_Slow else 'SELL'.

    Uses scalar extraction to avoid ambiguous Series truth-value errors.
    """
    try:
        # use .iat to get a scalar value rather than a single-element Series
        ef = float(df_qqq['EMA_Fast'].iat[-1])
        es = float(df_qqq['EMA_Slow'].iat[-1])
    except Exception:
        return 'HOLD'
    if math.isnan(ef) or math.isnan(es):
        return 'HOLD'
    return 'BUY' if ef > es else 'SELL'


def compute_trailing_stop(peak_price: Optional[float], current_price: float, pct: float) -> tuple[float, float]:
    """Update peak price and compute trailing stop price.

    Returns (new_peak, stop_price)
    """
    new_peak = current_price if peak_price is None else max(peak_price, current_price)
    stop_price = new_peak * (1.0 - pct / 100.0) if pct > 0 else 0.0
    return new_peak, stop_price


def run_once(ema_fast: int, ema_slow: int, stop_pct: float, capital: float, live: bool = False, dry_run: bool = True):
    # Fetch history for indicators (QQQ) and use Alpaca/yfinance for current prices
    # Fetch raw history (keep a clean copy without EMA columns)
    hist = fetch_history(SYMBOL_QQQ, days=HISTORY_DAYS_QQQ)
    start_ts = datetime.datetime.now().isoformat()

    # Use the EMA spans provided by the caller/config without modification.
    fast_span = int(ema_fast)
    slow_span = int(ema_slow)

    # Compute EMAs from the raw history DataFrame (qqq is the annotated DF)
    qqq = compute_double_ema(hist, fast_span, slow_span)

    # Log which spans are used and the configured stop-loss percentage (compact)
    logger.debug('QQQ tail after EMA compute:\n%s', qqq[['Close', 'EMA_Fast', 'EMA_Slow']].tail())

    # Determine current market prices; prefer Alpaca live trade when in live mode
    try:
        if live:
            tqqq_price = get_latest_trade_from_alpaca(SYMBOL_TQQQ)
            qqq_live_price = get_latest_trade_from_alpaca(SYMBOL_QQQ)
            # Inject live price into qqq for a realtime EMA comparison.
            # Instead of overwriting the last historical row (which can
            # create NaNs if the last row is incomplete), append a new
            # one-row DataFrame with the live price as the most-recent
            # point and then recompute EMAs.
            try:
                last_index = qqq.index[-1]
            except Exception:
                last_index = None

            new_idx = pd.Timestamp.now()
            # Preserve the original row structure: copy the last row and set
            # its Close to the live price so column/index layout remains
            # consistent (avoids MultiIndex/column-mismatch issues).
            # Safer approach: operate on a single 'Close' Series to avoid
            # column/index shape mismatches (MultiIndex vs single-level).
            # Extract the Close series from hist (or first numeric column),
            # append the live price, compute EMAs on that Series, and then
            # attach the EMA columns back onto a copy of the original history.
            try:
                if 'Close' in hist.columns:
                    close_ser = hist['Close'].astype(float).copy()
                else:
                    num = hist.select_dtypes(include='number')
                    if num.empty:
                        raise RuntimeError('No numeric column available to extract Close')
                    close_ser = num.iloc[:, 0].astype(float).copy()

                # Append live price as a new timestamped row
                close_ser = pd.concat([close_ser, pd.Series([qqq_live_price], index=[new_idx])])

                # Compute EMAs on the Close series
                ema_fast_ser = close_ser.ewm(span=fast_span, adjust=False).mean()
                ema_slow_ser = close_ser.ewm(span=slow_span, adjust=False).mean()

                # Build qqq as a copy of hist indexed to include the new timestamp
                qqq = hist.copy()
                if new_idx not in qqq.index:
                    # replicate last row to preserve columns then set its Close
                    last_row = qqq.iloc[-1].copy()
                    last_row.name = new_idx
                    qqq = pd.concat([qqq, last_row.to_frame().T])

                # Assign the scalar Close and EMA series into qqq (align by index)
                qqq['Close'] = close_ser.reindex(qqq.index)
                qqq['EMA_Fast'] = ema_fast_ser.reindex(qqq.index)
                qqq['EMA_Slow'] = ema_slow_ser.reindex(qqq.index)
            except Exception:
                # If anything goes wrong, fallback to original (less safe) method
                last_row = hist.iloc[-1].copy()
                last_row.name = new_idx
                last_row['Close'] = qqq_live_price
                hist2 = pd.concat([hist, last_row.to_frame().T])
                qqq = compute_double_ema(hist2, fast_span, slow_span)

            # If EMAs ended up NaN, log the tail for debugging
            if pd.isna(qqq['EMA_Fast'].iat[-1]) or pd.isna(qqq['EMA_Slow'].iat[-1]):
                logger.debug('QQQ tail after live injection:\n%s', qqq.tail())
            # If the fast and slow EMAs are identical (possible when spans are
            # misconfigured or when there's insufficient history), log a short
            # diagnostic to help the user understand why.
            # No assumption: keep spans and values as computed. If you want
            # to inspect the tail for diagnosis, enable DEBUG logging.
        else:
            tqqq_price = latest_price(fetch_history(SYMBOL_TQQQ, days=HISTORY_DAYS_TQQQ))
    except Exception as e:
        logger.warning('Could not fetch Alpaca live prices, falling back to yfinance: %s', e)
        tqqq_price = latest_price(fetch_history(SYMBOL_TQQQ, days=HISTORY_DAYS_TQQQ))

    qqq_close = latest_price(qqq)
    signal = compute_signal_from_qqq(qqq)

    # Position bookkeeping via Alpaca when live, otherwise local simulated state
    # Initialize these before computing the SL preview so the preview can
    # use a defined `peak_price` (None when flat) and compute a hypothetical
    # stop correctly.
    position_shares = 0
    entry_price = 0.0
    peak_price: Optional[float] = None

    # Compute a preview trailing-stop using the current TQQQ price so we can
    # display the stop-loss value in Analysis logs (even if no position exists).
    try:
        preview_peak, preview_stop = compute_trailing_stop(peak_price, tqqq_price, stop_pct)
    except Exception:
        preview_stop = float('nan')

    # Format the SL preview as a human-friendly string (N/A when not available)
    if isinstance(preview_stop, float) and math.isfinite(preview_stop):
        sl_display = f"{preview_stop:.2f}"
    else:
        sl_display = 'N/A'

    # Extract EMA scalars robustly (handle MultiIndex or unexpected column shapes)
    try:
        ef = float(qqq['EMA_Fast'].to_numpy().item() if qqq['EMA_Fast'].to_numpy().size == 1 else qqq['EMA_Fast'].to_numpy()[-1])
    except Exception:
        try:
            ef = float(qqq['EMA_Fast'].iat[-1])
        except Exception:
            ef = float('nan')
    try:
        es = float(qqq['EMA_Slow'].to_numpy().item() if qqq['EMA_Slow'].to_numpy().size == 1 else qqq['EMA_Slow'].to_numpy()[-1])
    except Exception:
        try:
            es = float(qqq['EMA_Slow'].iat[-1])
        except Exception:
            es = float('nan')

    # Reformat Analysis log as requested: compact, CSV-like with parentheses
    ef_str = (f"{ef:.2f}" if not math.isnan(ef) else 'nan')
    es_str = (f"{es:.2f}" if not math.isnan(es) else 'nan')
    # Log only EMA values and signal; omit raw QQQ price as requested
    logger.info('qqq_ema_fast=%s, qqq_ema_slow=%s, sl_preview=%s, signal=%s', ef_str, es_str, sl_display, signal)

    # (compact logging) the most-recent values are included in the Analysis line above

    # Position bookkeeping via Alpaca when live, otherwise local simulated state
    position_shares = 0
    entry_price = 0.0
    peak_price: Optional[float] = None

    # If live, check current position via Alpaca API
    if live and ALPACA_API_KEY and ALPACA_API_SECRET:
        try:
            url = f"{ALPACA_BASE_URL}/v2/positions/{SYMBOL_TQQQ}"
            resp = requests.get(url, headers=_alpaca_headers(), timeout=10)
            if resp.status_code == 200:
                pos = resp.json()
                position_shares = float(pos.get('qty', 0))
                entry_price = float(pos.get('avg_entry_price', 0.0)) if pos.get('avg_entry_price') else 0.0
                peak_price = None
            else:
                position_shares = 0
        except Exception:
            position_shares = 0

    # If we have a live position, compute and log the active stop-loss and
    # the drawdown percent so you can see how close we are to the configured SL.
    if position_shares > 0:
        try:
            # compute trailing stop using stored peak_price (may be None)
            cur_peak, cur_stop = compute_trailing_stop(peak_price, tqqq_price, stop_pct)
        except Exception:
            cur_peak, cur_stop = (peak_price, None)

        # compute drawdown percent relative to peak if available, else entry_price
        drawdown_pct: Optional[float] = None
        ref_price = cur_peak if cur_peak is not None else (entry_price if entry_price > 0 else None)
        if ref_price is not None and ref_price > 0:
            drawdown_pct = (ref_price - tqqq_price) / ref_price * 100.0

        # Compute distance to stop in dollars and percent (percent of peak when available)
        dist_dollars: Optional[float] = None
        pct_to_stop: Optional[float] = None
        if cur_stop is not None:
            dist_dollars = tqqq_price - cur_stop
            if cur_peak is not None and cur_peak > 0:
                pct_to_stop = dist_dollars / cur_peak * 100.0
            else:
                pct_to_stop = (dist_dollars / tqqq_price) * 100.0 if tqqq_price > 0 else None

        sl_str = f"${cur_stop:.2f}" if cur_stop is not None else 'N/A'
        draw_str = f"{drawdown_pct:.2f}%" if drawdown_pct is not None else 'N/A'
        dist_str = f"${dist_dollars:.2f}" if dist_dollars is not None else 'N/A'
        pct_str = f"{pct_to_stop:.2f}%" if pct_to_stop is not None else 'N/A'

        logger.info('position_shares=%.2f, peak=%s, tqqq_price=%.2f, stop=%s, dist_to_stop=%s, pct_to_stop=%s, drawdown=%s, sl_cfg_pct=%.2f', position_shares, f"{cur_peak:.2f}" if cur_peak is not None else 'N/A', tqqq_price, sl_str, dist_str, pct_str, draw_str, stop_pct)

    # If running live, attempt to read account cash and override provided capital
    capital_local = capital
    if live:
        try:
            capital_local = float(get_alpaca_account_cash())
            logger.info('alpaca_cash=%.2f', capital_local)
        except Exception as e:
            logger.warning('Could not read Alpaca account cash, using provided capital %.2f: %s', capital_local, e)

    # Execute orders if signal indicates and we are in live mode (and not dry-run)
    if signal == 'BUY' and position_shares == 0:
        qty = round(capital_local / tqqq_price)
        budget = qty * tqqq_price
        if qty > 0:
            logger.info('order_action=BUY, order_qty=%.0f, order_symbol=%s, order_price=%.2f, order_budget=%.2f', qty, SYMBOL_TQQQ, tqqq_price, budget)
            if live and not dry_run:
                order = submit_order_alpaca(SYMBOL_TQQQ, qty, side='buy')
                logger.info('order_submitted=true, order_side=BUY, order_id=%s', order.get('id'))
            else:
                logger.info('dry_run=true, order_side=BUY, submitted=false')
        else:
            logger.info('insufficient_cash, signal=BUY, cash=%.2f', capital_local)
    elif signal == 'SELL' and position_shares >= 0.5:
        qty = round(position_shares)
        if qty > 0:
            logger.info('order_action=SELL, order_qty=%.0f, order_symbol=%s, order_price=%.2f', qty, SYMBOL_TQQQ, tqqq_price)
            if live and not dry_run:
                order = submit_order_alpaca(SYMBOL_TQQQ, qty, side='sell')
                logger.info('order_submitted=true, order_side=SELL, order_id=%s', order.get('id'))
            else:
                logger.info('dry_run=true, order_side=SELL, submitted=false')
        else:
            logger.info('insufficient_shares_to_sell, signal=SELL, shares=%.2f', position_shares)
    elif signal == 'SELL' and 0 < position_shares < 0.5:
        logger.info('insufficient_shares_to_sell, signal=SELL, shares=%.2f', position_shares)
    else:
        logger.info('signal=%s, position=%s', signal, 'long' if position_shares > 0 else 'flat')

    # Compute trailing stop preview (even if not selling)
    if position_shares > 0:
        peak_price, stop_price = compute_trailing_stop(peak_price, tqqq_price, stop_pct)
        # Removed verbose trailing stop line per request; keep only trigger event.
        # logger.info('trailing_peak=%.2f, trailing_stop_pct=%.2f, trailing_stop_price=%.2f', peak_price, stop_pct, stop_price)
        if tqqq_price <= stop_price:
            logger.info('stop_triggered=true, price_current=%.2f, price_stop=%.2f', tqqq_price, stop_price)
    else:
        logger.info('position=None, trailing_stop=None')

    # Single-line OP summary (brief, human readable) logged every loop
    try:
        pos_side = 'long' if position_shares > 0 else 'flat'
        entry_txt = f"@{entry_price:.2f}" if position_shares > 0 and entry_price > 0 else ''
        stop_txt = f"Stop={sl_str}" if position_shares > 0 else 'Stop=N/A'
        dist_txt = f"{pct_str}/{dist_str}" if position_shares > 0 else ''
        op_line = (
            f"OP Sig={signal}, EMA{fast_span}={ef_str}, EMA{slow_span}={es_str}, "
            f"TQQQ={tqqq_price:.2f}, Pos={pos_side}:{position_shares:.2f}{('/'+entry_txt) if entry_txt else ''}, "
            f"{stop_txt}{(', Dist='+dist_txt) if dist_txt else ''}, Cash=${capital_local:.2f}, SLprev=${sl_display}"
        )
        logger.info(op_line)
    except Exception:
        pass


def run_loop(ema_fast: int, ema_slow: int, stop_pct: float, capital: float, interval: int, live: bool = False, dry_run: bool = True):
    logger.info('loop_ema_fast=%d, loop_ema_slow=%d, loop_sl_pct=%.2f, loop_interval=%d', ema_fast, ema_slow, stop_pct, interval)
    while True:
        try:
            run_once(ema_fast, ema_slow, stop_pct, capital, live=live, dry_run=dry_run)
        except Exception as e:
            logger.exception('Error during iteration: %s', e)
        logger.info('sleep_seconds=%d', interval)
        time.sleep(interval)


def main():
    # Simplified runner: no command-line options. Always run live and loop.
    logger.info('mode_live=True, mode_loop=True')
    run_loop(EMA_FAST, EMA_SLOW, STOP_LOSS_PCT, INITIAL_CAPITAL, DEFAULT_LOOP_INTERVAL, live=True, dry_run=False)


if __name__ == '__main__':
    main()
