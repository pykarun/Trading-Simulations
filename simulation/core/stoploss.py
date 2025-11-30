"""Shared stop-loss helper utilities.

Provides a minimal, well-tested set of pure functions to compute/update
trailing stop prices and evaluate whether a current price breaches the
configured stop. This module is intentionally small to keep integration
risk low when called from both the live trader and the backtester.
"""
from typing import Any, Dict, Optional
import pandas as pd


def compute_stop_state(params: Dict[str, Any], indicators: pd.Series, entry_price: float,
                       peak_price: Optional[float], current_price: float,
                       tqqq_row: Optional[pd.Series] = None) -> Dict[str, Any]:
    """Compute an updated stop-loss state.

    Args:
        params: Strategy parameters (dict). Expected keys: `use_stop_loss`,
            `stop_loss_pct`, `use_atr`, `atr_multiplier`, `use_msl_msh`.
        indicators: Indicator series (usually the QQQ indicator row).
        entry_price: Entry price for the current position (float).
        peak_price: Previously-observed peak price while position held or None.
        current_price: Current market price (float) used to update peak.
        tqqq_row: Optional TQQQ row (pd.Series) to prefer TQQQ-based ATR when available.

    Returns:
        A dict with keys:
          - 'peak_price': updated peak price (float)
          - 'stop_price': computed stop price or None
          - 'triggered': bool indicating if current_price breaches the stop
          - 'reason': short string explaining the result
          - 'method': one of 'percentage', 'atr', 'msl' or None
    """
    result = {
        'peak_price': peak_price,
        'stop_price': None,
        'triggered': False,
        'reason': '',
        'method': None
    }

    # Update peak price (if provided, otherwise initialize to current_price)
    if peak_price is None:
        result['peak_price'] = current_price
    else:
        result['peak_price'] = max(peak_price, current_price)

    # Percentage-based trailing stop (based on peak price)
    if params.get('use_stop_loss', False):
        pct = float(params.get('stop_loss_pct', 0.0) or 0.0)
        if pct > 0:
            stop = result['peak_price'] * (1.0 - (pct / 100.0))
            result['stop_price'] = float(stop)
            result['method'] = 'percentage'
            result['reason'] = f'Percentage-based SL: pct={pct:.2f} peak={result["peak_price"]:.2f}'
            if current_price <= result['stop_price']:
                result['triggered'] = True
            return result

    # ATR-based stop: prefer ATR from provided tqqq_row, otherwise from indicators
    if params.get('use_atr', False):
        atr_val = None
        if tqqq_row is not None:
            atr_val = tqqq_row.get('ATR', None)
        if atr_val is None:
            atr_val = indicators.get('ATR') if indicators is not None else None
        try:
            if pd.notna(atr_val) and float(atr_val) > 0:
                mult = float(params.get('atr_multiplier', 2.0) or 2.0)
                # Keep the default behavior: ATR stop relative to entry price
                stop = float(entry_price) - (float(atr_val) * mult)
                result['stop_price'] = stop
                result['method'] = 'atr'
                result['reason'] = f'ATR-based SL: atr={atr_val:.2f} mult={mult:.2f} entry={entry_price:.2f}'
                if current_price <= result['stop_price']:
                    result['triggered'] = True
                return result
        except Exception:
            # Safely ignore calculation problems and fall through
            pass

    # MSL/MSH: we don't compute a single numeric stop here; consumer should
    # use indicator values directly (MSL/MSH in the data). Return method hint.
    if params.get('use_msl_msh', False):
        result['method'] = 'msl'
        result['reason'] = 'MSL/MSH active; check indicator value for stop'
        # No numeric stop_price provided
        return result

    # No stop configured
    result['reason'] = 'No stop-loss configured'
    return result
