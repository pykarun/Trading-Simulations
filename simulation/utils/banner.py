"""Applied parameters banner utility."""
import streamlit as st
import json


def show_applied_params_banner():
    """Show persistent banner with currently applied parameters.
    
    Returns:
        True if banner was shown, False otherwise
    """
    # Only show if user has explicitly applied parameters
    if st.session_state.get('best_params') is not None and st.session_state.get('user_applied_params', False):
        applied_rank = st.session_state.get('applied_rank', 'Unknown')
        params = st.session_state.best_params
        
        # Build parameter summary
        if params.get('use_double_ema', False):
            ema_text = f"EMA({params.get('ema_fast', 9)}/{params.get('ema_slow', 21)})"
        else:
            ema_text = f"EMA({params.get('ema_period', 50)})"
        
        rsi_text = f"RSI>{params.get('rsi_threshold', 0)}" if params.get('use_rsi', False) and params.get('rsi_threshold', 0) > 0 else "No RSI"
        sl_text = f"SL:{params.get('stop_loss_pct', 0)}%" if params.get('use_stop_loss', False) else "No SL"
        bb_text = "BB:On" if params.get('use_bb', False) else ""
        atr_text = f"ATR:{params.get('atr_multiplier', 0)}x" if params.get('use_atr', False) else ""
        msl_text = "MSL:On" if params.get('use_msl_msh', False) else ""
        macd_text = "MACD:On" if params.get('use_macd', False) else ""
        adx_text = "ADX:On" if params.get('use_adx', False) else ""
        st_text = "ST:On" if params.get('use_supertrend', False) else ""
        pivot_text = f"Pivot:{params.get('pivot_left', 5)}/{params.get('pivot_right', 5)}" if params.get('use_pivot', False) else ""
        
        # Build full summary
        summary_parts = [ema_text, rsi_text, sl_text]
        if bb_text:
            summary_parts.append(bb_text)
        if atr_text:
            summary_parts.append(atr_text)
        if msl_text:
            summary_parts.append(msl_text)
        if macd_text:
            summary_parts.append(macd_text)
        if adx_text:
            summary_parts.append(adx_text)
        if st_text:
            summary_parts.append(st_text)
        if pivot_text:
            summary_parts.append(pivot_text)
        
        summary = " | ".join(summary_parts)
        
        st.success(f"✅ **Currently Applied: Rank #{applied_rank}** | {summary}")
        
        # Show both Python-dict and JSON representations in expandable section for easy copy/paste
        with st.expander("📋 View Applied Parameters (Copy)", expanded=False):
            try:
                canonical = _build_canonical_params(params)
                # Python-style dict (single-quotes, True/False) for quick paste into scripts
                st.markdown("**Python dict (copy into Python scripts):**")
                st.code(repr(canonical), language='python')
                st.caption("💡 Paste this block into your Python code (keys use single-quotes).")

                st.markdown("---")
                # JSON version for config files or systems expecting JSON
                st.markdown("**JSON version (for config files):**")
                st.code(json.dumps(canonical, indent=2), language='json')
                st.caption("💡 Copy this JSON into config files or APIs.")
            except Exception:
                # Fallback: show raw params as repr and JSON
                st.code(repr(params), language='python')
                st.markdown('---')
                st.code(json.dumps(params, indent=2), language='json')
                st.caption("💡 Parameters (fallback) - some values may not be normalized.")
        
        return True
    return False


def _build_canonical_params(params: dict) -> dict:
    """Return a canonical params dict with stable keys and defaults suitable for JSON export.

    This mirrors the keys shown in the Python-dict view so users can copy a ready-to-use
    configuration object.
    """
    return {
        'use_ema': params.get('use_ema', True),
        'use_double_ema': params.get('use_double_ema', False),
        'ema_period': params.get('ema_period', 21),
        'ema_fast': params.get('ema_fast', 9),
        'ema_slow': params.get('ema_slow', 21),

        'use_rsi': params.get('use_rsi', False),
        'rsi_threshold': params.get('rsi_threshold', 50),
        'rsi_oversold': params.get('rsi_oversold', 30),
        'rsi_overbought': params.get('rsi_overbought', 70),

        'use_bb': params.get('use_bb', False),
        'bb_period': params.get('bb_period', 20),
        'bb_std_dev': params.get('bb_std_dev', 2.0),
        'bb_buy_threshold': params.get('bb_buy_threshold', 0.2),
        'bb_sell_threshold': params.get('bb_sell_threshold', 0.8),

        'use_atr': params.get('use_atr', False),
        'atr_period': params.get('atr_period', 14),
        'atr_multiplier': params.get('atr_multiplier', 2.0),

        'use_msl_msh': params.get('use_msl_msh', False),
        'msl_period': params.get('msl_period', 20),
        'msh_period': params.get('msh_period', 20),
        'msl_lookback': params.get('msl_lookback', 5),
        'msh_lookback': params.get('msh_lookback', 5),

        'use_macd': params.get('use_macd', False),
        'macd_fast': params.get('macd_fast', 12),
        'macd_slow': params.get('macd_slow', 26),
        'macd_signal_period': params.get('macd_signal_period', 9),

        'use_adx': params.get('use_adx', False),
        'adx_period': params.get('adx_period', 14),
        'adx_threshold': params.get('adx_threshold', 25),

        'use_supertrend': params.get('use_supertrend', False),
        'st_period': params.get('st_period', 10),
        'st_multiplier': params.get('st_multiplier', 3.0),

        'use_pivot': params.get('use_pivot', False),
        'pivot_left': params.get('pivot_left', 5),
        'pivot_right': params.get('pivot_right', 5),

        # Stop-loss canonical keys
        'use_stop_loss': params.get('use_stop_loss', False),
        'stop_loss_type': params.get('stop_loss_type', 'percentage'),
        'stop_loss_pct': params.get('stop_loss_pct', 0),
    }
