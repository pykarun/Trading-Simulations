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
        
        # Build full summary
        summary_parts = [ema_text, rsi_text, sl_text]
        if bb_text:
            summary_parts.append(bb_text)
        if atr_text:
            summary_parts.append(atr_text)
        if msl_text:
            summary_parts.append(msl_text)
        
        summary = " | ".join(summary_parts)
        
        st.success(f"✅ **Currently Applied: Rank #{applied_rank}** | {summary}")
        
        # Show JSON string in expandable section
        with st.expander("📋 View Applied Parameters (JSON)", expanded=False):
            st.code(json.dumps(params, indent=2), language='json')
            st.caption("💡 These parameters are stored in session state and ready to use.")
        
        return True
    return False
