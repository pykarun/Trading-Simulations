"""Step 3: Verify & Customize Strategy.

This module allows users to:
- Review applied strategy parameters
- Manually adjust any parameter
- See buy/sell signal conditions
- Apply custom configuration
"""

import streamlit as st
import json
from utils import show_applied_params_banner

try:
    rerun = st.rerun
except AttributeError:
    rerun = st.experimental_rerun


def render_step3():
    """Render Step 3: Verify & Customize Strategy."""
    
    st.markdown("## 📝 Step 3: Verify & Customize Strategy")
    st.markdown("---")
    
    # Show applied parameters banner
    show_applied_params_banner()
    
    st.markdown("---")
    
    # Auto-load strategy if best_params exists
    if st.session_state.get('best_params') is not None:
        if not st.session_state.get('manual_config_loaded', False):
            _load_params_to_manual_config()
            st.session_state.manual_config_loaded = True
        
        st.markdown("###### Strategy loaded. You can modify parameters below if needed.")
        
        # Get defaults from session state
        defaults = _get_manual_defaults()
        
        # Render configuration UI
        manual_params = _render_manual_configuration(defaults)
        
        # Signal Summary
        _render_signal_summary(manual_params)
        
        # Apply button
        if st.button("✅ Apply Manual Configuration", type="primary", use_container_width=True, key="manual_apply"):
            _apply_manual_configuration(manual_params)
        
        # Navigation buttons
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            if st.button("⬅️ Previous Step", use_container_width=True, key="step3_prev"):
                st.session_state.wizard_step = 2
                rerun()
        
        with col2:
            pass  # Empty middle column for spacing
        
        with col3:
            if st.button("Next Step ➡️", type="primary", use_container_width=True, key="step3_next"):
                st.session_state.wizard_step = 4
                rerun()
        
        # Add bottom padding for mobile visibility
        st.markdown("<br><br>", unsafe_allow_html=True)
    else:
        st.warning("⚠️ No strategy selected. Please find best signals in Step 2 first.")
        st.stop()


def _load_params_to_manual_config():
    """Load best_params into manual configuration session state."""
    best_params = st.session_state.best_params
    
    st.session_state.manual_use_double_ema = best_params.get('use_double_ema', False)
    st.session_state.manual_ema_period = best_params.get('ema_period', 50)
    st.session_state.manual_ema_fast = best_params.get('ema_fast', 9)
    st.session_state.manual_ema_slow = best_params.get('ema_slow', 21)
    st.session_state.manual_use_rsi = best_params.get('use_rsi', False)
    st.session_state.manual_rsi_threshold = best_params.get('rsi_threshold', 50)
    st.session_state.manual_rsi_oversold = best_params.get('rsi_oversold', 30)
    st.session_state.manual_rsi_overbought = best_params.get('rsi_overbought', 70)
    st.session_state.manual_use_stop_loss = best_params.get('use_stop_loss', False)
    st.session_state.manual_stop_loss_pct = best_params.get('stop_loss_pct', 10)
    st.session_state.manual_use_bb = best_params.get('use_bb', False)
    st.session_state.manual_bb_period = best_params.get('bb_period', 20)
    st.session_state.manual_bb_std_dev = best_params.get('bb_std_dev', 2.0)
    st.session_state.manual_bb_buy_threshold = best_params.get('bb_buy_threshold', 0.2)
    st.session_state.manual_bb_sell_threshold = best_params.get('bb_sell_threshold', 0.8)
    st.session_state.manual_use_atr = best_params.get('use_atr', False)
    st.session_state.manual_atr_period = best_params.get('atr_period', 14)
    st.session_state.manual_atr_multiplier = best_params.get('atr_multiplier', 2.0)
    st.session_state.manual_use_msl_msh = best_params.get('use_msl_msh', False)
    st.session_state.manual_msl_period = best_params.get('msl_period', 20)
    st.session_state.manual_msl_lookback = best_params.get('msl_lookback', 5)


def _get_manual_defaults():
    """Get default values from session state."""
    return {
        'double_ema': st.session_state.manual_use_double_ema,
        'ema_period': st.session_state.manual_ema_period,
        'ema_fast': st.session_state.manual_ema_fast,
        'ema_slow': st.session_state.manual_ema_slow,
        'use_rsi': st.session_state.manual_use_rsi,
        'rsi_threshold': st.session_state.manual_rsi_threshold,
        'rsi_oversold': st.session_state.manual_rsi_oversold,
        'rsi_overbought': st.session_state.manual_rsi_overbought,
        'use_stop_loss': st.session_state.manual_use_stop_loss,
        'stop_loss_pct': st.session_state.manual_stop_loss_pct,
        'use_bb': st.session_state.manual_use_bb,
        'bb_period': st.session_state.manual_bb_period,
        'bb_std_dev': st.session_state.manual_bb_std_dev,
        'bb_buy_threshold': st.session_state.manual_bb_buy_threshold,
        'bb_sell_threshold': st.session_state.manual_bb_sell_threshold,
        'use_atr': st.session_state.manual_use_atr,
        'atr_period': st.session_state.manual_atr_period,
        'atr_multiplier': st.session_state.manual_atr_multiplier,
        'use_msl_msh': st.session_state.manual_use_msl_msh,
        'msl_period': st.session_state.manual_msl_period,
        'msl_lookback': st.session_state.manual_msl_lookback
    }


def _render_manual_configuration(defaults):
    """Render manual configuration UI and return selected parameters."""
    
    st.markdown("---")
    
    # EMA Strategy
    st.markdown("**Condition 1: EMA Strategy**")
    manual_use_double_ema = st.checkbox(
        "Use Double EMA Crossover",
        value=defaults['double_ema'],
        help="Use two EMAs (fast/slow crossover) instead of single EMA"
    )
    
    if manual_use_double_ema:
        col1, col2 = st.columns(2)
        with col1:
            manual_ema_fast = st.number_input("Fast EMA", min_value=5, max_value=100, value=defaults['ema_fast'], step=1)
        with col2:
            manual_ema_slow = st.number_input("Slow EMA", min_value=10, max_value=200, value=defaults['ema_slow'], step=1)
        manual_ema_period = manual_ema_slow
    else:
        manual_ema_period = st.number_input("EMA Period", min_value=5, max_value=200, value=defaults['ema_period'], step=5)
        manual_ema_fast = 9
        manual_ema_slow = 21
    
    # RSI Filter
    st.markdown("**Condition 2: RSI Filter (Optional)**")
    manual_use_rsi = st.checkbox("Enable RSI Filter", value=defaults['use_rsi'])
    
    if manual_use_rsi:
        col1, col2, col3 = st.columns(3)
        with col1:
            manual_rsi_threshold = st.number_input(
                "RSI Momentum Threshold", min_value=0, max_value=100,
                value=defaults['rsi_threshold'], step=5,
                help="Buy when RSI > threshold (0 = disabled)"
            )
        with col2:
            manual_rsi_oversold = st.number_input(
                "RSI Oversold (Buy)", min_value=10, max_value=50,
                value=defaults['rsi_oversold'], step=5,
                help="Buy signal when RSI < oversold level"
            )
        with col3:
            manual_rsi_overbought = st.number_input(
                "RSI Overbought (Sell)", min_value=50, max_value=90,
                value=defaults['rsi_overbought'], step=5,
                help="Sell signal when RSI > overbought level"
            )
    else:
        manual_rsi_threshold = 0
        manual_rsi_oversold = 30
        manual_rsi_overbought = 70
    
    # Stop-Loss
    st.markdown("**Condition 3: Stop-Loss (Optional)**")
    col1, col2 = st.columns(2)
    with col1:
        manual_use_stop_loss = st.checkbox("Enable Stop-Loss", value=defaults['use_stop_loss'])
    with col2:
        if manual_use_stop_loss:
            manual_stop_loss_pct = st.number_input("Stop-Loss %", min_value=5, max_value=30, value=defaults['stop_loss_pct'], step=1)
        else:
            manual_stop_loss_pct = 0
    
    # Bollinger Bands
    st.markdown("**Condition 4: Bollinger Bands Filter (Optional)**")
    manual_use_bb = st.checkbox(
        "Enable Bollinger Bands Filter",
        value=defaults['use_bb'],
        help="Use Bollinger Bands to filter buy/sell signals based on price position"
    )
    
    if manual_use_bb:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            manual_bb_period = st.number_input("BB Period", min_value=10, max_value=50, value=defaults['bb_period'], step=5)
        with col2:
            manual_bb_std_dev = st.number_input("BB Std Dev", min_value=1.0, max_value=3.0, value=defaults['bb_std_dev'], step=0.5)
        with col3:
            manual_bb_buy_threshold = st.number_input("Buy Threshold", min_value=0.0, max_value=0.5, value=defaults['bb_buy_threshold'], step=0.1)
        with col4:
            manual_bb_sell_threshold = st.number_input("Sell Threshold", min_value=0.5, max_value=1.0, value=defaults['bb_sell_threshold'], step=0.1)
    else:
        manual_bb_period = 20
        manual_bb_std_dev = 2.0
        manual_bb_buy_threshold = 0.2
        manual_bb_sell_threshold = 0.8
    
    # ATR Stop-Loss
    st.markdown("**Condition 5: ATR Stop-Loss (Optional)**")
    manual_use_atr = st.checkbox("Enable ATR Stop-Loss", value=defaults['use_atr'])
    
    if manual_use_atr:
        st.info("💡 **ATR Multiplier Guide:** 1.5-2.0 = Day Trading | 2.0-3.0 = Swing Trading | 3.0-4.0 = Position Trading")
        col1, col2 = st.columns(2)
        with col1:
            manual_atr_period = st.slider("ATR Period", min_value=5, max_value=30, value=defaults['atr_period'], step=1)
        with col2:
            manual_atr_multiplier = st.slider("ATR Multiplier", min_value=1.0, max_value=4.0, value=defaults['atr_multiplier'], step=0.25)
    else:
        manual_atr_period = 14
        manual_atr_multiplier = 2.0
    
    # MSL/MSH Stop-Loss
    st.markdown("**Condition 6: MSL/MSH Stop-Loss (Optional)**")
    manual_use_msl_msh = st.checkbox("Enable MSL/MSH Stop-Loss", value=defaults['use_msl_msh'])
    
    if manual_use_msl_msh:
        col1, col2 = st.columns(2)
        with col1:
            manual_msl_period = st.slider("MSL Smoothing Period", min_value=5, max_value=50, value=defaults['msl_period'], step=5)
        with col2:
            manual_msl_lookback = st.slider("MSL Lookback Period", min_value=3, max_value=20, value=defaults['msl_lookback'], step=1)
    else:
        manual_msl_period = 20
        manual_msl_lookback = 5
    
    return {
        'use_double_ema': manual_use_double_ema,
        'ema_period': manual_ema_period,
        'ema_fast': manual_ema_fast,
        'ema_slow': manual_ema_slow,
        'use_rsi': manual_use_rsi,
        'rsi_threshold': manual_rsi_threshold,
        'rsi_oversold': manual_rsi_oversold,
        'rsi_overbought': manual_rsi_overbought,
        'use_stop_loss': manual_use_stop_loss,
        'stop_loss_pct': manual_stop_loss_pct,
        'use_bb': manual_use_bb,
        'bb_period': manual_bb_period,
        'bb_std_dev': manual_bb_std_dev,
        'bb_buy_threshold': manual_bb_buy_threshold,
        'bb_sell_threshold': manual_bb_sell_threshold,
        'use_atr': manual_use_atr,
        'atr_period': manual_atr_period,
        'atr_multiplier': manual_atr_multiplier,
        'use_msl_msh': manual_use_msl_msh,
        'msl_period': manual_msl_period,
        'msl_lookback': manual_msl_lookback
    }


def _render_signal_summary(params):
    """Render signal summary showing buy/sell conditions."""
    
    st.markdown("---")
    st.markdown("### 📋 Your Strategy Signals")
    
    # Build BUY conditions
    buy_conditions = []
    if params['use_double_ema']:
        buy_conditions.append(f"**Fast EMA ({params['ema_fast']}d) > Slow EMA ({params['ema_slow']}d)**")
    else:
        buy_conditions.append(f"**QQQ Price > {params['ema_period']}-day EMA**")
    
    if params['use_rsi']:
        if params['rsi_threshold'] > 0:
            buy_conditions.append(f"**RSI > {params['rsi_threshold']} (momentum)**")
        buy_conditions.append(f"**RSI < {params['rsi_oversold']} (oversold)**")
    
    if params['use_bb']:
        buy_conditions.append(f"**Price in lower {params['bb_buy_threshold']*100:.0f}% of Bollinger Bands**")
    
    # Build SELL conditions
    sell_conditions = []
    if params['use_double_ema']:
        sell_conditions.append(f"**Fast EMA ({params['ema_fast']}d) < Slow EMA ({params['ema_slow']}d)**")
    else:
        sell_conditions.append(f"**QQQ Price < {params['ema_period']}-day EMA**")
    
    if params['use_rsi']:
        sell_conditions.append(f"**RSI > {params['rsi_overbought']} (overbought)**")
    
    if params['use_stop_loss']:
        sell_conditions.append(f"**Portfolio drops {params['stop_loss_pct']}% from peak**")
    
    if params['use_bb']:
        sell_conditions.append(f"**Price in upper {(1-params['bb_sell_threshold'])*100:.0f}% of Bollinger Bands**")
    
    if params['use_atr']:
        sell_conditions.append(f"**ATR Stop-Loss triggered ({params['atr_multiplier']}x ATR-{params['atr_period']})**")
    
    if params['use_msl_msh']:
        sell_conditions.append(f"**MSL/MSH Stop-Loss triggered (Period: {params['msl_period']}, Lookback: {params['msl_lookback']})**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.success("**🟢 BUY TQQQ Signal**")
        st.markdown("**When ALL conditions are met:**")
        for i, condition in enumerate(buy_conditions, 1):
            st.markdown(f"{i}. {condition}")
        st.caption("→ Enter or hold TQQQ position")
    
    with col2:
        st.error("**🔴 SELL TQQQ Signal**")
        st.markdown("**When ANY condition is met:**")
        for i, condition in enumerate(sell_conditions, 1):
            st.markdown(f"{i}. {condition}")
        st.caption("→ Exit TQQQ and move to cash")


def _apply_manual_configuration(params):
    """Apply manual configuration to session state."""
    
    params_to_apply = {
        'use_ema': True,
        'use_double_ema': params['use_double_ema'],
        'ema_period': params['ema_period'],
        'ema_fast': params['ema_fast'],
        'ema_slow': params['ema_slow'],
        'use_rsi': params['use_rsi'],
        'rsi_threshold': params['rsi_threshold'],
        'rsi_oversold': params['rsi_oversold'],
        'rsi_overbought': params['rsi_overbought'],
        'use_stop_loss': params['use_stop_loss'],
        'stop_loss_pct': params['stop_loss_pct'],
        'use_bb': params['use_bb'],
        'bb_period': params['bb_period'],
        'bb_std_dev': params['bb_std_dev'],
        'bb_buy_threshold': params['bb_buy_threshold'],
        'bb_sell_threshold': params['bb_sell_threshold'],
        'use_atr': params['use_atr'],
        'atr_period': params['atr_period'],
        'atr_multiplier': params['atr_multiplier'],
        'use_msl_msh': params['use_msl_msh'],
        'msl_period': params['msl_period'],
        'msh_period': params['msl_period'],
        'msl_lookback': params['msl_lookback'],
        'msh_lookback': params['msl_lookback']
    }
    
    st.session_state.best_params = params_to_apply
    st.session_state.user_applied_params = True
    st.session_state.applied_rank = "Manual"
    st.session_state.navigate_to_step2 = True
    st.session_state.manual_config_loaded = False
    st.session_state.testing_params_loaded = False
    
    st.success("✅ **Manual Configuration Applied Successfully!**")
    st.markdown("### 📋 Applied Parameters (JSON)")
    st.code(json.dumps(params_to_apply, indent=2), language='json')
    st.info("💡 Parameters saved! Navigate to **Step 4: Testing** to run your strategy.")
