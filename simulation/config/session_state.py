"""Session state initialization for the Streamlit app."""
import streamlit as st


def initialize_session_state():
    """Initialize all session state variables."""
    
    # Wizard navigation
    if 'wizard_step' not in st.session_state:
        st.session_state.wizard_step = 1
    
    # Analysis type
    if 'analysis_type' not in st.session_state:
        st.session_state.analysis_type = "📊 Daily Signal"
    
    # Section visibility
    if 'show_backtest' not in st.session_state:
        st.session_state.show_backtest = True
    if 'show_historical' not in st.session_state:
        st.session_state.show_historical = True
    
    # Manual Configuration parameters
    manual_params = {
        'manual_use_double_ema': False,
        'manual_ema_period': 50,
        'manual_ema_fast': 9,
        'manual_ema_slow': 21,
        'manual_use_rsi': False,
        'manual_rsi_threshold': 50,
        'manual_rsi_oversold': 30,
        'manual_rsi_overbought': 70,
        'manual_use_stop_loss': False,
        'manual_stop_loss_pct': 10,
        'manual_use_bb': False,
        'manual_bb_period': 20,
        'manual_bb_std_dev': 2.0,
        'manual_bb_buy_threshold': 0.2,
        'manual_bb_sell_threshold': 0.8,
        'manual_use_atr': False,
        'manual_atr_period': 14,
        'manual_atr_multiplier': 2.0,
        'manual_use_msl_msh': False,
        'manual_msl_period': 20,
        'manual_msl_lookback': 5,
    }
    
    for key, default_value in manual_params.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    
    # Testing parameters
    testing_params = {
        'use_ema': True,
        'use_double_ema': False,
        'ema_period': 50,
        'ema_fast': 9,
        'ema_slow': 21,
        'use_rsi': False,
        'rsi_threshold': 50,
        'rsi_oversold': 30,
        'rsi_overbought': 70,
        'use_stop_loss': False,
        'stop_loss_pct': 10,
        'use_bb': False,
        'bb_period': 20,
        'bb_std_dev': 2.0,
        'bb_buy_threshold': 0.2,
        'bb_sell_threshold': 0.8,
        'use_atr': False,
        'atr_period': 14,
        'atr_multiplier': 2.0,
        'use_msl_msh': False,
        'msl_period': 20,
        'msh_period': 20,
        'msl_lookback': 5,
        'msh_lookback': 5,
    }
    
    for key, default_value in testing_params.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
    
    # Grid search results
    if 'grid_search_results' not in st.session_state:
        st.session_state.grid_search_results = None
    if 'best_params' not in st.session_state:
        st.session_state.best_params = None
    if 'show_grid_search' not in st.session_state:
        st.session_state.show_grid_search = True
