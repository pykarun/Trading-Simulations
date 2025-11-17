"""TQQQ Trading Strategy - Main Application

A 5-step wizard interface for creating and testing leveraged ETF trading strategies.

Steps:
1. Introduction - Overview and workflow
2. Find Best Signals - Grid search algorithm finds optimal parameters
3. Verify & Customize - Review and adjust strategy
4. Testing - Daily signals, backtests, Monte Carlo
5. AI Summary - Generate comprehensive AI-friendly reports

Author: Trading Strategy Wizard
Version: 2.1 (AI Summary Export)
"""

import streamlit as st

# Import configuration
from config import setup_page, initialize_session_state

# Import UI components
from ui import (
    render_navigation,
    render_step1,
    render_step2,
    render_step3,
    render_step4,
    render_step5
)

# Compatibility layer for st.rerun
try:
    rerun = st.rerun
except AttributeError:
    rerun = st.experimental_rerun


def main():
    """Main application entry point."""
    
    # Setup page configuration and styling
    setup_page()
    
    # Initialize session state
    initialize_session_state()
    
    # Add separator
    st.markdown("---")
    
    # Render wizard navigation
    render_navigation()
    
    # Get current wizard step
    wizard_step = st.session_state.wizard_step
    
    # Render appropriate step
    if wizard_step == 1:
        render_step1()
    elif wizard_step == 2:
        render_step2()
    elif wizard_step == 3:
        render_step3()
    elif wizard_step == 4:
        render_step4()
    elif wizard_step == 5:
        render_step5()
    else:
        # Fallback to step 1 if invalid step
        st.session_state.wizard_step = 1
        rerun()


if __name__ == "__main__":
    main()
