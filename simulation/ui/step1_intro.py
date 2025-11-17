"""Step 1: Introduction."""
import streamlit as st

try:
    rerun = st.rerun
except AttributeError:
    rerun = st.experimental_rerun


def render_step1():
    """Render Step 1: Introduction page."""
    st.markdown("### Welcome to the TQQQ Trading Strategy Wizard")
    
    st.markdown("""
    This wizard will guide you through creating and testing a leveraged ETF trading strategy.
    
    ### 🎯 What You'll Do:
    
    **Step 1: Introduction** (You are here)
    - Understand the workflow
    
    **Step 2: Find Best Signals**
    - Enable features you want to test
    - Grid search algorithm tests thousands of combinations
    - Find the best strategy for your risk tolerance
    - Compare performance across different time periods
    
    **Step 3: Verify & Customize**
    - Review the selected strategy
    - Manually adjust parameters if needed
    - Understand the trading signals
    
    **Step 4: Testing**
    - Get today's buy/sell signal
    - Run custom backtests
    - Perform Monte Carlo simulations
    
    **Step 5: AI Summary**
    - Generate comprehensive AI-friendly reports
    - Export detailed analysis for any AI tool
    - Download complete strategy documentation
    - Share results with ChatGPT, Claude, or other AI assistants
    
    ### ⚠️ Important Notes:
    
    - **TQQQ is 3x leveraged** - High risk, high reward
    - **Past performance ≠ future results**
    - **Educational purposes only** - Not financial advice
    - **Always use stop-losses** - Protect your capital
    
    ### 🚀 Ready to Start?
    
    Click **Step 2: Find Best Signals** above to begin finding your optimal strategy!
    """)
    
    # Navigation buttons
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.button("⬅️ Previous Step", disabled=True, use_container_width=True, help="Already at first step")
    
    with col2:
        pass  # Empty middle column for spacing
    
    with col3:
        if st.button("Next Step ➡️", type="primary", use_container_width=True, key="step1_next"):
            st.session_state.wizard_step = 2
            rerun()
    
    # Add bottom padding for mobile visibility
    st.markdown("<br><br>", unsafe_allow_html=True)
