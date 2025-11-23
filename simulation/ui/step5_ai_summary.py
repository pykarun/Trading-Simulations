"""Step 5: AI Summary Export - Generate comprehensive AI-friendly reports.

This module provides AI-ready summary exports for all testing results:
1. Daily Signal Summary
2. Custom Simulation Summary
3. Monte Carlo Simulation Summary
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime

try:
    rerun = st.rerun
except AttributeError:
    rerun = st.experimental_rerun


def render_step5():
    """Render Step 5: AI Summary Export interface."""
    
    st.markdown("## 📄 Step 5: AI Summary Export")
    st.markdown("---")
    
    st.info("""
    **Generate comprehensive AI-friendly reports** that can be used with ChatGPT, Claude, or other AI tools for:
    - Strategy analysis and validation
    - Performance evaluation
    - Risk assessment
    - Decision support
    - Documentation and sharing
    """)
    
    # Check if any test results exist
    has_daily_signal = st.session_state.get('daily_signal_results') is not None
    has_custom_sim = st.session_state.get('custom_sim_results') is not None
    has_monte_carlo = st.session_state.get('monte_carlo_results') is not None
    
    if not (has_daily_signal or has_custom_sim or has_monte_carlo):
        st.warning("⚠️ No test results available. Please run tests in Step 4 first.")
        st.markdown("---")
        return
    
    st.markdown("---")
    st.markdown("### 📊 Available Reports")
    
    # Create tabs for different report types
    tabs = []
    if has_daily_signal:
        tabs.append("📊 Daily Signal")
    if has_custom_sim:
        tabs.append("📈 Custom Simulation")
    if has_monte_carlo:
        tabs.append("🎲 Monte Carlo")
    
    if len(tabs) == 0:
        st.warning("No reports available.")
        return
    
    tab_objects = st.tabs(tabs)
    
    tab_idx = 0
    if has_daily_signal:
        with tab_objects[tab_idx]:
            _render_daily_signal_report()
        tab_idx += 1
    
    if has_custom_sim:
        with tab_objects[tab_idx]:
            _render_custom_simulation_report()
        tab_idx += 1
    
    if has_monte_carlo:
        with tab_objects[tab_idx]:
            _render_monte_carlo_report()
    
    # Add bottom padding for mobile visibility
    st.markdown("<br><br>", unsafe_allow_html=True)

    # Navigation buttons
    _render_navigation_buttons()


def _render_daily_signal_report():
    """Render daily signal AI summary report."""
    
    results = st.session_state.daily_signal_results
    params = results['params']
    
    st.markdown("### 📊 Daily Signal AI Summary")
    st.caption("Comprehensive report of today's trading signal with full context")
    
    ai_summary = _generate_daily_signal_ai_summary(
        params,
        results['date'],
        results['qqq_price'],
        results['tqqq_price'],
        results['ema'],
        results['rsi'],
        results['signal'],
        results['market_state'],
        results.get('ema_fast'),
        results.get('ema_slow'),
        results.get('bb_position')
    )
    
    with st.expander("📋 View Full Report", expanded=True):
        st.text_area("AI Summary (Copy for AI Analysis)", ai_summary, height=500, key="daily_summary_view", label_visibility="collapsed")
        st.caption("💡 **Tip:** Click inside the text area above, press Ctrl+A (or Cmd+A on Mac) to select all, then Ctrl+C (or Cmd+C) to copy")
    
    st.download_button(
        "📥 Download as TXT",
        ai_summary,
        f"daily_signal_ai_summary_{results['date'].strftime('%Y%m%d')}.txt",
        "text/plain",
        use_container_width=True
    )


def _render_custom_simulation_report():
    """Render custom simulation AI summary report."""
    
    results = st.session_state.custom_sim_results
    params = results['params']
    
    st.markdown("### 📈 Custom Simulation AI Summary")
    st.caption("Detailed backtest analysis with performance metrics and trade logs")
    
    ai_summary = _generate_custom_simulation_ai_summary(
        params,
        results['start_date'],
        results['end_date'],
        results['initial_capital'],
        results['result'],
        results['qqq_bh_value'],
        results['qqq_bh_return']
    )
    
    with st.expander("📋 View Full Report", expanded=True):
        st.text_area("AI Summary (Copy for AI Analysis)", ai_summary, height=500, key="custom_summary_view", label_visibility="collapsed")
        st.caption("💡 **Tip:** Click inside the text area above, press Ctrl+A (or Cmd+A on Mac) to select all, then Ctrl+C (or Cmd+C) to copy")
    
    st.download_button(
        "📥 Download as TXT",
        ai_summary,
        f"custom_simulation_ai_summary_{results['start_date'].strftime('%Y%m%d')}_{results['end_date'].strftime('%Y%m%d')}.txt",
        "text/plain",
        use_container_width=True
    )


def _render_monte_carlo_report():
    """Render Monte Carlo simulation AI summary report."""
    
    results = st.session_state.monte_carlo_results
    params = results['params']
    
    st.markdown("### 🎲 Monte Carlo Simulation AI Summary")
    st.caption("Probabilistic analysis with risk metrics and outcome distributions")
    
    ai_summary = _generate_monte_carlo_ai_summary(
        params,
        results['start_date'],
        results['end_date'],
        results['initial_capital'],
        results['num_simulations'],
        results['simulation_days'],
        results['confidence_level'],
        results['mean_final_value'],
        results['median_final_value'],
        results['ci_lower'],
        results['ci_upper'],
        results['prob_profit'],
        results['qqq_mean_final_value'],
        results['prob_outperform'],
        results['mean_outperformance'],
        results['final_values']
    )
    
    with st.expander("📋 View Full Report", expanded=True):
        st.text_area("AI Summary (Copy for AI Analysis)", ai_summary, height=500, key="monte_summary_view", label_visibility="collapsed")
        st.caption("💡 **Tip:** Click inside the text area above, press Ctrl+A (or Cmd+A on Mac) to select all, then Ctrl+C (or Cmd+C) to copy")
    
    st.download_button(
        "📥 Download as TXT",
        ai_summary,
        f"monte_carlo_ai_summary_{results['start_date'].strftime('%Y%m%d')}_{results['end_date'].strftime('%Y%m%d')}.txt",
        "text/plain",
        use_container_width=True
    )





def _render_navigation_buttons():
    """Render navigation buttons."""
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        if st.button("⬅️ Previous Step", use_container_width=True, key="step5_prev"):
            st.session_state.wizard_step = 4
            rerun()
    
    with col2:
        pass  # Empty middle column for spacing
    
    with col3:
        st.button("Next Step ➡️", disabled=True, use_container_width=True, help="Already at last step")
    
    # Add bottom padding for mobile visibility
    st.markdown("<br><br>", unsafe_allow_html=True)


def _generate_daily_signal_ai_summary(params, date, qqq_price, tqqq_price, ema, rsi, signal, market_state, ema_fast=None, ema_slow=None, bb_position=None):

    """Generate AI-friendly summary for daily signal."""
    
    summary = f"""
================================================================================
TQQQ TRADING STRATEGY - DAILY SIGNAL AI SUMMARY
================================================================================

REPORT GENERATED: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
SIGNAL DATE: {date.strftime('%Y-%m-%d')}
MARKET STATE: {market_state}

================================================================================
1. STRATEGY CONFIGURATION
================================================================================

EMA Strategy:
  - Type: {'Double EMA Crossover' if params['use_double_ema'] else 'Single EMA'}
"""
    
    if params['use_double_ema']:
        summary += f"""  - Fast EMA Period: {params['ema_fast']} days (Current: ${ema_fast:.2f})
  - Slow EMA Period: {params['ema_slow']} days (Current: ${ema_slow:.2f})
"""
    else:
        summary += f"""  - EMA Period: {params['ema_period']} days
  - Current EMA: ${ema:.2f}
"""

    summary += f"""
RSI Filter:
  - Enabled: {params['use_rsi']}
  - Momentum Threshold: {params['rsi_threshold']} (Buy when RSI > threshold)
  - Oversold Level: {params['rsi_oversold']} (Buy signal)
  - Overbought Level: {params['rsi_overbought']} (Sell signal)
  - Current RSI: {rsi:.2f}

Stop-Loss:
  - Enabled: {params['use_stop_loss']}
  - Stop-Loss Percentage: {params['stop_loss_pct']}% (Exit if portfolio drops X% from peak)

Bollinger Bands:
  - Enabled: {params['use_bb']}
"""
    
    if params['use_bb']:
        summary += f"""  - Period: {params['bb_period']} days
  - Standard Deviation: {params['bb_std_dev']}
  - Buy Threshold: {params['bb_buy_threshold']} (lower band position)
  - Sell Threshold: {params['bb_sell_threshold']} (upper band position)
  - Current BB Position: {bb_position:.2f} (0.0=lower, 0.5=middle, 1.0=upper)
"""
    
    summary += f"""
ATR Stop-Loss:
  - Enabled: {params['use_atr']}
"""
    if params['use_atr']:
        summary += f"""  - ATR Period: {params['atr_period']} days
  - ATR Multiplier: {params['atr_multiplier']}x
"""
    
    summary += f"""
MSL/MSH Stop-Loss:
  - Enabled: {params['use_msl_msh']}
"""
    if params['use_msl_msh']:
        summary += f"""  - MSL Period: {params['msl_period']} days
  - MSL Lookback: {params['msl_lookback']} days
"""

    summary += f"""
================================================================================
2. CURRENT MARKET DATA
================================================================================

QQQ (Nasdaq-100 ETF):
  - Current Price: ${qqq_price:.2f}
  - EMA Reference: ${ema:.2f}
  - Price vs EMA: {'ABOVE' if qqq_price > ema else 'BELOW'} ({((qqq_price/ema - 1) * 100):+.2f}%)

TQQQ (3x Leveraged Nasdaq-100 ETF):
  - Current Price: ${tqqq_price:.2f}
  - Leverage: 3x daily returns of QQQ
  - Risk Level: HIGH (leveraged ETF)

Technical Indicators:
  - RSI (14-day): {rsi:.2f}
  - RSI Status: {'Overbought' if rsi > 70 else 'Oversold' if rsi < 30 else 'Neutral'}

================================================================================
3. SIGNAL ANALYSIS
================================================================================

FINAL SIGNAL: {signal}

Signal Logic:
"""

    if params['use_double_ema']:
        summary += f"""  1. Base Signal: {'BUY' if ema_fast > ema_slow else 'SELL'} (Fast EMA {'>' if ema_fast > ema_slow else '<'} Slow EMA)
"""
    else:
        summary += f"""  1. Base Signal: {'BUY' if qqq_price > ema else 'SELL'} (QQQ Price {'>' if qqq_price > ema else '<'} EMA)
"""
    
    if params['use_rsi']:
        summary += f"""  2. RSI Filter: {'PASS' if rsi > params['rsi_threshold'] else 'FAIL'} (RSI {rsi:.2f} {'>' if rsi > params['rsi_threshold'] else '<'} {params['rsi_threshold']})
"""
    
    if params['use_bb'] and bb_position is not None:
        summary += f"""  3. Bollinger Bands: Position {bb_position:.2f}
"""

    summary += f"""
================================================================================
4. RECOMMENDED ACTION
================================================================================

"""
    
    if signal == 'BUY':
        summary += """ACTION: BUY TQQQ (or HOLD if already in position)

Rationale:
  - All buy conditions are met
  - Trend indicators suggest upward momentum
  - Entry signal is active

Risk Management:
  - Use stop-loss orders to protect capital
  - Monitor position daily before market close (3:55 PM ET)
  - Be prepared for high volatility due to 3x leverage
"""
    else:
        summary += """ACTION: SELL TQQQ / STAY IN CASH

Rationale:
  - Sell conditions are met or buy conditions not satisfied
  - Risk management suggests exiting position
  - Preserve capital in cash until next buy signal

Next Steps:
  - Wait for buy signal to re-enter
  - Monitor market conditions daily
  - Keep capital safe in cash/money market
"""

    summary += """
================================================================================
5. TESTING METHODOLOGY
================================================================================

Data Source: Yahoo Finance (yfinance)
Analysis Time: End of day (3:55 PM ET recommended)
Execution Window: 3:55 PM - 4:00 PM ET (before market close)

Signal Generation Process:
  1. Fetch latest market data for QQQ and TQQQ
  2. Calculate technical indicators (EMA, RSI, BB, etc.)
  3. Apply strategy rules to determine buy/sell signal
  4. Generate recommendation with risk management guidelines

Limitations:
  - Signals based on historical patterns (past ≠ future)
  - Does not account for news, earnings, or macro events
  - Leveraged ETFs have decay risk over time
  - Requires disciplined execution and risk management

================================================================================
6. DISCLAIMER
================================================================================

This is an educational tool for learning about algorithmic trading strategies.
This is NOT financial advice. Trading leveraged ETFs involves significant risk
of loss. Past performance does not guarantee future results. Always:
  - Do your own research
  - Understand the risks
  - Never invest more than you can afford to lose
  - Consider consulting a licensed financial advisor

================================================================================
SAMPLE AI QUESTIONS
================================================================================

Copy this report and paste it into ChatGPT, Claude, or any AI assistant along
with one of these questions:

1. "Validate this trading strategy and tell me if I should take this signal"

2. "What are the main risks I should be aware of with this signal?"

3. "Based on the current market conditions and indicators, how confident 
   should I be in this signal?"

4. "Compare this signal to typical market conditions - is this a strong or 
   weak signal?"

5. "What additional factors should I consider before executing this trade?"

6. "Explain this strategy in simple terms and whether it makes sense for 
   a beginner trader"

7. "What's the probability this signal will be profitable based on the 
   technical indicators shown?"

8. "Should I adjust my position size based on these indicators? If so, how?"

9. "What are the key things to monitor after taking this signal?"

10. "Create a risk management plan for this trade based on the strategy 
    parameters"

================================================================================
END OF DAILY SIGNAL AI SUMMARY
================================================================================
"""
    
    return summary


def _generate_custom_simulation_ai_summary(params, start_date, end_date, initial_capital, result, qqq_bh_value, qqq_bh_return):
    """Generate AI-friendly summary for custom simulation."""
    
    days = (end_date - start_date).days
    years = days / 365.25
    cagr_strategy = ((result['final_value'] / initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
    cagr_qqq = ((qqq_bh_value / initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
    outperformance = result['total_return_pct'] - qqq_bh_return
    
    summary = f"""
================================================================================
TQQQ TRADING STRATEGY - CUSTOM SIMULATION AI SUMMARY
================================================================================

REPORT GENERATED: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
BACKTEST PERIOD: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}
DURATION: {days} days ({years:.2f} years)

================================================================================
1. STRATEGY CONFIGURATION
================================================================================

EMA Strategy:
  - Type: {'Double EMA Crossover' if params['use_double_ema'] else 'Single EMA'}
"""
    
    if params['use_double_ema']:
        summary += f"""  - Fast EMA Period: {params['ema_fast']} days
  - Slow EMA Period: {params['ema_slow']} days
  - Signal: Buy when Fast EMA > Slow EMA, Sell when Fast EMA < Slow EMA
"""
    else:
        summary += f"""  - EMA Period: {params['ema_period']} days
  - Signal: Buy when QQQ Price > EMA, Sell when QQQ Price < EMA
"""
    
    summary += f"""
RSI Filter:
  - Enabled: {params['use_rsi']}
  - Momentum Threshold: {params['rsi_threshold']}
  - Oversold Level: {params['rsi_oversold']}
  - Overbought Level: {params['rsi_overbought']}

Stop-Loss:
  - Enabled: {params['use_stop_loss']}
  - Stop-Loss Percentage: {params['stop_loss_pct']}%

Bollinger Bands:
  - Enabled: {params['use_bb']}
"""
    
    if params['use_bb']:
        summary += f"""  - Period: {params['bb_period']} days
  - Standard Deviation: {params['bb_std_dev']}
  - Buy Threshold: {params['bb_buy_threshold']}
  - Sell Threshold: {params['bb_sell_threshold']}
"""
    
    summary += f"""
ATR Stop-Loss:
  - Enabled: {params['use_atr']}
"""
    if params['use_atr']:
        summary += f"""  - ATR Period: {params['atr_period']} days
  - ATR Multiplier: {params['atr_multiplier']}x
"""
    
    summary += f"""
MSL/MSH Stop-Loss:
  - Enabled: {params['use_msl_msh']}
"""
    if params['use_msl_msh']:
        summary += f"""  - MSL Period: {params['msl_period']} days
  - MSL Lookback: {params['msl_lookback']} days
"""

    summary += f"""
================================================================================
2. BACKTEST PARAMETERS
================================================================================

Initial Capital: ${initial_capital:,.2f}
Start Date: {start_date.strftime('%Y-%m-%d')}
End Date: {end_date.strftime('%Y-%m-%d')}
Trading Days: {days}
Years: {years:.2f}

Execution Model:
  - Analysis Time: 3:50 PM ET (10 minutes before close)
  - Execution Time: 3:55 PM ET (5 minutes before close)
  - Slippage (Buy): {result['slippage_buy_pct']:.2f}%
  - Slippage (Sell): {result['slippage_sell_pct']:.2f}%
  - Total Execution Costs: {result['estimated_total_costs_pct']:.2f}%

================================================================================
3. STRATEGY PERFORMANCE RESULTS
================================================================================

Final Portfolio Value: ${result['final_value']:,.2f}
Total Return: {result['total_return_pct']:.2f}%
CAGR (Compound Annual Growth Rate): {cagr_strategy:.2f}%
Maximum Drawdown: {result['max_drawdown']:.2f}%
Total Trades: {result['num_trades']}

Risk Metrics:
  - Sharpe Ratio: {result.get('sharpe_ratio', 'N/A')}
  - Win Rate: {result.get('win_rate', 'N/A')}
  - Average Trade: {result.get('avg_trade', 'N/A')}

================================================================================
4. BENCHMARK COMPARISON (QQQ Buy & Hold)
================================================================================

QQQ Final Value: ${qqq_bh_value:,.2f}
QQQ Total Return: {qqq_bh_return:.2f}%
QQQ CAGR: {cagr_qqq:.2f}%

Strategy vs QQQ:
  - Outperformance: {outperformance:+.2f}%
  - Value Difference: ${result['final_value'] - qqq_bh_value:+,.2f}
  - Result: {'✅ WINNING' if outperformance > 0 else '❌ LOSING'}

================================================================================
5. TRADE LOG SUMMARY
================================================================================

Total Trades: {result['num_trades']}
Trade Log Entries: {len(result['trade_log'])}

Sample Trades (First 10):
"""
    
    # Add first 10 trades
    trade_df = pd.DataFrame(result['trade_log'])
    trades_only = trade_df[trade_df['Action'].str.contains('BUY TQQQ|SELL', case=False, na=False)]
    
    for idx, trade in trades_only.head(10).iterrows():
        summary += f"\n  {trade['Date']} | {trade['Action']} | Price: ${trade.get('TQQQ_Price', 'N/A')} | Portfolio: ${trade.get('Portfolio_Value', 'N/A')}"
    
    if len(trades_only) > 10:
        summary += f"\n\n  ... and {len(trades_only) - 10} more trades (see full trade log download)"
    
    summary += f"""

================================================================================
6. TESTING METHODOLOGY
================================================================================

Data Source: Yahoo Finance (yfinance)
Tickers: QQQ (Nasdaq-100 ETF), TQQQ (3x Leveraged Nasdaq-100 ETF)

Backtest Process:
  1. Download historical price data for QQQ and TQQQ
  2. Calculate technical indicators (EMA, RSI, BB, ATR, MSL/MSH)
  3. Apply strategy rules day-by-day
  4. Simulate realistic execution with slippage
  5. Track portfolio value, trades, and drawdowns
  6. Compare against QQQ buy-and-hold benchmark

Execution Simulation:
  - Signals generated at 3:50 PM ET (10 min before close)
  - Orders executed at 3:55 PM ET (5 min before close)
  - Slippage accounts for:
    * Bid-ask spread
    * Market impact
    * 5-minute price movement
    * Execution uncertainty

Limitations:
  - Based on historical data (past ≠ future)
  - Does not account for:
    * Dividends
    * Stock splits (auto-adjusted)
    * Trading halts
    * Extreme market events
    * Psychological factors
  - Leveraged ETF decay not explicitly modeled
  - Assumes sufficient liquidity for all trades

================================================================================
7. KEY INSIGHTS
================================================================================

Performance Analysis:
  - Strategy {'outperformed' if outperformance > 0 else 'underperformed'} QQQ by {abs(outperformance):.2f}%
  - CAGR of {cagr_strategy:.2f}% vs QQQ's {cagr_qqq:.2f}%
  - Maximum drawdown of {result['max_drawdown']:.2f}% (risk measure)
  - Executed {result['num_trades']} trades over {years:.2f} years

Risk Assessment:
  - {'HIGH' if result['max_drawdown'] > 30 else 'MODERATE' if result['max_drawdown'] > 15 else 'LOW'} risk based on max drawdown
  - Leveraged ETF (3x) amplifies both gains and losses
  - Stop-loss {'ENABLED' if params['use_stop_loss'] else 'DISABLED'} - {'helps limit losses' if params['use_stop_loss'] else 'no downside protection'}

================================================================================
8. DISCLAIMER
================================================================================

This is an educational tool for learning about algorithmic trading strategies.
This is NOT financial advice. Trading leveraged ETFs involves significant risk
of loss. Past performance does not guarantee future results.

IMPORTANT WARNINGS:
  - Leveraged ETFs (like TQQQ) can lose value rapidly
  - 3x leverage means 3x the daily volatility
  - Decay risk: leveraged ETFs lose value over time in sideways markets
  - Not suitable for long-term buy-and-hold
  - Requires active monitoring and risk management

Always:
  - Do your own research
  - Understand the risks
  - Never invest more than you can afford to lose
  - Consider consulting a licensed financial advisor
  - Test strategies thoroughly before risking real capital

================================================================================
SAMPLE AI QUESTIONS
================================================================================

Copy this report and paste it into ChatGPT, Claude, or any AI assistant along
with one of these questions:

1. "Validate this backtest strategy and tell me if it's worth implementing 
   with real money"

2. "What are the biggest weaknesses in this strategy based on the backtest 
   results?"

3. "Is the outperformance vs QQQ statistically significant or could it be 
   due to luck?"

4. "Analyze the risk-adjusted returns and tell me if this strategy is too 
   risky for my portfolio"

5. "What market conditions does this strategy perform best in? What about 
   worst?"

6. "How should I adjust the parameters to reduce the maximum drawdown while 
   maintaining good returns?"

7. "Based on the trade log, identify any patterns or issues with the entry/exit 
   timing"

8. "Compare this strategy to a simple buy-and-hold approach - which is better 
   for long-term investing?"

9. "What position sizing would you recommend based on these backtest results?"

10. "Create an implementation plan including risk management rules based on 
    this backtest"

11. "What additional tests should I run before trusting this strategy with 
    real capital?"

12. "Explain the execution costs and slippage - are they realistic for retail 
    trading?"

================================================================================
END OF CUSTOM SIMULATION AI SUMMARY
================================================================================
"""
    
    return summary


def _generate_monte_carlo_ai_summary(params, start_date, end_date, initial_capital, num_simulations, simulation_days,
                                      confidence_level, mean_final_value, median_final_value, ci_lower, ci_upper,
                                      prob_profit, qqq_mean_final_value, prob_outperform, mean_outperformance, final_values):
    """Generate AI-friendly summary for Monte Carlo simulation."""
    
    mean_return = (mean_final_value - initial_capital) / initial_capital * 100
    median_return = (median_final_value - initial_capital) / initial_capital * 100
    ci_lower_return = (ci_lower - initial_capital) / initial_capital * 100
    ci_upper_return = (ci_upper - initial_capital) / initial_capital * 100
    qqq_mean_return = (qqq_mean_final_value - initial_capital) / initial_capital * 100
    
    worst_case = np.min(final_values)
    best_case = np.max(final_values)
    worst_case_return = (worst_case - initial_capital) / initial_capital * 100
    best_case_return = (best_case - initial_capital) / initial_capital * 100
    std_dev = np.std(final_values)
    
    summary = f"""
================================================================================
TQQQ TRADING STRATEGY - MONTE CARLO SIMULATION AI SUMMARY
================================================================================

REPORT GENERATED: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
HISTORICAL DATA PERIOD: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}
SIMULATION PARAMETERS: {num_simulations:,} simulations over {simulation_days} days ({simulation_days/252:.1f} years)

================================================================================
1. STRATEGY CONFIGURATION
================================================================================

EMA Strategy:
  - Type: {'Double EMA Crossover' if params['use_double_ema'] else 'Single EMA'}
"""
    
    if params['use_double_ema']:
        summary += f"""  - Fast EMA Period: {params['ema_fast']} days
  - Slow EMA Period: {params['ema_slow']} days
"""
    else:
        summary += f"""  - EMA Period: {params['ema_period']} days
"""
    
    summary += f"""
RSI Filter: {'Enabled' if params['use_rsi'] else 'Disabled'}
Stop-Loss: {'Enabled (' + str(params['stop_loss_pct']) + '%)' if params['use_stop_loss'] else 'Disabled'}
Bollinger Bands: {'Enabled' if params['use_bb'] else 'Disabled'}
ATR Stop-Loss: {'Enabled' if params['use_atr'] else 'Disabled'}
MSL/MSH Stop-Loss: {'Enabled' if params['use_msl_msh'] else 'Disabled'}

================================================================================
2. MONTE CARLO METHODOLOGY
================================================================================

Simulation Approach:
  1. Run strategy on historical data ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})
  2. Extract daily returns from historical backtest
  3. Randomly sample from historical returns
  4. Generate {num_simulations:,} possible future paths
  5. Each path simulates {simulation_days} trading days
  6. Calculate statistics across all simulations

Random Sampling:
  - Method: Bootstrap resampling with replacement
  - Seed: 42 (for reproducibility)
  - Assumption: Future returns will be similar to historical returns
  - Limitation: Does not account for regime changes or black swan events

Confidence Interval: {confidence_level}%
  - Lower Bound: {(100 - confidence_level) / 2:.1f}th percentile
  - Upper Bound: {100 - (100 - confidence_level) / 2:.1f}th percentile

================================================================================
3. SIMULATION RESULTS - STRATEGY PERFORMANCE
================================================================================

Initial Capital: ${initial_capital:,.2f}
Simulation Period: {simulation_days} days ({simulation_days/252:.1f} years)

Central Tendency:
  - Mean Final Value: ${mean_final_value:,.2f} ({mean_return:+.2f}%)
  - Median Final Value: ${median_final_value:,.2f} ({median_return:+.2f}%)

Confidence Interval ({confidence_level}%):
  - Lower Bound: ${ci_lower:,.2f} ({ci_lower_return:+.2f}%)
  - Upper Bound: ${ci_upper:,.2f} ({ci_upper_return:+.2f}%)

Risk Metrics:
  - Standard Deviation: ${std_dev:,.2f}
  - Probability of Profit: {prob_profit:.1f}%
  - Worst Case: ${worst_case:,.2f} ({worst_case_return:+.2f}%)
  - Best Case: ${best_case:,.2f} ({best_case_return:+.2f}%)
  - Value at Risk ({confidence_level}% CI): ${initial_capital - ci_lower:,.2f}

================================================================================
4. BENCHMARK COMPARISON (QQQ Buy & Hold)
================================================================================

QQQ Mean Final Value: ${qqq_mean_final_value:,.2f} ({qqq_mean_return:+.2f}%)

Strategy vs QQQ:
  - Probability of Outperforming QQQ: {prob_outperform:.1f}%
  - Mean Outperformance: ${mean_outperformance:+,.2f}
  - Mean Return Difference: {mean_return - qqq_mean_return:+.2f}%
  - Result: {'✅ LIKELY TO WIN' if prob_outperform > 50 else '❌ LIKELY TO LOSE'}

Interpretation:
  - In {prob_outperform:.0f}% of simulations, the strategy beats QQQ
  - In {100 - prob_outperform:.0f}% of simulations, QQQ beats the strategy
  - {'Strategy has edge over buy-and-hold' if prob_outperform > 50 else 'Buy-and-hold has edge over strategy'}

================================================================================
5. PROBABILITY DISTRIBUTION ANALYSIS
================================================================================

Outcome Probabilities:
  - Probability of Profit (any gain): {prob_profit:.1f}%
  - Probability of Loss: {100 - prob_profit:.1f}%
  - Probability of Beating QQQ: {prob_outperform:.1f}%

Expected Value Analysis:
  - Mean Expected Return: {mean_return:.2f}%
  - Median Expected Return: {median_return:.2f}%
  - Risk-Adjusted Return: {mean_return / (std_dev / initial_capital * 100) if std_dev > 0 else 0:.2f}

Percentile Breakdown:
  - 10th Percentile: ${np.percentile(final_values, 10):,.2f}
  - 25th Percentile: ${np.percentile(final_values, 25):,.2f}
  - 50th Percentile (Median): ${median_final_value:,.2f}
  - 75th Percentile: ${np.percentile(final_values, 75):,.2f}
  - 90th Percentile: ${np.percentile(final_values, 90):,.2f}

================================================================================
6. RISK ASSESSMENT
================================================================================

Risk Level: {'HIGH' if std_dev / initial_capital > 0.3 else 'MODERATE' if std_dev / initial_capital > 0.15 else 'LOW'}

Risk Factors:
  - Volatility (Std Dev): ${std_dev:,.2f} ({std_dev / initial_capital * 100:.1f}% of capital)
  - Downside Risk: {100 - prob_profit:.1f}% chance of loss
  - Maximum Observed Loss: ${worst_case - initial_capital:+,.2f} ({worst_case_return:+.2f}%)
  - Value at Risk: ${initial_capital - ci_lower:,.2f} ({(initial_capital - ci_lower) / initial_capital * 100:.1f}% of capital)

Risk Mitigation:
  - Stop-Loss: {'✅ ENABLED (' + str(params['stop_loss_pct']) + '%)' if params['use_stop_loss'] else '❌ DISABLED'}
  - Position Sizing: Consider risking only 1-5% of total capital
  - Diversification: Don't put all capital in one strategy
  - Monitoring: Review performance regularly

================================================================================
7. KEY INSIGHTS & RECOMMENDATIONS
================================================================================

Performance Summary:
  - Expected return: {mean_return:.2f}% over {simulation_days/252:.1f} years
  - {prob_outperform:.0f}% chance of beating QQQ buy-and-hold
  - {prob_profit:.0f}% chance of making any profit
  - Risk level: {'HIGH' if std_dev / initial_capital > 0.3 else 'MODERATE' if std_dev / initial_capital > 0.15 else 'LOW'}

Recommendations:
"""
    
    if prob_outperform > 60 and prob_profit > 70:
        summary += """  ✅ FAVORABLE: Strategy shows strong probability of success
  - Consider implementing with proper risk management
  - Use position sizing (don't risk entire capital)
  - Monitor performance and adjust as needed
"""
    elif prob_outperform > 50 and prob_profit > 60:
        summary += """  ⚠️ MODERATE: Strategy shows slight edge but with uncertainty
  - Proceed with caution and smaller position sizes
  - Consider paper trading first
  - Have clear exit rules
"""
    else:
        summary += """  ❌ UNFAVORABLE: Strategy shows low probability of success
  - Consider revising strategy parameters
  - May be better to stick with QQQ buy-and-hold
  - If proceeding, use very small position sizes
"""
    
    summary += f"""
Important Considerations:
  - Monte Carlo assumes future similar to past (may not hold)
  - Does not account for black swan events
  - Leveraged ETFs have decay risk not fully captured
  - Requires disciplined execution and emotional control
  - Past performance does not guarantee future results

================================================================================
8. SIMULATION STATISTICS
================================================================================

Total Simulations Run: {num_simulations:,}
Simulation Length: {simulation_days} days ({simulation_days/252:.1f} years)
Historical Data Period: {(end_date - start_date).days} days
Random Seed: 42 (reproducible results)

Distribution Characteristics:
  - Mean: ${mean_final_value:,.2f}
  - Median: ${median_final_value:,.2f}
  - Std Dev: ${std_dev:,.2f}
  - Skewness: {((final_values - mean_final_value) ** 3).mean() / (std_dev ** 3) if std_dev > 0 else 0:.2f}
  - Min: ${worst_case:,.2f}
  - Max: ${best_case:,.2f}

================================================================================
9. DISCLAIMER
================================================================================

This Monte Carlo simulation is an educational tool for understanding potential
future outcomes based on historical performance. This is NOT financial advice.

CRITICAL WARNINGS:
  - Simulations are based on historical data (past ≠ future)
  - Cannot predict black swan events or regime changes
  - Leveraged ETFs (TQQQ) have unique risks:
    * 3x daily leverage amplifies volatility
    * Decay risk in sideways markets
    * Can lose value rapidly in downturns
  - Probabilities are estimates, not guarantees
  - Real trading involves emotions, slippage, and execution risk

Always:
  - Do your own research and due diligence
  - Understand the risks before trading
  - Never invest more than you can afford to lose
  - Consider consulting a licensed financial advisor
  - Test strategies thoroughly before risking real capital
  - Use proper risk management and position sizing

================================================================================
SAMPLE AI QUESTIONS
================================================================================

Copy this report and paste it into ChatGPT, Claude, or any AI assistant along
with one of these questions:

1. "Validate this Monte Carlo simulation and tell me if this strategy is 
   worth the risk"

2. "Based on the probability distribution, what's a realistic expectation 
   for returns over the next year?"

3. "The simulation shows X% chance of beating QQQ - is that good enough to 
   justify the extra risk?"

4. "Analyze the worst-case scenario - how should I prepare for it?"

5. "What does the confidence interval tell me about the reliability of this 
   strategy?"

6. "Based on the Value at Risk, how much capital should I allocate to this 
   strategy?"

7. "Compare the mean vs median returns - what does the difference tell us 
   about the strategy?"

8. "Is the probability of profit high enough to justify using a leveraged 
   ETF like TQQQ?"

9. "What position sizing strategy would you recommend based on these Monte 
   Carlo results?"

10. "Explain the risk-adjusted return and whether it's attractive compared 
    to safer alternatives"

11. "What are the limitations of this Monte Carlo simulation that I should 
    be aware of?"

12. "Based on the percentile breakdown, what's a conservative, moderate, and 
    aggressive return expectation?"

13. "Should I implement this strategy given the probability of outperforming 
    QQQ is X%?"

14. "Create a risk management framework based on these simulation results"

15. "What market conditions would invalidate these Monte Carlo projections?"

================================================================================
END OF MONTE CARLO SIMULATION AI SUMMARY
================================================================================
"""
    
    return summary
