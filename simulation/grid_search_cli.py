"""Command-line Grid Search for TQQQ Trading Strategy

Run comprehensive grid search without browser timeout limitations.
Results are saved to CSV files for analysis.

Usage:
    python grid_search_cli.py
"""

import pandas as pd
import numpy as np
import datetime
import itertools
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core import get_data, run_tqqq_only_strategy


def generate_param_combinations():
    """Generate all parameter combinations for comprehensive grid search."""
    
    # EMA strategies
    ema_strategy_params = []
    
    # Single EMA
    single_ema_periods = [10, 20, 21, 30, 40, 50, 60, 80, 100]
    for ema in single_ema_periods:
        ema_strategy_params.append({
            'use_double_ema': False,
            'ema_period': ema,
            'ema_fast': 9,
            'ema_slow': 21
        })
    
    # Double EMA Crossover
    fast_ema_range = [5, 8, 9, 10, 12, 15, 20, 21]
    slow_ema_range = [15, 20, 21, 25, 30, 40, 50]
    for fast in fast_ema_range:
        for slow in slow_ema_range:
            if fast < slow:
                ema_strategy_params.append({
                    'use_double_ema': True,
                    'ema_period': slow,
                    'ema_fast': fast,
                    'ema_slow': slow
                })
    
    # RSI parameters
    rsi_range = [0, 40, 45, 50, 55, 60]
    rsi_oversold_range = [20, 25, 30, 35, 40]
    rsi_overbought_range = [60, 65, 70, 75, 80]
    rsi_params = list(itertools.product(rsi_range, rsi_oversold_range, rsi_overbought_range))
    
    # Stop-Loss
    stop_loss_range = [0, 5, 8, 10, 12, 15, 20]
    
    # Bollinger Bands
    bb_params = list(itertools.product(
        ["Enabled", "Disabled"],
        [10, 15, 20, 25, 30],  # bb_period
        [1.5, 2.0, 2.5],  # bb_std_dev
        [0.0, 0.1, 0.2, 0.3],  # bb_buy_threshold
        [0.7, 0.8, 0.9, 1.0]  # bb_sell_threshold
    ))
    
    # ATR Stop-Loss
    atr_params = list(itertools.product(
        ["Enabled", "Disabled"],
        [7, 10, 14, 20, 30],  # atr_period
        [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]  # atr_multiplier
    ))
    
    # MSL/MSH Stop-Loss
    msl_params = list(itertools.product(
        ["Enabled", "Disabled"],
        [10, 15, 20, 25, 30],  # msl_period
        [3, 5, 7, 10, 15]  # msl_lookback
    ))
    
    # MACD
    macd_params = list(itertools.product(
        ["Enabled", "Disabled"],
        [9, 10, 12, 15],  # macd_fast
        [20, 21, 25, 26],  # macd_slow
        [7, 8, 9, 10]  # macd_signal
    ))
    
    # ADX
    adx_params = list(itertools.product(
        ["Enabled", "Disabled"],
        [10, 12, 14, 20],  # adx_period
        [20, 25, 30]  # adx_threshold
    ))
    
    # Supertrend
    st_params = list(itertools.product(
        ["Enabled", "Disabled"],
        [7, 10, 14, 21],  # st_period
        [1.5, 2.0, 2.5, 3.0]  # st_multiplier
    ))
    
    # Combine all parameter sets
    all_combinations = itertools.product(
        ema_strategy_params, rsi_params, stop_loss_range, bb_params,
        atr_params, msl_params, macd_params, adx_params, st_params
    )
    
    param_combinations = []
    for combo in all_combinations:
        ema_p, rsi_p, sl_p, bb_p, atr_p, msl_p, macd_p, adx_p, st_p = combo
        
        use_bb = bb_p[0] == "Enabled"
        use_atr = atr_p[0] == "Enabled"
        use_msl = msl_p[0] == "Enabled"
        use_macd = macd_p[0] == "Enabled"
        use_adx = adx_p[0] == "Enabled"
        use_st = st_p[0] == "Enabled"
        
        param_dict = {
            'use_ema': True,
            'use_double_ema': ema_p['use_double_ema'],
            'ema_period': ema_p['ema_period'],
            'ema_fast': ema_p['ema_fast'],
            'ema_slow': ema_p['ema_slow'],
            'rsi_threshold': rsi_p[0],
            'use_rsi': rsi_p[0] > 0,
            'rsi_oversold': rsi_p[1],
            'rsi_overbought': rsi_p[2],
            'stop_loss_pct': sl_p,
            'use_stop_loss': sl_p > 0,
            'use_bb': use_bb,
            'bb_period': bb_p[1],
            'bb_std_dev': bb_p[2],
            'bb_buy_threshold': bb_p[3],
            'bb_sell_threshold': bb_p[4],
            'use_atr': use_atr,
            'atr_period': atr_p[1],
            'atr_multiplier': atr_p[2],
            'use_msl_msh': use_msl,
            'msl_period': msl_p[1],
            'msh_period': msl_p[1],
            'msl_lookback': msl_p[2],
            'msh_lookback': msl_p[2],
            'use_macd': use_macd,
            'macd_fast': macd_p[1],
            'macd_slow': macd_p[2],
            'macd_signal_period': macd_p[3],
            'use_adx': use_adx,
            'adx_period': adx_p[1],
            'adx_threshold': adx_p[2],
            'use_supertrend': use_st,
            'st_period': st_p[1],
            'st_multiplier': st_p[2]
        }
        param_combinations.append(param_dict)
    
    return param_combinations


def build_param_string(params):
    """Build a human-readable parameter string."""
    if params['use_double_ema']:
        param_str = f"EMA({params['ema_fast']}/{params['ema_slow']})"
    else:
        param_str = f"EMA({params['ema_period']})"
    
    if params['use_rsi']:
        param_str += f" | RSI>{params['rsi_threshold']}"
    
    if params['use_stop_loss']:
        param_str += f" | SL:{params['stop_loss_pct']}%"
    
    if params['use_bb']:
        param_str += f" | BB({params['bb_period']},{params['bb_std_dev']},{params['bb_buy_threshold']}/{params['bb_sell_threshold']})"
    
    if params['use_atr']:
        param_str += f" | ATR({params['atr_period']},{params['atr_multiplier']}x)"
    
    if params['use_msl_msh']:
        param_str += f" | MSL({params['msl_period']},{params['msl_lookback']})"
        
    if params.get('use_macd', False):
        param_str += f" | MACD({params['macd_fast']},{params['macd_slow']},{params['macd_signal_period']})"
        
    if params.get('use_adx', False):
        param_str += f" | ADX({params['adx_period']},{params['adx_threshold']})"
        
    if params.get('use_supertrend', False):
        param_str += f" | ST({params['st_period']},{params['st_multiplier']})"
    
    return param_str


def run_grid_search():
    """Execute comprehensive grid search."""
    
    print("=" * 80)
    print("TQQQ Trading Strategy - Comprehensive Grid Search (CLI)")
    print("=" * 80)
    
    # Configuration
    time_periods = {
        "3M": 90,
        "6M": 180,
        "9M": 270,
        "1Y": 365,
        "2Y": 730,
        "3Y": 1095,
        "4Y": 1460,
        "5Y": 1825
    }
    
    initial_capital = 10000
    
    print("\nConfiguration:")
    print(f"  Initial Capital: ${initial_capital:,.2f}")
    print(f"  Time Periods: {', '.join(time_periods.keys())}")
    
    # Generate parameter combinations
    print("\nGenerating parameter combinations...")
    param_combinations = generate_param_combinations()
    total_combinations = len(param_combinations) * len(time_periods)
    
    print(f"  Parameter combinations: {len(param_combinations):,}")
    print(f"  Time periods: {len(time_periods)}")
    print(f"  Total tests: {total_combinations:,}")
    
    # Download data
    print("\nDownloading historical data...")
    tickers = ["QQQ", "TQQQ"]
    max_days = max(time_periods.values())
    
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=max_days)
    
    try:
        raw_data = get_data(tickers, start_date, end_date, buffer_days=500)
        qqq = raw_data["QQQ"].copy()
        tqqq = raw_data["TQQQ"].copy()
        print(f"  Downloaded data from {qqq.index[0].date()} to {qqq.index[-1].date()}")
    except Exception as e:
        print(f"ERROR: Failed to download data: {e}")
        return
    
    # Run grid search
    print("\nRunning grid search...")
    print("  (This may take several hours for comprehensive testing)")
    
    results = []
    test_counter = 0
    
    for period_name, days_back in time_periods.items():
        period_end_date = datetime.date.today()
        period_start_date = period_end_date - datetime.timedelta(days=days_back)
        
        print(f"\n  Testing period: {period_name} ({period_start_date} to {period_end_date})")
        
        # Calculate QQQ benchmark
        qqq_period = qqq.loc[period_start_date:period_end_date]
        if len(qqq_period) == 0:
            print(f"    WARNING: No data available for period {period_name}")
            continue
        
        qqq_start = qqq_period.iloc[0]['Close']
        qqq_end = qqq_period.iloc[-1]['Close']
        qqq_bh_value = (qqq_end / qqq_start) * initial_capital
        qqq_bh_return = ((qqq_bh_value - initial_capital) / initial_capital) * 100
        
        print(f"    QQQ Benchmark Return: {qqq_bh_return:.2f}%")
        
        period_results = []
        
        for idx, params in enumerate(param_combinations):
            test_counter += 1
            
            if (idx + 1) % 100 == 0:
                progress = (test_counter / total_combinations) * 100
                print(f"    Progress: {idx + 1}/{len(param_combinations)} combinations ({progress:.1f}% overall)")
            
            try:
                result = run_tqqq_only_strategy(
                    qqq.copy(), tqqq.copy(),
                    period_start_date, period_end_date,
                    initial_capital,
                    params['ema_period'],
                    params['rsi_threshold'],
                    params['use_rsi'],
                    params['rsi_oversold'],
                    params['rsi_overbought'],
                    params['stop_loss_pct'],
                    params['use_stop_loss'],
                    params['use_double_ema'],
                    params['ema_fast'],
                    params['ema_slow'],
                    params['use_bb'],
                    params['bb_period'],
                    params['bb_std_dev'],
                    params['bb_buy_threshold'],
                    params['bb_sell_threshold'],
                    params['use_atr'],
                    params['atr_period'],
                    params['atr_multiplier'],
                    params['use_msl_msh'],
                    params['msl_period'],
                    params['msh_period'],
                    params['msl_lookback'],
                    params['msh_lookback'],
                    params.get('use_ema', True),
                    params['use_macd'],
                    params['macd_fast'],
                    params['macd_slow'],
                    params['macd_signal_period'],
                    params['use_adx'],
                    params['adx_period'],
                    params['adx_threshold'],
                    params.get('use_supertrend', False),
                    params.get('st_period', 10),
                    params.get('st_multiplier', 3.0)
                )
                
                # Calculate metrics
                days = (period_end_date - period_start_date).days
                years = days / 365.25
                cagr = ((result['final_value'] / initial_capital) ** (1/years) - 1) * 100 if years > 0 else 0
                
                # Calculate Sharpe Ratio
                portfolio_df = result['portfolio_df'].copy()
                portfolio_df['Daily_Return'] = portfolio_df['Value'].pct_change()
                daily_returns = portfolio_df['Daily_Return'].dropna()
                
                if len(daily_returns) > 0 and daily_returns.std() > 0:
                    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
                else:
                    sharpe = 0
                
                outperformance = result['total_return_pct'] - qqq_bh_return
                
                param_str = build_param_string(params)
                
                period_results.append({
                    'Period': period_name,
                    'Parameters': param_str,
                    'Final Value': result['final_value'],
                    'Total Return %': result['total_return_pct'],
                    'CAGR %': cagr,
                    'Max Drawdown %': result['max_drawdown'],
                    'Sharpe Ratio': sharpe,
                    'Trades': result['num_trades'],
                    'vs QQQ %': outperformance,
                    'QQQ Return %': qqq_bh_return
                })
                
            except Exception as e:
                print(f"    ERROR in test {test_counter}: {str(e)}")
        
        results.extend(period_results)
        
        # Save period results
        if period_results:
            period_df = pd.DataFrame(period_results)
            period_df = period_df.sort_values('vs QQQ %', ascending=False)
            output_file = f"grid_search_results_{period_name}.csv"
            period_df.to_csv(output_file, index=False)
            print(f"    Saved results to: {output_file}")
            print(f"    Best result: {period_df.iloc[0]['Parameters']}")
            print(f"    Best return: {period_df.iloc[0]['Total Return %']:.2f}% (vs QQQ: {period_df.iloc[0]['vs QQQ %']:.2f}%)")
    
    # Save combined results
    if results:
        print("\n" + "=" * 80)
        print("Grid Search Complete!")
        print("=" * 80)
        
        all_results_df = pd.DataFrame(results)
        all_results_df = all_results_df.sort_values('vs QQQ %', ascending=False)
        
        output_file = "grid_search_results_all.csv"
        all_results_df.to_csv(output_file, index=False)
        
        print(f"\nTotal tests completed: {len(results):,}")
        print(f"Results saved to: {output_file}")
        print("\nTop 5 Overall Strategies:")
        print(all_results_df[['Period', 'Parameters', 'Total Return %', 'vs QQQ %', 'Max Drawdown %', 'Sharpe Ratio']].head(5).to_string(index=False))
        
        print("\nIndividual period results saved to:")
        for period_name in time_periods.keys():
            print(f"  - grid_search_results_{period_name}.csv")
    else:
        print("\nNo results generated. Please check for errors.")


if __name__ == "__main__":
    run_grid_search()
