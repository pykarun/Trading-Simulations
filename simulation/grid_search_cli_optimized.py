"""Command-line Grid Search for TQQQ Trading Strategy - Optimized Version

Memory-efficient version using file-based processing to prevent crashes.
Saves intermediate results to CSV files instead of keeping everything in memory.

Usage:
    python grid_search_cli_optimized.py
"""

import pandas as pd
import numpy as np
import datetime
import itertools
import sys
import os
import csv
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from core import get_data, run_tqqq_only_strategy


def generate_param_combinations_file():
    """Generate all parameter combinations and save to CSV file."""
    
    print("Generating parameter combinations...")
    
    # EMA strategies
    single_ema_periods = [10, 20, 21, 30, 40, 50, 60, 80, 100]
    fast_ema_range = [5, 8, 9, 10, 12, 15, 20, 21]
    slow_ema_range = [15, 20, 21, 25, 30, 40, 50]
    
    # RSI parameters
    rsi_range = [0, 40, 45, 50, 55, 60]
    rsi_oversold_range = [20, 25, 30, 35, 40]
    rsi_overbought_range = [60, 65, 70, 75, 80]
    
    # Stop-Loss
    stop_loss_range = [0, 5, 8, 10, 12, 15, 20]
    
    # Bollinger Bands
    bb_enabled = ["Enabled", "Disabled"]
    bb_period_range = [10, 15, 20, 25, 30]
    bb_std_dev_range = [1.5, 2.0, 2.5]
    bb_buy_threshold_range = [0.0, 0.1, 0.2, 0.3]
    bb_sell_threshold_range = [0.7, 0.8, 0.9, 1.0]
    
    # ATR Stop-Loss
    atr_enabled = ["Enabled", "Disabled"]
    atr_period_range = [7, 10, 14, 20, 30]
    atr_multiplier_range = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    
    # MSL/MSH Stop-Loss
    msl_enabled = ["Enabled", "Disabled"]
    msl_period_range = [10, 15, 20, 25, 30]
    msl_lookback_range = [3, 5, 7, 10, 15]
    
    # MACD
    macd_enabled = ["Enabled", "Disabled"]
    macd_fast_range = [9, 10, 12, 15]
    macd_slow_range = [20, 21, 25, 26]
    macd_signal_range = [7, 8, 9, 10]
    
    # ADX
    adx_enabled = ["Enabled", "Disabled"]
    adx_period_range = [10, 12, 14, 20]
    adx_threshold_range = [20, 25, 30]
    
    # Supertrend
    st_enabled = ["Enabled", "Disabled"]
    st_period_range = [7, 10, 14, 21]
    st_multiplier_range = [1.5, 2.0, 2.5, 3.0]
    
    # Pivot Points
    pivot_enabled = ["Enabled", "Disabled"]
    pivot_left_range = [3, 5, 7, 10]
    pivot_right_range = [3, 5, 7, 10]
    
    # Create temp file for parameter combinations
    temp_file = Path(__file__).parent / "temp_param_combinations.csv"
    
    fieldnames = [
        'use_ema', 'use_double_ema', 'ema_period', 'ema_fast', 'ema_slow',
        'rsi_threshold', 'use_rsi', 'rsi_oversold', 'rsi_overbought',
        'stop_loss_pct', 'use_stop_loss',
        'use_bb', 'bb_period', 'bb_std_dev', 'bb_buy_threshold', 'bb_sell_threshold',
        'use_atr', 'atr_period', 'atr_multiplier',
        'use_msl_msh', 'msl_period', 'msh_period', 'msl_lookback', 'msh_lookback',
        'use_macd', 'macd_fast', 'macd_slow', 'macd_signal_period',
        'use_adx', 'adx_period', 'adx_threshold',
        'use_supertrend', 'st_period', 'st_multiplier',
        'use_pivot', 'pivot_left', 'pivot_right'
    ]
    
    combo_count = 0
    
    with open(temp_file, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        # Single EMA combinations
        for ema in single_ema_periods:
            for rsi_t, rsi_os, rsi_ob in itertools.product(rsi_range, rsi_oversold_range, rsi_overbought_range):
                for sl in stop_loss_range:
                    for bb_en, bb_p, bb_std, bb_buy, bb_sell in itertools.product(bb_enabled, bb_period_range, bb_std_dev_range, bb_buy_threshold_range, bb_sell_threshold_range):
                        for atr_en, atr_p, atr_m in itertools.product(atr_enabled, atr_period_range, atr_multiplier_range):
                            for msl_en, msl_p, msl_lb in itertools.product(msl_enabled, msl_period_range, msl_lookback_range):
                                for macd_en, macd_f, macd_s, macd_sig in itertools.product(macd_enabled, macd_fast_range, macd_slow_range, macd_signal_range):
                                    for adx_en, adx_p, adx_t in itertools.product(adx_enabled, adx_period_range, adx_threshold_range):
                                        for st_en, st_p, st_m in itertools.product(st_enabled, st_period_range, st_multiplier_range):
                                            for pivot_en, pivot_l, pivot_r in itertools.product(pivot_enabled, pivot_left_range, pivot_right_range):
                                                writer.writerow({
                                                    'use_ema': True, 'use_double_ema': False,
                                                    'ema_period': ema, 'ema_fast': 9, 'ema_slow': 21,
                                                    'rsi_threshold': rsi_t, 'use_rsi': rsi_t > 0,
                                                    'rsi_oversold': rsi_os, 'rsi_overbought': rsi_ob,
                                                    'stop_loss_pct': sl, 'use_stop_loss': sl > 0,
                                                    'use_bb': bb_en == "Enabled", 'bb_period': bb_p,
                                                    'bb_std_dev': bb_std, 'bb_buy_threshold': bb_buy,
                                                    'bb_sell_threshold': bb_sell,
                                                    'use_atr': atr_en == "Enabled", 'atr_period': atr_p,
                                                    'atr_multiplier': atr_m,
                                                    'use_msl_msh': msl_en == "Enabled", 'msl_period': msl_p,
                                                    'msh_period': msl_p, 'msl_lookback': msl_lb,
                                                    'msh_lookback': msl_lb,
                                                    'use_macd': macd_en == "Enabled", 'macd_fast': macd_f,
                                                    'macd_slow': macd_s, 'macd_signal_period': macd_sig,
                                                    'use_adx': adx_en == "Enabled", 'adx_period': adx_p,
                                                    'adx_threshold': adx_t,
                                                    'use_supertrend': st_en == "Enabled", 'st_period': st_p,
                                                    'st_multiplier': st_m,
                                                    'use_pivot': pivot_en == "Enabled", 'pivot_left': pivot_l,
                                                    'pivot_right': pivot_r
                                                })
                                                combo_count += 1
        
        # Double EMA crossover combinations
        for fast, slow in itertools.product(fast_ema_range, slow_ema_range):
            if fast >= slow:
                continue
                
            for rsi_t, rsi_os, rsi_ob in itertools.product(rsi_range, rsi_oversold_range, rsi_overbought_range):
                for sl in stop_loss_range:
                    for bb_en, bb_p, bb_std, bb_buy, bb_sell in itertools.product(bb_enabled, bb_period_range, bb_std_dev_range, bb_buy_threshold_range, bb_sell_threshold_range):
                        for atr_en, atr_p, atr_m in itertools.product(atr_enabled, atr_period_range, atr_multiplier_range):
                            for msl_en, msl_p, msl_lb in itertools.product(msl_enabled, msl_period_range, msl_lookback_range):
                                for macd_en, macd_f, macd_s, macd_sig in itertools.product(macd_enabled, macd_fast_range, macd_slow_range, macd_signal_range):
                                    for adx_en, adx_p, adx_t in itertools.product(adx_enabled, adx_period_range, adx_threshold_range):
                                        for st_en, st_p, st_m in itertools.product(st_enabled, st_period_range, st_multiplier_range):
                                            for pivot_en, pivot_l, pivot_r in itertools.product(pivot_enabled, pivot_left_range, pivot_right_range):
                                                writer.writerow({
                                                    'use_ema': True, 'use_double_ema': True,
                                                    'ema_period': slow, 'ema_fast': fast, 'ema_slow': slow,
                                                    'rsi_threshold': rsi_t, 'use_rsi': rsi_t > 0,
                                                    'rsi_oversold': rsi_os, 'rsi_overbought': rsi_ob,
                                                    'stop_loss_pct': sl, 'use_stop_loss': sl > 0,
                                                    'use_bb': bb_en == "Enabled", 'bb_period': bb_p,
                                                    'bb_std_dev': bb_std, 'bb_buy_threshold': bb_buy,
                                                    'bb_sell_threshold': bb_sell,
                                                    'use_atr': atr_en == "Enabled", 'atr_period': atr_p,
                                                    'atr_multiplier': atr_m,
                                                    'use_msl_msh': msl_en == "Enabled", 'msl_period': msl_p,
                                                    'msh_period': msl_p, 'msl_lookback': msl_lb,
                                                    'msh_lookback': msl_lb,
                                                    'use_macd': macd_en == "Enabled", 'macd_fast': macd_f,
                                                    'macd_slow': macd_s, 'macd_signal_period': macd_sig,
                                                    'use_adx': adx_en == "Enabled", 'adx_period': adx_p,
                                                    'adx_threshold': adx_t,
                                                    'use_supertrend': st_en == "Enabled", 'st_period': st_p,
                                                    'st_multiplier': st_m,
                                                    'use_pivot': pivot_en == "Enabled", 'pivot_left': pivot_l,
                                                    'pivot_right': pivot_r
                                                })
                                                combo_count += 1
    
    print(f"  Generated {combo_count:,} parameter combinations")
    print(f"  Saved to: {temp_file}")
    
    return temp_file, combo_count


def build_param_string(params):
    """Build a human-readable parameter string."""
    param_str = ""
    
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
        
    if params['use_macd']:
        param_str += f" | MACD({params['macd_fast']},{params['macd_slow']},{params['macd_signal_period']})"
        
    if params['use_adx']:
        param_str += f" | ADX({params['adx_period']},{params['adx_threshold']})"
        
    if params['use_supertrend']:
        param_str += f" | ST({params['st_period']},{params['st_multiplier']})"
    
    if params['use_pivot']:
        param_str += f" | Pivot({params['pivot_left']},{params['pivot_right']})"
    
    return param_str


def process_period_file_based(period_name, days_back, qqq, tqqq, initial_capital, param_file):
    """Process one time period using file-based approach."""
    
    print(f"\n  Processing period: {period_name}")
    
    period_end_date = datetime.date.today()
    period_start_date = period_end_date - datetime.timedelta(days=days_back)
    
    # Calculate QQQ benchmark
    qqq_period = qqq.loc[period_start_date:period_end_date]
    if len(qqq_period) == 0:
        print(f"    WARNING: No data available for period {period_name}")
        return None
    
    qqq_start = qqq_period.iloc[0]['Close']
    qqq_end = qqq_period.iloc[-1]['Close']
    qqq_bh_value = (qqq_end / qqq_start) * initial_capital
    qqq_bh_return = ((qqq_bh_value - initial_capital) / initial_capital) * 100
    
    print(f"    QQQ Benchmark Return: {qqq_bh_return:.2f}%")
    
    # Create intermediate results file for this period
    temp_results_file = Path(__file__).parent / f"temp_results_{period_name}.csv"
    
    result_fieldnames = [
        'Period', 'Parameters', 'Final Value', 'Total Return %', 'CAGR %',
        'Max Drawdown %', 'Sharpe Ratio', 'Trades', 'vs QQQ %', 'QQQ Return %'
    ]
    
    # Process parameter combinations in batches
    batch_size = 100
    test_counter = 0
    
    with open(param_file, 'r') as pf:
        reader = csv.DictReader(pf)
        
        with open(temp_results_file, 'w', newline='') as rf:
            writer = csv.DictWriter(rf, fieldnames=result_fieldnames)
            writer.writeheader()
            
            batch = []
            for row in reader:
                # Convert string values to appropriate types
                params = {}
                for key, value in row.items():
                    if key.startswith('use_'):
                        params[key] = value == 'True'
                    elif key in ['ema_period', 'ema_fast', 'ema_slow', 'rsi_threshold', 
                                 'rsi_oversold', 'rsi_overbought', 'stop_loss_pct',
                                 'bb_period', 'atr_period', 'msl_period', 'msh_period',
                                 'msl_lookback', 'msh_lookback', 'macd_fast', 'macd_slow',
                                 'macd_signal_period', 'adx_period', 'adx_threshold',
                                 'st_period', 'pivot_left', 'pivot_right']:
                        params[key] = int(value)
                    elif key in ['bb_std_dev', 'bb_buy_threshold', 'bb_sell_threshold',
                                 'atr_multiplier', 'st_multiplier']:
                        params[key] = float(value)
                    else:
                        params[key] = value
                
                batch.append(params)
                
                if len(batch) >= batch_size:
                    # Process batch
                    for params in batch:
                        test_counter += 1
                        
                        if test_counter % 500 == 0:
                            print(f"    Processed {test_counter:,} combinations...")
                        
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
                                params['use_ema'],
                                params['use_macd'],
                                params['macd_fast'],
                                params['macd_slow'],
                                params['macd_signal_period'],
                                params['use_adx'],
                                params['adx_period'],
                                params['adx_threshold'],
                                params['use_supertrend'],
                                params['st_period'],
                                params['st_multiplier'],
                                params['use_pivot'],
                                params['pivot_left'],
                                params['pivot_right']
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
                            
                            writer.writerow({
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
                    
                    batch = []
            
            # Process remaining batch
            for params in batch:
                test_counter += 1
                
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
                        params['use_ema'],
                        params['use_macd'],
                        params['macd_fast'],
                        params['macd_slow'],
                        params['macd_signal_period'],
                        params['use_adx'],
                        params['adx_period'],
                        params['adx_threshold'],
                        params['use_supertrend'],
                        params['st_period'],
                        params['st_multiplier'],
                        params['use_pivot'],
                        params['pivot_left'],
                        params['pivot_right']
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
                    
                    writer.writerow({
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
    
    print(f"    Completed {test_counter:,} tests")
    print(f"    Intermediate results saved to: {temp_results_file}")
    
    # Sort and save final results for this period
    df = pd.read_csv(temp_results_file)
    df_sorted = df.sort_values('vs QQQ %', ascending=False)
    
    output_file = f"grid_search_results_{period_name}.csv"
    df_sorted.to_csv(output_file, index=False)
    
    print(f"    Final sorted results saved to: {output_file}")
    
    if len(df_sorted) > 0:
        print(f"    Best result: {df_sorted.iloc[0]['Parameters']}")
        print(f"    Best return: {df_sorted.iloc[0]['Total Return %']:.2f}% (vs QQQ: {df_sorted.iloc[0]['vs QQQ %']:.2f}%)")
    
    return temp_results_file


def run_grid_search():
    """Execute comprehensive grid search with file-based processing."""
    
    print("=" * 80)
    print("TQQQ Trading Strategy - Comprehensive Grid Search (File-Based)")
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
    print(f"  Processing: File-based (memory efficient)")
    
    # Generate parameter combinations file
    print("\n" + "=" * 80)
    param_file, total_combos = generate_param_combinations_file()
    print(f"  Total tests to run: {total_combos * len(time_periods):,}")
    print("=" * 80)
    
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
    
    # Process each time period
    print("\nRunning grid search...")
    print("  (Processing each period separately to manage memory)")
    
    temp_result_files = []
    
    for period_name, days_back in time_periods.items():
        try:
            temp_file = process_period_file_based(
                period_name, days_back, qqq, tqqq, initial_capital, param_file
            )
            if temp_file:
                temp_result_files.append(temp_file)
        except Exception as e:
            print(f"  ERROR processing period {period_name}: {e}")
    
    # Combine all results
    print("\n" + "=" * 80)
    print("Combining all results...")
    print("=" * 80)
    
    all_results = []
    for temp_file in temp_result_files:
        if temp_file and os.path.exists(temp_file):
            df = pd.read_csv(temp_file)
            all_results.append(df)
    
    if all_results:
        combined_df = pd.concat(all_results, ignore_index=True)
        combined_df_sorted = combined_df.sort_values('vs QQQ %', ascending=False)
        
        output_file = "grid_search_results_all.csv"
        combined_df_sorted.to_csv(output_file, index=False)
        
        print(f"\nTotal tests completed: {len(combined_df):,}")
        print(f"All results saved to: {output_file}")
        print("\nTop 5 Overall Strategies:")
        print(combined_df_sorted[['Period', 'Parameters', 'Total Return %', 'vs QQQ %', 'Max Drawdown %', 'Sharpe Ratio']].head(5).to_string(index=False))
        
        print("\nIndividual period results saved to:")
        for period_name in time_periods.keys():
            print(f"  - grid_search_results_{period_name}.csv")
        
        # Cleanup temp files
        print("\nCleaning up temporary files...")
        if os.path.exists(param_file):
            os.remove(param_file)
            print(f"  Removed: {param_file}")
        
        for temp_file in temp_result_files:
            if temp_file and os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"  Removed: {temp_file}")
        
        print("\n" + "=" * 80)
        print("Grid Search Complete!")
        print("=" * 80)
    else:
        print("\nNo results generated. Please check for errors.")


if __name__ == "__main__":
    run_grid_search()
