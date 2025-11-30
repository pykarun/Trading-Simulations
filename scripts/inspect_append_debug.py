import runpy
import pandas as pd

g = runpy.run_path('scripts\\alpaca_trader_minimal.py')
fetch_history = g['fetch_history']
compute_double_ema = g['compute_double_ema']
EMA_FAST = g.get('EMA_FAST', 9)
EMA_SLOW = g.get('EMA_SLOW', 21)
HISTORY_DAYS_QQQ = g.get('HISTORY_DAYS_QQQ', 365)

qqq = fetch_history('QQQ', days=HISTORY_DAYS_QQQ)
print('Original columns:', qqq.columns)
print('Original tail:')
print(qqq[['Close']].tail().to_string())

# Simulate appending as main script does
live_price = float(g.get('get_latest_trade_from_alpaca')('QQQ')) if g.get('get_latest_trade_from_alpaca') else float(qqq['Close'].iloc[-1])
new_idx = pd.Timestamp.now()
last_row = qqq.iloc[-1].copy()
last_row.name = new_idx
last_row['Close'] = live_price
qqq2 = pd.concat([qqq, last_row.to_frame().T])
print('\nAfter append columns:', qqq2.columns)
print('After append tail:')
print(qqq2[['Close']].tail().to_string())

try:
    qqq3 = compute_double_ema(qqq2, EMA_FAST, EMA_SLOW)
    print('\nAfter recompute tail:')
    print(qqq3[['Close','EMA_Fast','EMA_Slow']].tail().to_string())
except Exception as e:
    print('\ncompute_double_ema error:', e)
    print('dtypes:', qqq2.dtypes)
    print('columns:', qqq2.columns)
