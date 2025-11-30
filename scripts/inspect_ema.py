import runpy
import pandas as pd

# Load functions from your minimal trader script
g = runpy.run_path('scripts\\alpaca_trader_minimal.py')
fetch_history = g['fetch_history']
compute_double_ema = g['compute_double_ema']
EMA_FAST = g.get('EMA_FAST', 9)
EMA_SLOW = g.get('EMA_SLOW', 21)
HISTORY_DAYS_QQQ = g.get('HISTORY_DAYS_QQQ', 365)

print(f'Using spans: EMA_FAST={EMA_FAST} EMA_SLOW={EMA_SLOW}\n')

# Fetch history and compute EMAs
qqq = fetch_history('QQQ', days=HISTORY_DAYS_QQQ)
qqq = compute_double_ema(qqq, EMA_FAST, EMA_SLOW)

print(qqq[['Close','EMA_Fast','EMA_Slow']].tail(8).to_string())

# Now simulate appending a live price (same as last close) and recompute EMAs
try:
	latest_price = g.get('latest_price')
	if latest_price is None:
		last_close = float(qqq['Close'].iat[-1])
	else:
		last_close = float(latest_price(qqq))
	new_idx = pd.Timestamp.now()
	new_row = qqq.tail(1).copy()
	new_row.index = [new_idx]
	new_row['Close'] = last_close
	qqq2 = pd.concat([qqq, new_row])
	qqq2 = compute_double_ema(qqq2, EMA_FAST, EMA_SLOW)
	print('\nAfter appending a live row (same as last close):')
	print(qqq2[['Close','EMA_Fast','EMA_Slow']].tail(6).to_string())
except Exception as e:
	print('\nCould not simulate live append:', e)
