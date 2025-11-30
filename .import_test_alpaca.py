import types, sys
sys.modules['alpaca_secrets'] = types.SimpleNamespace(ALPACA_API_KEY_ID='x', ALPACA_SECRET_KEY='y', ALPACA_PAPER_TRADING=True, ALPACA_BASE_URL='https://paper-api.alpaca.markets', ALPACA_DATA_URL='https://data.alpaca.markets')
import importlib
m = importlib.import_module('scripts.alpaca_trader')
print('module loaded, has run_trading_loop=', hasattr(m,'run_trading_loop'))
