# TradingSimulations

A comprehensive trading simulation and automated trading platform for leveraged ETFs, specifically designed for trading TQQQ (3x leveraged QQQ ETF) based on QQQ technical indicators.

## Overview

This project provides two main components:

1. **Strategy Simulation UI** - A Streamlit-based web application for backtesting and optimizing trading strategies
2. **Alpaca Trader** - An automated trading script that executes strategies on the Alpaca paper/live trading platform

## Features

### Technical Indicators
- **EMA (Exponential Moving Average)** - Single or Double EMA crossover strategies
- **RSI (Relative Strength Index)** - Momentum-based filtering
- **Bollinger Bands** - Volatility-based entry/exit signals
- **MACD (Moving Average Convergence Divergence)** - Trend and momentum analysis
- **ADX (Average Directional Index)** - Trend strength measurement
- **Supertrend** - Trend-following indicator
- **ATR (Average True Range)** - Volatility measurement for stop-loss

### Stop-Loss Methods
- **Percentage-based trailing stop** - Trail stop based on peak price
- **ATR-based stop** - Dynamic stop based on market volatility
- **MSL/MSH (Multi-timeframe Stop Low/High)** - Support/resistance based stops

### Strategy Wizard (Streamlit UI)
A 5-step wizard interface for creating and testing trading strategies:
1. **Introduction** - Overview and workflow
2. **Find Best Signals** - Grid search algorithm to find optimal parameters
3. **Verify & Customize** - Review and adjust strategy parameters
4. **Testing** - Daily signals, backtests, Monte Carlo simulations
5. **AI Summary** - Generate comprehensive AI-friendly reports

## Installation

### Prerequisites
- Python 3.11+
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone https://github.com/pykarun/TradingSimulations.git
cd TradingSimulations
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

The main dependencies include:
- `streamlit` - Web UI framework
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `yfinance` - Yahoo Finance data fetching
- `plotly` - Interactive charts
- `requests` - HTTP client for Alpaca API
- `curl_cffi` - Enhanced HTTP client

## Usage

### Running the Strategy Simulation UI

Start the Streamlit application:
```bash
streamlit run simulation/start.py
```

The application will be available at `http://localhost:8501`

### Using GitHub Codespaces

This project includes a devcontainer configuration for GitHub Codespaces. Simply open the repository in a Codespace and the Streamlit application will start automatically on port 8501.

### Running the Alpaca Trader

#### Setup Alpaca Credentials

Create a file `scripts/alpaca_secrets.py` with your Alpaca API credentials:
```python
ALPACA_API_KEY_ID = "YOUR_ALPACA_API_KEY_ID"
ALPACA_SECRET_KEY = "YOUR_ALPACA_SECRET_KEY"
ALPACA_PAPER_TRADING = True
ALPACA_BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_DATA_URL = "https://data.alpaca.markets"
```

#### Running the Full Trader

```bash
python scripts/alpaca_trader.py
```

Command-line options:
- `--dry-run` - Run without submitting actual orders
- `--validate-only` - Validate Alpaca credentials and exit
- `--once` - Run a single trading loop iteration and exit

#### Running the Minimal Trader

A simplified trader with double-EMA and trailing stop:
```bash
python scripts/alpaca_trader_minimal.py
```

Environment variables can be used instead of the secrets file:
- `ALPACA_API_KEY`
- `ALPACA_API_SECRET`
- `ALPACA_BASE_URL`
- `ALPACA_DATA_URL`

## Strategy Configuration

The trading strategy can be configured via `simulation/config/strategy_config.json`:

```json
{
    "use_ema": true,
    "use_double_ema": true,
    "ema_period": 50,
    "ema_fast": 2,
    "ema_slow": 21,
    "use_rsi": false,
    "use_bb": false,
    "use_atr": false,
    "use_msl_msh": false,
    "use_macd": false,
    "use_adx": false,
    "use_supertrend": false,
    "use_stop_loss": true,
    "stop_loss_pct": 15
}
```

### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `use_ema` | Enable EMA-based signals | `true` |
| `use_double_ema` | Use double EMA crossover | `true` |
| `ema_fast` | Fast EMA period | `2` |
| `ema_slow` | Slow EMA period | `21` |
| `use_stop_loss` | Enable trailing stop-loss | `true` |
| `stop_loss_pct` | Stop-loss percentage | `15` |

## Project Structure

```
TradingSimulations/
├── simulation/              # Main simulation package
│   ├── start.py            # Streamlit app entry point
│   ├── config/             # Configuration modules
│   │   ├── strategy_config.json  # Strategy parameters
│   │   ├── page_config.py        # Streamlit page settings
│   │   └── session_state.py      # Session state management
│   ├── core/               # Core trading logic
│   │   ├── data.py         # Data fetching utilities
│   │   ├── indicators.py   # Technical indicator calculations
│   │   ├── strategy.py     # Strategy execution engine
│   │   └── stoploss.py     # Stop-loss calculations
│   ├── ui/                 # UI components
│   │   ├── step1_intro.py         # Introduction wizard step
│   │   ├── step2_grid_search.py   # Parameter optimization
│   │   ├── step3_verify.py        # Strategy verification
│   │   ├── step4_testing.py       # Backtesting & Monte Carlo
│   │   └── step5_ai_summary.py    # AI report generation
│   └── utils/              # Utility modules
│       ├── banner.py       # UI banner components
│       └── charts.py       # Chart generation
├── scripts/                # Trading scripts
│   ├── alpaca_trader.py          # Full Alpaca trading bot
│   ├── alpaca_trader_minimal.py  # Minimal trading bot
│   └── tradingview_strategy.pine # TradingView Pine Script
├── utils/                  # Shared utilities
│   └── logging.py          # Logging configuration
├── requirements.txt        # Python dependencies
└── .devcontainer/          # GitHub Codespaces configuration
```

## TradingView Integration

A TradingView Pine Script is included at `scripts/tradingview_strategy.pine` for visualizing the double-EMA crossover strategy with trailing stop-loss on TradingView charts.

## Logging

Logs are stored in the `logs/` directory with hourly rotation. The logging system provides:
- Timestamped entries with millisecond precision
- Console and file output
- Configurable log levels

## Disclaimer

This software is for educational and research purposes only. Trading leveraged ETFs carries significant risk. Always test strategies thoroughly with paper trading before using real funds. The authors are not responsible for any financial losses incurred through the use of this software.

## License

This project is provided as-is for educational purposes.
