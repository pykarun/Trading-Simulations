"""Chart creation utilities."""
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from core.indicators import calculate_ema, calculate_pivot_points, calculate_supertrend


def create_performance_chart(result, qqq_data, tqqq_data, start_date, initial_capital, ema_period, params=None):
    """Create performance chart with QQQ benchmark.
    
    Args:
        result: Strategy backtest results dictionary
        qqq_data: QQQ historical data
        tqqq_data: TQQQ historical data
        start_date: Backtest start date
        initial_capital: Initial capital amount
        ema_period: EMA period for chart
        params: Strategy parameters dict (optional, for showing pivot points and other indicators)
        
    Returns:
        Plotly figure object
    """
    
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.5, 0.25, 0.25],
        subplot_titles=(
            'Portfolio Value vs QQQ Benchmark',
            f'QQQ Price vs {ema_period}-day EMA',
            'Drawdown %'
        )
    )
    
    # Portfolio value
    fig.add_trace(
        go.Scatter(
            x=result['portfolio_df'].index,
            y=result['portfolio_df']['Value'],
            name='Smart Leverage Strategy',
            line=dict(color='green', width=3)
        ),
        row=1, col=1
    )
    
    # Trade markers
    trade_log_df = pd.DataFrame(result['trade_log'])
    trade_log_df['Date'] = pd.to_datetime(trade_log_df['Date'])
    trade_log_df = trade_log_df.set_index('Date')
    trade_log_df['Portfolio_Value_Numeric'] = trade_log_df['Portfolio_Value'].str.replace('$', '').str.replace(',', '').astype(float)
    
    buy_trades = trade_log_df[trade_log_df['Action'].str.contains('BUY TQQQ', case=False, na=False)]
    if len(buy_trades) > 0:
        fig.add_trace(
            go.Scatter(
                x=buy_trades.index,
                y=buy_trades['Portfolio_Value_Numeric'],
                mode='markers',
                name='Buy TQQQ',
                marker=dict(symbol='triangle-up', size=12, color='lime', line=dict(color='darkgreen', width=1)),
                hovertemplate='<b>BUY TQQQ</b><br>Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    sell_trades = trade_log_df[trade_log_df['Action'].str.contains('SELL', case=False, na=False)]
    if len(sell_trades) > 0:
        fig.add_trace(
            go.Scatter(
                x=sell_trades.index,
                y=sell_trades['Portfolio_Value_Numeric'],
                mode='markers',
                name='Sell to Cash',
                marker=dict(symbol='triangle-down', size=12, color='yellow', line=dict(color='orange', width=1)),
                hovertemplate='<b>SELL to CASH</b><br>Date: %{x}<br>Value: $%{y:,.2f}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # QQQ benchmark
    qqq_benchmark = qqq_data.loc[start_date:]['Close']
    qqq_performance = (qqq_benchmark / qqq_benchmark.iloc[0]) * initial_capital
    fig.add_trace(
        go.Scatter(
            x=qqq_performance.index,
            y=qqq_performance,
            name='QQQ Buy & Hold',
            line=dict(color='gray', dash='dot', width=2)
        ),
        row=1, col=1
    )
    
    # QQQ Price vs EMA and indicators
    qqq_with_ema = calculate_ema(qqq_data, ema_period)
    
    # Add pivot points if enabled
    if params and params.get('use_pivot', False):
        qqq_with_ema = calculate_pivot_points(
            qqq_with_ema, 
            pivot_left=params.get('pivot_left', 5), 
            pivot_right=params.get('pivot_right', 5)
        )
    
    # Add supertrend if enabled
    if params and params.get('use_supertrend', False):
        qqq_with_ema = calculate_supertrend(
            qqq_with_ema,
            period=params.get('st_period', 10),
            multiplier=params.get('st_multiplier', 3.0)
        )
    
    qqq_display = qqq_with_ema.loc[start_date:]
    
    fig.add_trace(
        go.Scatter(
            x=qqq_display.index,
            y=qqq_display['Close'],
            name='QQQ Price',
            line=dict(color='black', width=2)
        ),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Scatter(
            x=qqq_display.index,
            y=qqq_display['EMA'],
            name=f'{ema_period}-day EMA',
            line=dict(color='red', width=2, dash='dash')
        ),
        row=2, col=1
    )
    
    # Plot pivot points if available
    if params and params.get('use_pivot', False) and 'Pivot_High' in qqq_display.columns:
        pivot_highs = qqq_display[qqq_display['Pivot_High'].notna()]
        if len(pivot_highs) > 0:
            # Add markers for pivot highs
            fig.add_trace(
                go.Scatter(
                    x=pivot_highs.index,
                    y=pivot_highs['Pivot_High'],
                    mode='markers',
                    name=f'🔴 Pivot High Resistance (L:{params["pivot_left"]}/R:{params["pivot_right"]})',
                    marker=dict(symbol='triangle-down', size=14, color='red', line=dict(color='darkred', width=2)),
                    hovertemplate='<b>🔴 PIVOT HIGH (Resistance)</b><br>Date: %{x}<br>Price: $%{y:.2f}<br>Left Bars: ' + str(params["pivot_left"]) + '<br>Right Bars: ' + str(params["pivot_right"]) + '<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Add horizontal lines extending from pivot highs
            for idx, row_data in pivot_highs.iterrows():
                fig.add_hline(
                    y=row_data['Pivot_High'],
                    line_dash="dash",
                    line_color="rgba(255, 0, 0, 0.3)",
                    line_width=1,
                    row=2, col=1
                )
        
        pivot_lows = qqq_display[qqq_display['Pivot_Low'].notna()]
        if len(pivot_lows) > 0:
            # Add markers for pivot lows
            fig.add_trace(
                go.Scatter(
                    x=pivot_lows.index,
                    y=pivot_lows['Pivot_Low'],
                    mode='markers',
                    name=f'🟢 Pivot Low Support (L:{params["pivot_left"]}/R:{params["pivot_right"]})',
                    marker=dict(symbol='triangle-up', size=14, color='lime', line=dict(color='darkgreen', width=2)),
                    hovertemplate='<b>🟢 PIVOT LOW (Support)</b><br>Date: %{x}<br>Price: $%{y:.2f}<br>Left Bars: ' + str(params["pivot_left"]) + '<br>Right Bars: ' + str(params["pivot_right"]) + '<extra></extra>'
                ),
                row=2, col=1
            )
            
            # Add horizontal lines extending from pivot lows
            for idx, row_data in pivot_lows.iterrows():
                fig.add_hline(
                    y=row_data['Pivot_Low'],
                    line_dash="dash",
                    line_color="rgba(0, 255, 0, 0.3)",
                    line_width=1,
                    row=2, col=1
                )
    
    # Plot supertrend if available
    if params and params.get('use_supertrend', False) and 'ST' in qqq_display.columns:
        fig.add_trace(
            go.Scatter(
                x=qqq_display.index,
                y=qqq_display['ST'],
                name=f'Supertrend ({params["st_period"]},{params["st_multiplier"]})',
                line=dict(color='purple', width=2, dash='dot')
            ),
            row=2, col=1
        )
    
    # Drawdown
    fig.add_trace(
        go.Scatter(
            x=result['portfolio_df'].index,
            y=result['portfolio_df']['Drawdown'],
            name='Drawdown',
            fill='tozeroy',
            line=dict(color='red', width=1)
        ),
        row=3, col=1
    )
    
    fig.update_layout(
        height=800,
        title_text="Smart Leverage Strategy - Performance Analysis",
        showlegend=True,
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)
    fig.update_yaxes(title_text="Price ($)", row=2, col=1)
    fig.update_yaxes(title_text="Drawdown %", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1)
    
    return fig
