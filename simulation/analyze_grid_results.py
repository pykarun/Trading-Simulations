"""
Grid Search Results Analyzer
Analyze parameter combinations across multiple time periods to find optimal configurations
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import numpy as np

# Page config
st.set_page_config(
    page_title="Grid Search Analyzer",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Grid Search Results Analyzer")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    """Load the grid search results CSV"""
    # Try multiple possible locations
    possible_paths = [
        Path(__file__).parent.parent / "grid_search_results_all.csv",  # Project root
        Path(__file__).parent / "grid_search_results_all.csv",  # simulation folder
        Path("grid_search_results_all.csv"),  # Current directory
        Path("c:/Projects/TradingSimulations/grid_search_results_all.csv")  # Absolute path
    ]
    
    found_path = None
    for csv_path in possible_paths:
        if csv_path.exists():
            found_path = csv_path
            break
    
    if found_path:
        try:
            df = pd.read_csv(found_path)
            return df, str(found_path)
        except Exception as e:
            return None, f"Error reading {found_path}: {e}"
    
    return None, "File not found in any location"

result = load_data()

if result[0] is None:
    st.error(f"❌ Could not find grid_search_results_all.csv. Please run the grid search first.")
    st.info(f"Debug: {result[1]}")
    st.info("Searched locations:")
    for path in [
        Path(__file__).parent.parent / "grid_search_results_all.csv",
        Path(__file__).parent / "grid_search_results_all.csv",
        Path("grid_search_results_all.csv"),
        Path("c:/Projects/TradingSimulations/grid_search_results_all.csv")
    ]:
        st.write(f"- {path} (exists: {path.exists()})")
    st.stop()

df = result[0]
st.success(f"✅ Loaded data from: {result[1]}")

# Display basic stats
st.header("📈 Overview")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Combinations Tested", f"{len(df):,}")
with col2:
    st.metric("Time Periods", df['Period'].nunique())
with col3:
    st.metric("Best Return", f"{df['Total Return %'].max():.2f}%")
with col4:
    st.metric("Best vs QQQ", f"{df['vs QQQ %'].max():.2f}%")

st.markdown("---")

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Period filter
periods = ['All'] + sorted(df['Period'].unique().tolist())
selected_period = st.sidebar.selectbox("Time Period", periods)

if selected_period != 'All':
    df_filtered = df[df['Period'] == selected_period].copy()
else:
    df_filtered = df.copy()

# Metric to optimize
metric_options = ['vs QQQ %', 'Total Return %', 'CAGR %', 'Sharpe Ratio']
optimize_metric = st.sidebar.selectbox("Optimize By", metric_options, index=0)

# Risk filters
st.sidebar.subheader("Risk Constraints")
max_dd_threshold = st.sidebar.slider("Max Drawdown Threshold (%)", -100.0, 0.0, -50.0, 1.0)
min_sharpe = st.sidebar.slider("Min Sharpe Ratio", -2.0, 5.0, 0.0, 0.1)
min_trades = st.sidebar.number_input("Min Number of Trades", 0, 1000, 5, 1)

# Apply filters
df_filtered = df_filtered[
    (df_filtered['Max Drawdown %'] >= max_dd_threshold) &
    (df_filtered['Sharpe Ratio'] >= min_sharpe) &
    (df_filtered['Trades'] >= min_trades)
]

st.sidebar.markdown(f"**{len(df_filtered):,}** combinations match filters")

# Main analysis tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Top Performers", 
    "📊 Parameter Analysis", 
    "🎯 Consistency Score",
    "📉 Risk vs Return",
    "🔬 Deep Dive"
])

# ===== TAB 1: Top Performers =====
with tab1:
    st.header("🏆 Top Performing Combinations")
    
    # Show top N
    top_n = st.slider("Show Top N Results", 5, 50, 20)
    df_top = df_filtered.nlargest(top_n, optimize_metric)
    
    # Display table
    display_cols = [
        'Period', 'Parameters', 'Total Return %', 'vs QQQ %', 
        'CAGR %', 'Max Drawdown %', 'Sharpe Ratio', 'Trades'
    ]
    
    st.dataframe(
        df_top[display_cols].style.format({
            'Total Return %': '{:.2f}%',
            'vs QQQ %': '{:.2f}%',
            'CAGR %': '{:.2f}%',
            'Max Drawdown %': '{:.2f}%',
            'Sharpe Ratio': '{:.2f}',
            'Trades': '{:.0f}'
        }).background_gradient(subset=['vs QQQ %'], cmap='RdYlGn'),
        use_container_width=True,
        height=600
    )
    
    # Download filtered results
    st.download_button(
        "📥 Download Filtered Results",
        df_filtered.to_csv(index=False),
        "filtered_grid_results.csv",
        "text/csv"
    )

# ===== TAB 2: Parameter Analysis =====
with tab2:
    st.header("📊 Parameter Impact Analysis")
    
    param_cols = [
        'use_double_ema', 'ema_period', 'ema_fast', 'ema_slow',
        'use_rsi', 'rsi_threshold', 'use_stop_loss', 'stop_loss_pct',
        'use_bb', 'use_atr', 'use_msl_msh', 'use_macd', 'use_adx',
        'use_supertrend', 'use_pivot', 'pivot_left', 'pivot_right'
    ]
    
    # Boolean indicators analysis
    st.subheader("📍 Indicator Usage Impact")
    boolean_params = [col for col in param_cols if col.startswith('use_')]
    
    indicator_stats = []
    for param in boolean_params:
        if param in df_filtered.columns:
            enabled = df_filtered[df_filtered[param] == True]
            disabled = df_filtered[df_filtered[param] == False]
            
            if len(enabled) > 0 and len(disabled) > 0:
                indicator_stats.append({
                    'Indicator': param.replace('use_', '').replace('_', ' ').title(),
                    'Enabled Avg Return': enabled['Total Return %'].mean(),
                    'Disabled Avg Return': disabled['Total Return %'].mean(),
                    'Difference': enabled['Total Return %'].mean() - disabled['Total Return %'].mean(),
                    'Enabled Count': len(enabled),
                    'Disabled Count': len(disabled)
                })
    
    if indicator_stats:
        df_indicator_stats = pd.DataFrame(indicator_stats).sort_values('Difference', ascending=False)
        
        fig_indicators = px.bar(
            df_indicator_stats,
            x='Indicator',
            y='Difference',
            color='Difference',
            color_continuous_scale='RdYlGn',
            title=f'Impact of Each Indicator on {optimize_metric}',
            labels={'Difference': f'Avg Difference in {optimize_metric}'}
        )
        st.plotly_chart(fig_indicators, use_container_width=True)
        
        st.dataframe(
            df_indicator_stats.style.format({
                'Enabled Avg Return': '{:.2f}%',
                'Disabled Avg Return': '{:.2f}%',
                'Difference': '{:.2f}%'
            }).background_gradient(subset=['Difference'], cmap='RdYlGn'),
            use_container_width=True
        )
    
    # Numeric parameter analysis
    st.subheader("🔢 Numeric Parameter Ranges")
    
    col1, col2 = st.columns(2)
    
    with col1:
        numeric_param = st.selectbox(
            "Select Parameter",
            ['ema_period', 'ema_fast', 'ema_slow', 'rsi_threshold', 
             'stop_loss_pct', 'pivot_left', 'pivot_right']
        )
    
    with col2:
        analysis_metric = st.selectbox(
            "Metric to Analyze",
            ['Total Return %', 'vs QQQ %', 'CAGR %', 'Sharpe Ratio', 'Max Drawdown %']
        )
    
    if numeric_param in df_filtered.columns:
        # Group by parameter value and calculate statistics
        param_analysis = df_filtered.groupby(numeric_param).agg({
            analysis_metric: ['mean', 'median', 'std', 'count']
        }).reset_index()
        param_analysis.columns = [numeric_param, 'Mean', 'Median', 'Std Dev', 'Count']
        
        # Scatter plot
        fig_scatter = px.scatter(
            df_filtered,
            x=numeric_param,
            y=analysis_metric,
            color='Period',
            size='Trades',
            hover_data=['Parameters'],
            title=f'{analysis_metric} vs {numeric_param}',
            opacity=0.6
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Box plot
        fig_box = px.box(
            df_filtered,
            x=numeric_param,
            y=analysis_metric,
            title=f'Distribution of {analysis_metric} by {numeric_param}',
            points='outliers'
        )
        st.plotly_chart(fig_box, use_container_width=True)

# ===== TAB 3: Consistency Score =====
with tab3:
    st.header("🎯 Cross-Period Consistency Analysis")
    st.markdown("Find parameter combinations that perform well across **all** time periods")
    
    # Calculate consistency metrics for each unique parameter combination
    if 'Parameters' in df.columns:
        consistency_data = []
        
        for params_str in df['Parameters'].unique():
            param_rows = df[df['Parameters'] == params_str]
            
            if len(param_rows) >= 2:  # Need at least 2 periods
                consistency_data.append({
                    'Parameters': params_str,
                    'Periods Tested': len(param_rows),
                    'Avg Return': param_rows['Total Return %'].mean(),
                    'Std Return': param_rows['Total Return %'].std(),
                    'Min Return': param_rows['Total Return %'].min(),
                    'Max Return': param_rows['Total Return %'].max(),
                    'Avg vs QQQ': param_rows['vs QQQ %'].mean(),
                    'Min vs QQQ': param_rows['vs QQQ %'].min(),
                    'Avg Sharpe': param_rows['Sharpe Ratio'].mean(),
                    'Avg Max DD': param_rows['Max Drawdown %'].mean(),
                    'Worst Max DD': param_rows['Max Drawdown %'].min(),
                    'Total Trades': param_rows['Trades'].sum(),
                    # Consistency score: reward high avg return with low std dev
                    'Consistency Score': param_rows['Total Return %'].mean() / (param_rows['Total Return %'].std() + 1)
                })
        
        if consistency_data:
            df_consistency = pd.DataFrame(consistency_data)
            
            # Sort by consistency score
            df_consistency = df_consistency.sort_values('Consistency Score', ascending=False)
            
            st.subheader("📊 Most Consistent Performers")
            st.markdown("""
            **Consistency Score** = Average Return / (Std Dev + 1)
            - Higher score = More consistent performance across periods
            - Focus on combinations that work in multiple market conditions
            """)
            
            # Filters
            col1, col2 = st.columns(2)
            with col1:
                min_periods = st.slider("Min Periods Tested", 1, df['Period'].nunique(), 2)
            with col2:
                min_avg_return = st.slider("Min Average Return (%)", 0.0, 500.0, 0.0, 10.0)
            
            df_consistent = df_consistency[
                (df_consistency['Periods Tested'] >= min_periods) &
                (df_consistency['Avg Return'] >= min_avg_return)
            ]
            
            # Show top consistent combinations
            top_consistent = st.slider("Show Top N Consistent", 5, 50, 15)
            
            display_consistency_cols = [
                'Parameters', 'Consistency Score', 'Periods Tested',
                'Avg Return', 'Std Return', 'Min Return', 'Max Return',
                'Avg vs QQQ', 'Min vs QQQ', 'Avg Sharpe', 'Avg Max DD'
            ]
            
            st.dataframe(
                df_consistent.head(top_consistent)[display_consistency_cols].style.format({
                    'Consistency Score': '{:.2f}',
                    'Avg Return': '{:.2f}%',
                    'Std Return': '{:.2f}%',
                    'Min Return': '{:.2f}%',
                    'Max Return': '{:.2f}%',
                    'Avg vs QQQ': '{:.2f}%',
                    'Min vs QQQ': '{:.2f}%',
                    'Avg Sharpe': '{:.2f}',
                    'Avg Max DD': '{:.2f}%'
                }).background_gradient(subset=['Consistency Score'], cmap='RdYlGn'),
                use_container_width=True,
                height=600
            )
            
            # Scatter plot: Consistency vs Return
            fig_consistency = px.scatter(
                df_consistent,
                x='Std Return',
                y='Avg Return',
                size='Periods Tested',
                color='Consistency Score',
                hover_data=['Parameters', 'Min vs QQQ', 'Avg vs QQQ'],
                title='Return vs Volatility (Consistency Analysis)',
                labels={'Std Return': 'Return Std Dev (%)', 'Avg Return': 'Average Return (%)'},
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_consistency, use_container_width=True)
            
            # Download consistency analysis
            st.download_button(
                "📥 Download Consistency Analysis",
                df_consistent.to_csv(index=False),
                "consistency_analysis.csv",
                "text/csv"
            )

# ===== TAB 4: Risk vs Return =====
with tab4:
    st.header("📉 Risk-Return Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        x_metric = st.selectbox("X-Axis", ['Max Drawdown %', 'Sharpe Ratio', 'Trades'], index=0)
    with col2:
        y_metric = st.selectbox("Y-Axis", ['Total Return %', 'vs QQQ %', 'CAGR %'], index=1)
    
    # Scatter plot with color by period
    fig_risk = px.scatter(
        df_filtered,
        x=x_metric,
        y=y_metric,
        color='Period',
        size='Trades',
        hover_data=['Parameters', 'Sharpe Ratio'],
        title=f'{y_metric} vs {x_metric}',
        opacity=0.7
    )
    
    # Add efficient frontier line (top performers at each risk level)
    if x_metric == 'Max Drawdown %':
        # Group by drawdown buckets and find max return
        df_filtered['DD_Bucket'] = pd.cut(df_filtered[x_metric], bins=20)
        frontier = df_filtered.groupby('DD_Bucket')[y_metric].max().reset_index()
        frontier['DD_Mid'] = frontier['DD_Bucket'].apply(lambda x: x.mid)
        
        fig_risk.add_trace(
            go.Scatter(
                x=frontier['DD_Mid'],
                y=frontier[y_metric],
                mode='lines',
                name='Efficient Frontier',
                line=dict(color='red', width=2, dash='dash')
            )
        )
    
    st.plotly_chart(fig_risk, use_container_width=True)
    
    # Risk-adjusted returns
    st.subheader("🎯 Best Risk-Adjusted Returns")
    
    # Calculate risk-adjusted metric (custom)
    df_filtered['Risk_Adjusted_Return'] = (
        df_filtered['Total Return %'] / 
        (abs(df_filtered['Max Drawdown %']) + 1)
    )
    
    df_risk_adjusted = df_filtered.nlargest(20, 'Risk_Adjusted_Return')
    
    st.dataframe(
        df_risk_adjusted[display_cols + ['Risk_Adjusted_Return']].style.format({
            'Total Return %': '{:.2f}%',
            'vs QQQ %': '{:.2f}%',
            'CAGR %': '{:.2f}%',
            'Max Drawdown %': '{:.2f}%',
            'Sharpe Ratio': '{:.2f}',
            'Trades': '{:.0f}',
            'Risk_Adjusted_Return': '{:.2f}'
        }).background_gradient(subset=['Risk_Adjusted_Return'], cmap='RdYlGn'),
        use_container_width=True
    )

# ===== TAB 5: Deep Dive =====
with tab5:
    st.header("🔬 Deep Dive into Specific Combination")
    
    # Select a specific parameter combination
    param_options = sorted(df['Parameters'].unique().tolist())
    selected_params = st.selectbox(
        "Select Parameter Combination",
        param_options,
        index=0
    )
    
    df_selected = df[df['Parameters'] == selected_params]
    
    if not df_selected.empty:
        st.subheader(f"📋 Performance Across Periods: {selected_params}")
        
        # Show metrics for each period
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Return", f"{df_selected['Total Return %'].mean():.2f}%")
        with col2:
            st.metric("Avg vs QQQ", f"{df_selected['vs QQQ %'].mean():.2f}%")
        with col3:
            st.metric("Avg Sharpe", f"{df_selected['Sharpe Ratio'].mean():.2f}")
        with col4:
            st.metric("Periods Tested", len(df_selected))
        
        # Bar chart by period
        fig_periods = go.Figure()
        
        fig_periods.add_trace(go.Bar(
            x=df_selected['Period'],
            y=df_selected['Total Return %'],
            name='Strategy Return',
            marker_color='lightblue'
        ))
        
        fig_periods.add_trace(go.Bar(
            x=df_selected['Period'],
            y=df_selected['QQQ Return %'],
            name='QQQ Return',
            marker_color='orange'
        ))
        
        fig_periods.update_layout(
            title='Returns by Period',
            xaxis_title='Time Period',
            yaxis_title='Return (%)',
            barmode='group'
        )
        
        st.plotly_chart(fig_periods, use_container_width=True)
        
        # Detailed table
        st.subheader("📊 Detailed Metrics by Period")
        st.dataframe(
            df_selected[display_cols].style.format({
                'Total Return %': '{:.2f}%',
                'vs QQQ %': '{:.2f}%',
                'CAGR %': '{:.2f}%',
                'Max Drawdown %': '{:.2f}%',
                'Sharpe Ratio': '{:.2f}',
                'Trades': '{:.0f}'
            }),
            use_container_width=True
        )
        
        # Show individual parameter values
        st.subheader("⚙️ Parameter Configuration")
        param_config_cols = [col for col in param_cols if col in df_selected.columns]
        if param_config_cols:
            param_values = df_selected[param_config_cols].iloc[0].to_dict()
            
            col1, col2, col3 = st.columns(3)
            
            for idx, (param, value) in enumerate(param_values.items()):
                with [col1, col2, col3][idx % 3]:
                    st.metric(param.replace('_', ' ').title(), str(value))

# Footer
st.markdown("---")
st.markdown("💡 **Tip**: Use the consistency analysis to find robust strategies that work across multiple market conditions")
