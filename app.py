import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from prophet_model import get_comprehensive_analysis
from utils import load_dataset, load_default_dataset

# Page Configuration
st.set_page_config(
    page_title="Smart Energy Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp { background-color: #0d1117; }
    h1, h2, h3 { color: #58a6ff !important; }
    .stMetric { background-color: transparent !important; }
    div[data-testid="stMetricValue"] { font-size: 24px !important; color: #58a6ff !important; }
    div[data-testid="stMetricLabel"] { color: #8b949e !important; }
</style>
""", unsafe_allow_html=True)

# Header
st.title("⚡ Smart Energy Analytics")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    st.markdown("### Data Source")
    use_default = st.radio(
        "Select Data",
        ["Use Sample Dataset", "Upload Custom CSV"],
        horizontal=True
    )
    
    st.markdown("---")
    st.markdown("### Forecast Settings")
    forecast_days = st.slider(
        "Forecast Period",
        min_value=7,
        max_value=90,
        value=30
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.info("Smart Energy Analytics uses Facebook Prophet for time series forecasting.")

# Load Data
df = None

if use_default == "Use Sample Dataset":
    with st.spinner("Loading sample dataset..."):
        df = load_default_dataset()
    if df is not None:
        st.success(f"✅ Loaded sample dataset with {len(df):,} records")
else:
    uploaded_file = st.file_uploader(
        "Upload your CSV file",
        type=["csv", "txt"],
        help="CSV with Date, Time, and power consumption columns"
    )
    if uploaded_file:
        with st.spinner("Processing uploaded file..."):
            df = load_dataset(uploaded_file)
        st.success(f"✅ Loaded {len(df):,} records from uploaded file")

# Main Dashboard
if df is not None:
    # Overview Metrics
    st.markdown("## 📊 Overview")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Records", f"{len(df):,}")
    col2.metric("Date Range", f"{(df['ds'].max() - df['ds'].min()).days} days")
    col3.metric("Avg Consumption", f"{df['y'].mean():.2f} kW")
    col4.metric("Max Consumption", f"{df['y'].max():.2f} kW")
    col5.metric("Min Consumption", f"{df['y'].min():.2f} kW")
    
    st.markdown("---")
    
    # Run Analysis
    with st.spinner("🔄 Running comprehensive analysis..."):
        results = get_comprehensive_analysis(df, forecast_days)
    
    # Forecast Section
    st.markdown("## 🔮 Forecast Results")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("MAE", f"{results['forecast']['mae']:.4f}")
    col2.metric("RMSE", f"{results['forecast']['rmse']:.4f}")
    col3.metric("MAPE", f"{results['forecast']['mape']:.2f}%")
    col4.metric("Forecast Days", f"{forecast_days}")
    
    st.plotly_chart(results['forecast']['chart'], use_container_width=True)
    
    # Download Forecast
    csv = results['forecast']['output'].to_csv(index=False).encode("utf-8")
    st.download_button("📥 Download Forecast CSV", csv, "forecast_results.csv", "text/csv")
    
    st.markdown("---")
    
    # Peak Hours Analysis
    st.markdown("## ⏰ Peak Hours Analysis")
    
    peak = results['peak_analysis']
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Peak Hours %", f"{peak['peak_percentage']:.1f}%")
    col2.metric("Non-Peak Hours %", f"{peak['non_peak_percentage']:.1f}%")
    col3.metric("Peak Avg (kW)", f"{peak['peak_avg']:.2f}")
    
    st.info(f"Peak hours: {peak['peak_hours']['morning']} and {peak['peak_hours']['evening']}")
    
    st.markdown("---")
    
    # Hourly Distribution
    st.markdown("## 🕐 Hourly Distribution")
    
    hourly = results['hourly_analysis']
    hours = list(hourly['hourly_averages'].keys())
    values = list(hourly['hourly_averages'].values())
    percentages = list(hourly['hourly_percentage'].values())
    
    fig_hourly = go.Figure()
    colors = ['#ff6b6b' if h in [6,7,8,17,18,19,20,21] else '#4ecdc4' for h in hours]
    fig_hourly.add_trace(go.Bar(
        x=hours,
        y=values,
        marker_color=colors,
        text=[f"{p:.1f}%" for p in percentages],
        textposition='outside'
    ))
    fig_hourly.update_layout(
        title="Hourly Consumption Pattern",
        xaxis_title="Hour of Day",
        yaxis_title="Average Power (kW)",
        template="plotly_dark",
        height=400,
        xaxis=dict(tickmode='linear', dtick=1)
    )
    st.plotly_chart(fig_hourly, use_container_width=True)
    
    col1, col2 = st.columns(2)
    col1.info(f"Peak Hour: {hourly['peak_hour']:02d}:00")
    col2.info(f"Lowest Hour: {hourly['lowest_hour']:02d}:00")
    
    st.markdown("---")
    
    # Daily Patterns
    st.markdown("## 📅 Daily Patterns")
    
    daily = results['daily_analysis']
    days = list(daily['daily_averages'].keys())
    values = list(daily['daily_averages'].values())
    
    fig_daily = go.Figure()
    fig_daily.add_trace(go.Bar(
        x=days,
        y=values,
        marker_color=px.colors.qualitative.Prism,
        text=[f"{v:.2f}" for v in values],
        textposition='outside'
    ))
    fig_daily.update_layout(
        title="Consumption by Day of Week",
        xaxis_title="Day",
        yaxis_title="Average Power (kW)",
        template="plotly_dark",
        height=400
    )
    st.plotly_chart(fig_daily, use_container_width=True)
    
    col1, col2 = st.columns(2)
    col1.success(f"Highest: {daily['highest_day']}")
    col2.warning(f"Lowest: {daily['lowest_day']}")
    
    st.markdown("---")
    
    # Seasonal Patterns
    st.markdown("## 🌸 Seasonal Patterns")
    
    seasonal = results['seasonal_analysis']
    seasons = list(seasonal['seasonal_averages'].keys())
    values = list(seasonal['seasonal_averages'].values())
    
    fig_seasonal = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "pie"}, {"type": "bar"}]],
        subplot_titles=["Seasonal Distribution", "Seasonal Comparison"]
    )
    
    fig_seasonal.add_trace(
        go.Pie(
            labels=seasons,
            values=values,
            hole=0.4,
            marker=dict(colors=['#ff6b6b', '#4ecdc4', '#ffe66d', '#95e1d3'])
        ),
        row=1, col=1
    )
    
    fig_seasonal.add_trace(
        go.Bar(
            x=seasons,
            y=values,
            marker_color=['#ff6b6b', '#4ecdc4', '#ffe66d', '#95e1d3']
        ),
        row=1, col=2
    )
    
    fig_seasonal.update_layout(
        title="Seasonal Consumption Analysis",
        template="plotly_dark",
        height=400
    )
    st.plotly_chart(fig_seasonal, use_container_width=True)
    
    col1, col2 = st.columns(2)
    col1.error(f"Highest Season: {seasonal['highest_season']}")
    col2.success(f"Lowest Season: {seasonal['lowest_season']}")
    
    st.markdown("---")
    
    # Reduction Ideas
    st.markdown("## 💡 Energy Reduction Ideas")
    
    for idea in results['reduction_ideas']:
        with st.expander(f"{idea['title']} - {idea['potential_savings']}"):
            st.markdown(f"**Category:** {idea['category']}")
            st.markdown(f"**Description:** {idea['description']}")
            st.markdown(f"**Potential Savings:** {idea['potential_savings']}")
    
    st.markdown("---")
    
    # Executive Summary
    st.markdown("## 📋 Executive Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Key Insights")
        st.markdown(f"- Peak hours account for **{peak['peak_percentage']:.1f}%** of total consumption")
        st.markdown(f"- Highest consumption on **{daily['highest_day']}**")
        st.markdown(f"- **{seasonal['highest_season']}** is the most energy-intensive season")
        st.markdown(f"- Peak consumption occurs at **{hourly['peak_hour']:02d}:00**")
    
    with col2:
        st.markdown("### Recommended Actions")
        st.markdown("1. Shift high-energy tasks to off-peak hours")
        st.markdown("2. Implement smart scheduling for HVAC")
        st.markdown("3. Install real-time energy monitoring")
        st.markdown("4. Upgrade to energy-efficient appliances")
        st.markdown("5. Create awareness programs")

else:
    st.markdown("""
    <div style='text-align: center; padding: 80px 20px;'>
        <h2 style='color: #58a6ff;'>👈 Get Started</h2>
        <p style='color: #8b949e; font-size: 18px;'>Select a data source from the sidebar to begin analysis.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("*⚡ Smart Energy Analytics | Powered by Facebook Prophet & Plotly*")