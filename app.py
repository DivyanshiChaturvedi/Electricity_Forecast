import streamlit as st
import pandas as pd
from prophet_model import run_prophet_model
from utils import load_dataset, load_default_dataset

st.set_page_config(
    page_title="⚡ Electricity Forecasting Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #262730;
        padding: 20px;
        border-radius: 10px;
    }
    div[data-testid="stMetricValue"] {
        font-size: 28px;
    }
    .title-container {
        padding: 20px;
        background: linear-gradient(90deg, #1f77b4, #00cc96);
        border-radius: 10px;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("⚡ Electricity Consumption Forecasting")
st.markdown("---")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("### Data Source")
    
    use_default = st.radio(
        "Select Data Source",
        ["Use Sample Dataset", "Upload Custom CSV"],
        horizontal=True
    )
    
    st.markdown("---")
    st.markdown("### Forecast Settings")
    forecast_days = st.slider(
        "Forecast Period (Days)",
        min_value=7,
        max_value=90,
        value=30,
        help="Number of days to forecast ahead"
    )
    
    st.markdown("---")
    st.markdown("### About")
    st.info("This dashboard uses Facebook Prophet for time series forecasting of electricity consumption.")

# Main content
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
        help="Upload a CSV file with Date, Time, and power consumption columns"
    )
    if uploaded_file:
        with st.spinner("Processing uploaded file..."):
            df = load_dataset(uploaded_file)
        st.success(f"✅ Loaded {len(df):,} records from uploaded file")

# Display and forecast
if df is not None:
    st.markdown("---")
    
    # Dataset Preview
    col1, col2 = st.columns([3, 1])
    with col1:
        st.subheader("📊 Dataset Preview")
    with col2:
        st.metric("Total Records", f"{len(df):,}")
    
    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True
    )
    
    # Quick stats
    st.markdown("### 📈 Data Statistics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Min Value", f"{df['y'].min():.2f}")
    col2.metric("Max Value", f"{df['y'].max():.2f}")
    col3.metric("Mean Value", f"{df['y'].mean():.2f}")
    col4.metric("Date Range", f"{(df['ds'].max() - df['ds'].min()).days} days")
    
    st.markdown("---")
    
    # Run Forecast
    st.subheader("🔮 Forecast Results")
    
    with st.spinner("Running Prophet model... This may take a moment."):
        result = run_prophet_model(df, forecast_days)
    
    # Display chart
    st.plotly_chart(result["chart"], use_container_width=True)
    
    st.markdown("---")
    
    # Model Metrics
    st.subheader("📉 Model Performance Metrics")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"{result['mae']:.4f}")
    col2.metric("RMSE", f"{result['rmse']:.4f}")
    col3.metric("MAPE", f"{result['mape']:.2f}%")
    
    st.markdown("---")
    
    # Download
    st.subheader("💾 Export Results")
    
    csv = result["output"].to_csv(index=False).encode("utf-8")
    
    col1, col2 = st.columns([1, 4])
    with col1:
        st.download_button(
            label="📥 Download Forecast CSV",
            data=csv,
            file_name="forecast_results.csv",
            mime="text/csv",
            use_container_width=True
        )
    
else:
    # Empty state
    st.markdown("""
    <div style='text-align: center; padding: 50px;'>
        <h3>👈 Get Started</h3>
        <p>Select a data source from the sidebar to begin forecasting.</p>
    </div>
    """, unsafe_allow_html=True)