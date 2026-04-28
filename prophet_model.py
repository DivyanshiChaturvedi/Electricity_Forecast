import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

def run_prophet_model(df, forecast_days):
    """Run Prophet forecasting model"""
    
    # Resample to daily for better forecasting
    daily_df = df.resample("D", on="ds").mean().reset_index()
    
    # Train-test split
    train = daily_df[:-forecast_days]
    test = daily_df[-forecast_days:].reset_index(drop=True)
    
    # Build and fit model
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.05
    )
    
    model.fit(train)
    
    # Make predictions
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)
    
    # Get predictions for test period
    pred = forecast[["ds", "yhat"]].tail(forecast_days).reset_index(drop=True)
    
    # Calculate metrics
    mae = mean_absolute_error(test["y"], pred["yhat"])
    rmse = np.sqrt(mean_squared_error(test["y"], pred["yhat"]))
    mape = np.mean(np.abs((test["y"] - pred["yhat"]) / test["y"])) * 100
    
    # Create chart
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=test["ds"], y=test["y"], mode="lines", name="Actual", line=dict(color="#00cc96")))
    fig.add_trace(go.Scatter(x=pred["ds"], y=pred["yhat"], mode="lines", name="Forecast", line=dict(color="#1f77b4", dash="dash")))
    
    fig.update_layout(
        title="📈 Electricity Consumption Forecast",
        xaxis_title="Date",
        yaxis_title="Power (kW)",
        template="plotly_dark",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    output = pd.DataFrame({
        "Date": pred["ds"],
        "Actual": test["y"],
        "Forecast": pred["yhat"]
    })
    
    return {
        "chart": fig,
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "output": output
    }


def analyze_peak_hours(df):
    """Analyze peak and non-peak hours"""
    df = df.copy()
    df['hour'] = df['ds'].dt.hour
    
    # Define peak hours (typically 6-9 AM and 5-9 PM)
    peak_morning = (df['hour'] >= 6) & (df['hour'] <= 9)
    peak_evening = (df['hour'] >= 17) & (df['hour'] <= 21)
    df['is_peak'] = peak_morning | peak_evening
    
    peak_avg = df[df['is_peak']]['y'].mean()
    non_peak_avg = df[~df['is_peak']]['y'].mean()
    
    return {
        "peak_hours": {
            "morning": "6:00 - 9:00",
            "evening": "17:00 - 21:00"
        },
        "peak_avg": peak_avg,
        "non_peak_avg": non_peak_avg,
        "peak_percentage": (peak_avg / (peak_avg + non_peak_avg)) * 100,
        "non_peak_percentage": (non_peak_avg / (peak_avg + non_peak_avg)) * 100
    }


def analyze_daily_patterns(df):
    """Analyze patterns by day of week"""
    df = df.copy()
    df['day'] = df['ds'].dt.day_name()
    
    daily_avg = df.groupby('day')['y'].mean()
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    daily_avg = daily_avg.reindex(day_order)
    
    return {
        "daily_averages": daily_avg.to_dict(),
        "highest_day": daily_avg.idxmax(),
        "lowest_day": daily_avg.idxmin()
    }


def analyze_weekly_patterns(df):
    """Analyze weekly patterns"""
    df = df.copy()
    df['week'] = df['ds'].dt.isocalendar().week
    
    weekly_avg = df.groupby('week')['y'].mean()
    
    return {
        "weekly_averages": weekly_avg.to_dict(),
        "avg_per_week": df['y'].sum() / df['ds'].dt.isocalendar().week.nunique()
    }


def analyze_seasonal_patterns(df):
    """Analyze seasonal patterns"""
    df = df.copy()
    df['month'] = df['ds'].dt.month
    
    # Define seasons
    def get_season(month):
        if month in [12, 1, 2]:
            return "Winter"
        elif month in [3, 4, 5]:
            return "Spring"
        elif month in [6, 7, 8]:
            return "Summer"
        else:
            return "Fall"
    
    df['season'] = df['month'].apply(get_season)
    seasonal_avg = df.groupby('season')['y'].mean()
    
    return {
        "seasonal_averages": seasonal_avg.to_dict(),
        "highest_season": seasonal_avg.idxmax(),
        "lowest_season": seasonal_avg.idxmin()
    }


def analyze_hourly_distribution(df):
    """Analyze hourly distribution with percentages"""
    df = df.copy()
    df['hour'] = df['ds'].dt.hour
    
    hourly_avg = df.groupby('hour')['y'].mean()
    total_avg = df['y'].mean()
    
    # Calculate percentage of total for each hour
    hourly_percentage = (hourly_avg / total_avg) * 100
    
    return {
        "hourly_averages": hourly_avg.to_dict(),
        "hourly_percentage": hourly_percentage.to_dict(),
        "peak_hour": hourly_avg.idxmax(),
        "lowest_hour": hourly_avg.idxmin()
    }


def generate_reduction_ideas(analysis_results):
    """Generate ideas to reduce electricity consumption"""
    ideas = []
    
    # Peak hour suggestions
    if 'peak_analysis' in analysis_results:
        peak = analysis_results['peak_analysis']
        if peak['peak_percentage'] > 40:
            ideas.append({
                "category": "Peak Hour Management",
                "title": "Shift Non-Essential Loads",
                "description": f"Peak hours show {peak['peak_percentage']:.1f}% of consumption. Consider running high-energy appliances (dishwasher, laundry) during off-peak hours (before 6 AM or after 9 PM).",
                "potential_savings": "15-25%"
            })
            ideas.append({
                "category": "Peak Hour Management",
                "title": "Implement Time-of-Use Pricing",
                "description": "Review utility rate structures and consider time-of-use plans that offer lower rates during off-peak periods.",
                "potential_savings": "10-20%"
            })
    
    # Daily pattern suggestions
    if 'daily_analysis' in analysis_results:
        daily = analysis_results['daily_analysis']
        ideas.append({
            "category": "Daily Optimization",
            "title": f"Focus on {daily['highest_day']} Consumption",
            "description": f"{daily['highest_day']} shows highest average consumption. Investigate what activities drive usage on this day.",
            "potential_savings": "5-15%"
        })
    
    # Seasonal suggestions
    if 'seasonal_analysis' in analysis_results:
        seasonal = analysis_results['seasonal_analysis']
        highest = seasonal['highest_season']
        ideas.append({
            "category": "Seasonal Efficiency",
            "title": f"Reduce {highest} Consumption",
            "description": f"{highest} has the highest consumption. Consider HVAC optimization, insulation improvements, or scheduling adjustments for this season.",
            "potential_savings": "10-30%"
        })
    
    # Hourly suggestions
    if 'hourly_analysis' in analysis_results:
        hourly = analysis_results['hourly_analysis']
        ideas.append({
            "category": "Hourly Optimization",
            "title": "Reduce Peak Hour Usage",
            "description": f"Peak consumption at {hourly['peak_hour']:02d}:00. Review equipment schedules and consider pre-cooling/pre-heating strategies.",
            "potential_savings": "10-20%"
        })
    
    # General suggestions
    ideas.extend([
        {
            "category": "Smart Monitoring",
            "title": "Install Smart Meters",
            "description": "Real-time monitoring helps identify unusual consumption patterns and waste sources.",
            "potential_savings": "5-10%"
        },
        {
            "category": "Equipment Efficiency",
            "title": "Upgrade to Energy-Efficient Appliances",
            "description": "Replace old appliances with ENERGY STAR rated equipment. Old refrigerators and HVAC systems can use 30-50% more energy.",
            "potential_savings": "10-20%"
        },
        {
            "category": "Behavioral Changes",
            "title": "Implement Awareness Programs",
            "description": "Simple behavioral changes like turning off lights, unplugging devices, and reducing standby power can save 5-15%.",
            "potential_savings": "5-15%"
        }
    ])
    
    return ideas


def get_comprehensive_analysis(df, forecast_days=30):
    """Get all analysis results combined"""
    results = {}
    
    # Run forecasting
    results['forecast'] = run_prophet_model(df, forecast_days)
    
    # Run all analyses
    results['peak_analysis'] = analyze_peak_hours(df)
    results['daily_analysis'] = analyze_daily_patterns(df)
    results['weekly_analysis'] = analyze_weekly_patterns(df)
    results['seasonal_analysis'] = analyze_seasonal_patterns(df)
    results['hourly_analysis'] = analyze_hourly_distribution(df)
    
    # Generate reduction ideas
    results['reduction_ideas'] = generate_reduction_ideas(results)
    
    return results