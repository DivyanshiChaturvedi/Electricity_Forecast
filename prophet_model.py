import pandas as pd
import numpy as np
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

def run_prophet_model(df, forecast_days):

    daily_df = df.resample("D", on="ds").mean().reset_index()

    train = daily_df[:-forecast_days]
    test = daily_df[-forecast_days:].reset_index(drop=True)

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        changepoint_prior_scale=0.05
    )

    model.fit(train)

    future = model.make_future_dataframe(periods=forecast_days)

    forecast = model.predict(future)

    pred = forecast[["ds", "yhat"]].tail(forecast_days).reset_index(drop=True)

    mae = mean_absolute_error(test["y"], pred["yhat"])
    rmse = np.sqrt(mean_squared_error(test["y"], pred["yhat"]))
    mape = np.mean(
        np.abs((test["y"] - pred["yhat"]) / test["y"])
    ) * 100

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=test["ds"],
        y=test["y"],
        mode="lines",
        name="Actual"
    ))

    fig.add_trace(go.Scatter(
        x=pred["ds"],
        y=pred["yhat"],
        mode="lines",
        name="Forecast"
    ))

    fig.update_layout(
        title="Actual vs Forecast"
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