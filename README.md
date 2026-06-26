# Household Electricity Consumption Forecasting

A time-series forecasting pipeline built on the UCI Household Power Consumption dataset (~2M records, 2006–2010). Engineered for scale: data cleaning, model tuning, and a self-service Streamlit dashboard.

---

## 🎯 Problem Statement

Forecast household electricity consumption 30 days ahead and identify peak consumption windows to quantify optimization opportunities.

---

## 🏗️ Engineering Decisions

- **Data Pipeline:** Processed ~2 million records using Pandas and NumPy — handled missing values, outliers, and temporal alignment at scale
- **Model:** Facebook Prophet tuned for daily seasonality patterns; evaluated on out-of-sample 30-day forecasts
- **Insight Generation:** Identified peak windows (06:00–09:00 and 17:00–21:00) accounting for ~44% of daily load; translated into quantified recommendations

---

## 📊 Forecast Performance (30-Day Out-of-Sample)

| Metric | Value |
|---|---|
| MAE | 0.2491 kW |
| RMSE | 0.3149 kW |
| MAPE | 17.63% |

---

## 💡 Key Findings

- Peak consumption windows (06:00–09:00 and 17:00–21:00) account for **~44% of daily load**
- HVAC optimization potential: **10–30% savings**
- Load shifting opportunity: **15–25% reduction**

---

## 🛠️ Tech Stack

- **Language:** Python
- **Forecasting:** Facebook Prophet
- **Data Processing:** Pandas, NumPy
- **Dashboard:** Streamlit, Plotly
- **Dataset:** UCI Household Power Consumption (2006–2010)

---

## 🚀 How to Run

```bash
git clone https://github.com/DivyanshiChaturvedi/Electricity_Forecast
cd Electricity_Forecast
pip install -r requirements.txt
python prophet_model.py
streamlit run app.py
```

---

## 📁 Project Structure

```
Electricity_Forecast/
├── data/              # UCI dataset
├── outputs/           # Forecast outputs + charts
├── prophet_model.py   # Model training + evaluation
├── utils.py           # Data cleaning utilities
├── app.py             # Streamlit dashboard
└── requirements.txt
```
