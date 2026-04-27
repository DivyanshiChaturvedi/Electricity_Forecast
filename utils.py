import pandas as pd
import os

def load_dataset(file):
    # Auto-detect separator by checking first line
    first_line = file.read(1024).decode('utf-8')
    file.seek(0)
    
    sep = ';' if ';' in first_line else ','
    
    df = pd.read_csv(
        file,
        sep=sep,
        low_memory=False,
        na_values='?'
    )
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Handle case where Date column might have different name
    date_col = 'Date' if 'Date' in df.columns else df.columns[0]
    time_col = 'Time' if 'Time' in df.columns else df.columns[1] if len(df.columns) > 1 else 'Time'
    
    df["Datetime"] = pd.to_datetime(
        df[date_col] + " " + df[time_col],
        format="%d/%m/%Y %H:%M:%S",
        errors='coerce'
    )
    
    # Find the power column
    power_col = None
    for col in df.columns:
        if 'active_power' in col.lower() or 'global_active_power' in col.lower():
            power_col = col
            break
    
    if power_col is None:
        power_col = df.select_dtypes(include=['number']).columns[0] if len(df.select_dtypes(include=['number']).columns) > 0 else df.columns[2]
    
    df = df[["Datetime", power_col]]
    df.columns = ["ds", "y"]
    df["y"] = pd.to_numeric(df["y"], errors="coerce")
    df.dropna(inplace=True)
    df = df.sort_values("ds")
    
    return df


def load_default_dataset():
    """Load the default household power consumption dataset"""
    default_path = os.path.join(os.path.dirname(__file__), "data", "household_power_consumption.csv")
    if os.path.exists(default_path):
        with open(default_path, 'rb') as f:
            return load_dataset(f)
    return None