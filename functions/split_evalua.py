import polars as pl
import numpy as np
from datetime import date
from sklearn.preprocessing import StandardScaler


def split_prediccion(df: pl.DataFrame, scaler: StandardScaler):
    last_date = df['date'].max()
    df_last_date = df.filter(df['date'] == last_date)

    # df = df.filter((df['date'] >= date(2025, 8, 1)) & (df['date'] < last_date))   # Decidir con que modelo quedar
    df = df.filter(df['date'] >= date(2025, 8, 1))

    X_last = df_last_date.drop(['date', 'target_direction', "tipo_volatilidad", "close_tomorrow"]).to_numpy()
    X = df.sort('date').drop(['date', 'target_direction', "tipo_volatilidad", "close_tomorrow"]).to_numpy()

    y = df.sort('date')["target_direction"].to_numpy().flatten()

    X_last_scaled = scaler.transform(X_last)
    X_scaled = scaler.transform(X)

    return df, X_scaled, y, df_last_date, X_last_scaled
