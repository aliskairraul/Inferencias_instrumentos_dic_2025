import polars as pl
import copy
from datetime import date, datetime, timezone, timedelta
from utils.utils import paths
from functions.compute_rsi import compute_rsi
from functions.volatilidades import volatilidades


def add_bollinger_bands(df: pl.DataFrame, elegido: str, std_factor: float = 2.0) -> pl.DataFrame:
    window = 14 if elegido == "BTCUSD" else 14

    df = df.sort("date").with_columns([
        (pl.col(f"ema_{window}") + pl.col("close").rolling_std(window) * std_factor).alias("bollinger_upper"),
        (pl.col(f"ema_{window}") - pl.col("close").rolling_std(window) * std_factor).alias("bollinger_lower"),
    ])

    # df = df.sort("date").with_columns([
    #     (pl.col("close").rolling_mean(window) + pl.col("close").rolling_std(window) * std_factor).alias("bollinger_upper"),
    #     (pl.col("close").rolling_mean(window) - pl.col("close").rolling_std(window) * std_factor).alias("bollinger_lower"),
    # ])

    df = df.with_columns(((pl.col("bollinger_upper") - pl.col("bollinger_lower")) / pl.col("close")).alias("bollinger_volat"))

    df = df.with_columns(((pl.col("close") - pl.col('bollinger_lower')) / (pl.col("bollinger_upper") - pl.col("bollinger_lower"))).alias("bollinger_posicion"))

    return df


def add_maximos_minimos(df: pl.DataFrame) -> pl.DataFrame:

    df = df.sort("date").with_columns([
        pl.col("high").rolling_max(3).alias("maximo_3"),
        pl.col("low").rolling_min(3).alias("minimo_3")
    ])

    df = df.with_columns([
        ((pl.col("open") - pl.col("low")) / (pl.col("high") - pl.col("low"))).alias("openVsVariacion"),
        ((pl.col("close") - pl.col("low")) / (pl.col("high") - pl.col("low"))).alias("closeVsVariacion"),
        ((pl.col("close") - pl.col("minimo_3")) / (pl.col("maximo_3") - pl.col("minimo_3"))).alias("closeVsMaxMin")
    ])

    df = df.drop(["low", "high", "maximo_3", "minimo_3"])

    return df


def add_features(df: pl.DataFrame, elegido: str, only_predict: bool = False) -> pl.DataFrame:
    columns = [x for x in df.columns if x not in ["date", "open"]]

    dias = [1, 3, 7, 14, 28, 56, 90] if elegido == "BTCUSD" else [1, 3, 5, 10, 20, 40, 60]
    for column in columns:
        for dia in dias:
            df = df.sort("date").with_columns(pl.col(column).shift(dia).alias(f"{column}_lag_{dia}"))

    # **CALCULAR _change_1d
    for dia in dias:
        df = df.sort("date").with_columns(((pl.col("close") - pl.col("close").shift(dia)) / pl.col("close").shift(dia)).alias(f"{elegido}_change_{dia}d"))

    # Lags de precio Variados
    lags = [5, 10, 20, 40, 60] if elegido != "BTCUSD" else [7, 15, 30, 60, 90]
    for lag in lags:
        df = df.sort("date").with_columns(((pl.col("close") / pl.col("close").shift(lag)) - 1).alias(f"{elegido}_momentum_{lag}d"))
        # df = df.sort("date").with_columns(pl.col("close").shift(lag).alias(f"{elegido}_lag_{lag}"))
    for lag in lags:
        df = df.sort("date").with_columns(pl.col("close").rolling_std(lag).alias(f"std_{elegido}_{lag}"))

    # calculo del Rsi
    df = df.with_columns([compute_rsi(df, "close", 14).alias("rsi_14")])

    # Volatilidades
    df = volatilidades(df=df)

    # Emas
    emas = [3, 9, 14, 20] if elegido != "BTCUSD" else [3, 7, 11, 14]
    columns_emas = []
    for ema in emas:
        df = df.sort("date").with_columns((pl.col("close").ewm_mean(span=ema)).alias(f"ema_{ema}"))
        columns_emas.append(f"ema_{ema}")

    # Bandas de Bollinger
    df = add_bollinger_bands(df=df, elegido=elegido)

    # Maximos y Mínimos
    df = df.drop(["low", "high"])
    # df = add_maximos_minimos(df=df)

    df = df.with_columns([
        pl.col('date').dt.year().alias('year'),
        pl.col('date').dt.month().alias('month'),
        pl.col('date').dt.week().alias('week'),
        pl.col('date').dt.weekday().alias('weekday'),
        pl.col('date').dt.day().alias('day'),
        pl.col('date').dt.ordinal_day().alias('ordinal_day'),
        pl.col('date').dt.days_in_month().alias('days_in_month'),
        pl.col('date').dt.quarter().alias('quarter')
    ])

    feriados = pl.read_parquet(paths['feriados'])['date'].to_list()
    df = df.with_columns(pl.when(pl.col('date').is_in(feriados)).then(1).otherwise(0).alias('feriado_usa'))

    if elegido in ['BTCUSD', 'XAUUSD']:
        df = df.drop(['feriado_usa']).sort('date')

    # **TARGET: precio de mañana (SIEMPRE AL FINAL)**
    df = df.sort("date").with_columns(
        pl.when((pl.col('close').shift(-1) - pl.col('close')) > 0)
        .then(1)
        .when((pl.col('close').shift(-1) - pl.col('close')) < 0)
        .then(-1)
        .otherwise(0)
        .alias('target_direction')
    )
    df = df.with_columns(pl.col("close").shift(-1).alias("close_tomorrow"))  # Ojo con esta Linea
    df = df.drop('open')

    ayer = datetime.now(timezone.utc).date() - timedelta(days=1)

    mask = (df['target_direction'] == 0) & (df['date'] < ayer)
    df = df.filter(~mask)

    # Eliminar NaNs (ÚLTIMO PASO)
    max_offset = max(max(lags, default=0), max(dias, default=0), 1)
    if only_predict:
        return (df.slice(max_offset, df.shape[0] - max_offset)).sort("date").drop_nulls()

    df = df.filter(df['date'] < ayer)
    return (df.slice(max_offset, df.shape[0] - max_offset)).sort("date").drop_nulls()
