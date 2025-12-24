import polars as pl
from pathlib import Path
from datetime import date

paths = {
    'BTCUSD': Path('data/db/btcusd-D1_2010-07-17_actualidad.parquet'),
    'EURUSD': Path('data/db/eurusd-D1_2000-01-03_actualidad.parquet'),
    'SPX': Path('data/db/sp500-D1_2000-01-03_actualidad.parquet'),
    'XAUUSD': Path('data/db/xauusd-D1_2000-01-03_actualidad.parquet'),
    'US10Y': Path('data/db/us10y-D1_2000-01-03_actualidad.parquet'),
    'USDX': Path('data/db/usdx-D1_2000-01-03_actualidad.parquet'),
    'feriados': Path('data/db/feriados_usa.parquet')
}

paths_inferencias = {
    "BTCUSD": Path("data/inferencias/predicciones_aciertos_btcusd.parquet"),
    "EURUSD": Path("data/inferencias/predicciones_aciertos_eurusd.parquet"),
    "SPX": Path("data/inferencias/predicciones_aciertos_spx.parquet"),
    "XAUUSD": Path("data/inferencias/predicciones_aciertos_xauusd.parquet")
}

urls = {
    'BTCUSD': "https://raw.githubusercontent.com/aliskairraul/Actualiza-Data-Instrumentos/main/db/btcusd-D1_2010-07-17_actualidad.parquet",
    'EURUSD': "https://raw.githubusercontent.com/aliskairraul/Actualiza-Data-Instrumentos/main/db/eurusd-D1_2000-01-03_actualidad.parquet",
    'SPX': "https://raw.githubusercontent.com/aliskairraul/Actualiza-Data-Instrumentos/main/db/sp500-D1_2000-01-03_actualidad.parquet",
    'XAUUSD': "https://raw.githubusercontent.com/aliskairraul/Actualiza-Data-Instrumentos/main/db/xauusd-D1_2000-01-03_actualidad.parquet",
    'US10Y': "https://raw.githubusercontent.com/aliskairraul/Actualiza-Data-Instrumentos/main/db/us10y-D1_2000-01-03_actualidad.parquet",
    'USDX': "https://raw.githubusercontent.com/aliskairraul/Actualiza-Data-Instrumentos/main/db/usdx-D1_2000-01-03_actualidad.parquet"
}

paths_db2 = {
    'BTCUSD': Path('db_2/btcusd-D1_2010-07-17_actualidad.parquet'),
    'EURUSD': Path('db_2/eurusd-D1_2000-01-03_actualidad.parquet'),
    'SPX': Path('db_2/sp500-D1_2000-01-03_actualidad.parquet'),
    'XAUUSD': Path('db_2/xauusd-D1_2000-01-03_actualidad.parquet'),
    'US10Y': Path('db_2/us10y-D1_2000-01-03_actualidad.parquet'),
    'USDX': Path('db_2/usdx-D1_2000-01-03_actualidad.parquet'),
    'feriados': Path('db_2/feriados_usa.parquet')
}

paths_data_accuracy = {
    'BTCUSD': Path('models/data/data_accuracys_models_BTCUSD.parquet'),
    'EURUSD': Path('models/data/data_accuracys_models_EURUSD.parquet'),
    'SPX': Path('models/data/data_accuracys_models_SPX.parquet'),
    'XAUUSD': Path('models/data/data_accuracys_models_XAUUSD.parquet')
}


def calendario(fecha_inicio: date, fecha_fin: date) -> pl.DataFrame:
    fechas = pl.date_range(
        start=fecha_inicio,
        end=fecha_fin,
        interval="1d",
        eager=True  # Esto devuelve directamente una Series
    )
    df = pl.DataFrame({'date': fechas})
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

    return df


estrategia_btcusd = {
    "sklearn": "BAJA",  # listo
    "xgboost": "BAJA",  # listo
    "lightgbm": "BAJA",  # listo
    "pytorch": "",  # listo
    "tensorflow": "BAJA"  # listo
}

estrategia_eurusd = {
    "sklearn": "SUBE",  # listo
    "xgboost": "SUBE",  # listo
    "lightgbm": "SUBE",  # listo
    "pytorch": "SUBE",  # listo
    "tensorflow": "SUBE"  # listo
}

estrategia_xauusd = {
    "sklearn": "",
    "xgboost": "SUBE",
    "lightgbm": "SUBE",
    "pytorch": "",
    "tensorflow": "SUBE"
}

estrategia_spx = {
    "sklearn": "SUBE",
    "xgboost": "SUBE",
    "lightgbm": "SUBE",
    "pytorch": "SUBE",
    "tensorflow": "SUBE"
}
estrategias = {"BTCUSD": estrategia_btcusd, "EURUSD": estrategia_eurusd, "XAUUSD": estrategia_xauusd, "SPX": estrategia_spx}
