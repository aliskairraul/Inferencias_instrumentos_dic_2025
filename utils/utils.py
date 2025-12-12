import polars as pl
from pathlib import Path
from datetime import date

paths = {
    'BTCUSD': Path('db/btcusd-D1_2010-07-17_actualidad.parquet'),
    'EURUSD': Path('db/eurusd-D1_2000-01-03_actualidad.parquet'),
    'SPX': Path('db/sp500-D1_2000-01-03_actualidad.parquet'),
    'XAUUSD': Path('db/xauusd-D1_2000-01-03_actualidad.parquet'),
    'US10Y': Path('db/us10y-D1_2000-01-03_actualidad.parquet'),
    'USDX': Path('db/usdx-D1_2000-01-03_actualidad.parquet'),
    'feriados': Path('db/feriados_usa.parquet')
}

paths_resultados = {
    "BTCUSD": Path("models/data/predicciones_aciertos_btcusd.parquet"),
    "EURUSD": Path("models/data/predicciones_aciertos_eurusd.parquet"),
    "SPX": Path("models/data/predicciones_aciertos_spx.parquet"),
    "XAUUSD": Path("models/data/predicciones_aciertos_xauusd.parquet")
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
    "sklearn": "",
    "xgboost": "SUBE",
    "lightgbm": "",
    "pytorch": "SUBE",
    "tensorflow": "AMBOS"
}

estrategia_eurusd = {
    "sklearn": "SUBE",
    "xgboost": "SUBE",
    "lightgbm": "SUBE",
    "pytorch": "SUBE",
    "tensorflow": "BAJA"
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

estrategias = [estrategia_btcusd, estrategia_eurusd, estrategia_spx, estrategia_xauusd]
indice_estrategias = {"BTCUSD": 0, "EURUSD": 1, "SPX": 2, "XAUUSD": 3}
librerias = ["sklearn", "xgboost", "lightgbm", "pytorch", "tensorflow"]

'''
# Este String "20090522  07:30:00" Se transforma en 3 columnas nuevas una tipo `date`` y dos tipo `int``
def retorna_struct_datetime(cadena: str):
    cadena = cadena.replace("  ", " ")

    parte_1 = cadena.split(" ")[0]
    anio = int(parte_1[:4])
    mes = int(parte_1[4:6])
    dia = int(parte_1[6:])
    date_temp = date(anio, mes, dia)

    parte_2 = cadena.split(" ")[1]
    hour = int(parte_2.split(":")[0].strip())
    minute = int(parte_2.split(":")[1].strip())

    return {"date_temp": date_temp, "hour": hour, "minute": minute}


df = df.with_columns([
    pl.col("date")
      .map_elements(retorna_struct_datetime, return_dtype=pl.Struct({'date_temp': pl.Date, 'hour': pl.Int64, 'minute': pl.Int64}))
      .alias("datetime_struct")
]).unnest("datetime_struct")

df.head(1)
'''
