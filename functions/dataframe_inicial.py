import polars as pl
import requests
from pathlib import Path
from datetime import date
from utils.logger import get_logger
from utils.utils import paths, urls

logger = get_logger("Carga Data from github")


def cargar_data_github(instrumento: str):
    try:
        response = requests.get(urls[instrumento])
        if response.status_code == 200:
            with open(paths[instrumento], "wb") as f:
                f.write(response.content)
            logger.info(f"Data del instrumento {instrumento} actualizada correctamente")
        else:
            logger.info(f"Status code {response.status_code}")
    except Exception as e:
        logger.error(f"Error tratando de Cargar la data desde github:  {e}")
    return


def dataframe_inicial(paths: dict, elegido: str, actualiza_from_github: bool = False) -> pl.DataFrame:
    if actualiza_from_github:
        cargar_data_github(instrumento=elegido)

    # Cargar el dataset del instrumento elegido
    df_elegido = pl.read_parquet(paths[elegido]).select(["date", "open", "low", "high", "close"])

    if elegido == "BTCUSD":
        df_elegido = df_elegido.filter(df_elegido["date"] >= date(2011, 5, 5))
        df_elegido = df_elegido.sort("date").fill_null(strategy="forward")

    return df_elegido
