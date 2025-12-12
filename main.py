import polars as pl
from pathlib import Path
import sys
import joblib
import json

from functions.dataframe_inicial import dataframe_inicial
from functions.add_features import add_features
from functions.split_evalua import split_prediccion
from functions.cargar_modelos import cargar_modelos
from functions.other_functions import inferencias, sumariza_data
from utils.utils import paths, estrategias, indice_estrategias, librerias, paths_resultados
from utils.logger import get_logger

logger = get_logger("Inferencia de Predicciones")
instrumentos_finacieros = ["BTCUSD", "EURUSD", "SPX", "XAUUSD"]


def main():
    for elegido in instrumentos_finacieros:
        try:
            # ubicando en Variables los parámetros de ponderación y scaler con que se entrenaron los modelos del instrumento financiero
            logger.info(f"Ubicando parametros de ponderacion, scaler del instrumento {elegido}")
            with open(f"models/{elegido}/parametros_ponderacion_{elegido}.json", "r") as f:
                parametros = json.load(f)
            ruta_scaler = Path("models/") / elegido / f"scaler_{elegido}.joblib"
            scaler = joblib.load(ruta_scaler)

            # Pipeline de Datos: Carga la data del Instrumento, aplica Ingenieria de Features y Splitea la Data
            logger.info(f"Creacion de Dataframe Inicial, Ing. de Features y Spliteo de instrumento {elegido}")
            df, X, y, df_ayer, X_ayer = (
                dataframe_inicial(paths=paths, elegido=elegido, actualiza_from_github=True)
                .pipe(lambda df: add_features(df=df, elegido=elegido, only_predict=True))
                .pipe(lambda df: split_prediccion(df=df, scaler=scaler))
            )

            # Cargando TODOS los modelos en una Matriz
            logger.info(f"Cargando la matriz con los modelos del Instrumento {elegido}")
            matriz_modelos = cargar_modelos(instrumento=elegido)

            # Obteniendo las Inferencias de los Modelos con los Datos de Evaluación (Desconocidos por los modelos)
            logger.info(f"Inferencia de las prediciones instrumento {elegido}")
            df_votos, df_votos_pond = inferencias(matriz_modelos=matriz_modelos, X=X, y=y, df=df, parametros=parametros)

            # Totalizando las Inferencias para poder mostrar Resultados
            logger.info(f"Sumarizando Informacion instrumento {elegido}")
            df_sumariza, models_columns = sumariza_data(df=df, df_votos_pond=df_votos_pond, df_votos=df_votos, elegido=elegido,
                                                        librerias=librerias, estrategias=estrategias, indice_estrategias=indice_estrategias)
            logger.info(f"Guardando data sumarizada del instrumento {elegido}")
            df_sumariza.write_parquet(paths_resultados[elegido])
        except Exception as e:
            logger.error(f"Error en la ejecusion durante el instrumento {elegido} --> {e}")
    return


if __name__ == "__main__":
    main()
    sys.exit(0)
