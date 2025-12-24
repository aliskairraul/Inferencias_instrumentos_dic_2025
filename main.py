import polars as pl
from pathlib import Path
import sys
import joblib
import json
from functions.dataframe_inicial import dataframe_inicial
from functions.add_features import add_features
from functions.split_evalua import split_prediccion
from functions.cargar_diccionario_modelos import cargar_modelos
from functions.inferencias import inferencias
from utils.utils import paths, paths_inferencias
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
                .pipe(lambda df: add_features(df=df, elegido=elegido, only_predict=False))
                .pipe(lambda df: split_prediccion(df=df, scaler=scaler))
            )

            # Cargando TODOS los modelos en una Matriz
            logger.info(f"Cargando los modelos del Instrumento {elegido}")
            diccionario_modelos = cargar_modelos(instrumento=elegido)

            # Obteniendo las Inferencias de los Modelos con los Datos de Evaluación (Desconocidos por los modelos)
            logger.info(f"Inferencia de las prediciones instrumento {elegido}")
            df_pred, df_pred_pond = inferencias(modelos=diccionario_modelos, X=X, df=df, parametros=parametros, elegido=elegido)

            logger.info(f"Guardando data del instrumento {elegido}")
            df_pred_pond.write_parquet(paths_inferencias[elegido])
        except Exception as e:
            logger.error(f"Error en la ejecusion durante el instrumento {elegido} --> {e}")
    return


if __name__ == "__main__":
    main()
    sys.exit(0)
