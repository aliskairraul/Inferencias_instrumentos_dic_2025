"""
trades_dia:                        Numero de Librerias que hacen trade ese dia
 wins_dia:                         Numero de Librerias que ganan ese dia
 direccion_cuantificada_dia:       Suma las Direcciones de las Librerias que hacen trade ese dia
 mayoria_estricta_dir:             1 Si la mayoria de las librerias que hacen trade ese dia SUBEN -1 si la esa mayoria BAJA. 0 si no hay mayoria
 mayoria_estricta_estrategia:      Igual que mayoria_estricta_dir pero en vez de 1 y -1 usa "SUBE" "BAJA" y ""
 mayoria_estricta_trade:           1 ó 0 Si hay mayoria estricta entre las librerias que hacen trade ese dia
 mayoria_estricta_win:             1 ó 0 Si hay mayoria estricta entre las librerias que hacen trade ese dia y ademas GANAN (aciertan la Direccion)
 mayoria_global_dir:               1 si la mayoria de las librerias TOTAL (hagan trade o no) SUBEN -1 si la esa mayoria BAJA. 0 si no hay mayoria
 mayoria_global_estrategia:        Igual que mayoria_global_dir pero en vez de 1 y -1 usa "SUBE" "BAJA" y ""
 mayoria_global_trade:             1 ó 0 Si hay mayoria global entre todas las librerias (hagan trade o no)
 mayoria_global_win:               1 ó 0 Si hay mayoria global entre todas las librerias (hagan trade o no) y ademas GANAN (aciertan la Direccion)

 Cuantificacion de Estrategias
 - Estrategia Individual:
        - Significa que cada Instrumento recibe 1/Num Instrumentos del Capital Al inicio de la Simulacion
        - Cada Libreria dentro de un Instrumento recibe 1/Num Librerias del Capital disponible o lote asignado del Instrumento el dia del trade
 - Estrategia Global:
        - Igual que la Estrategia Anterior cada Instrumento recibe 1/Num Instrumentos del Capital Al inicio de la Simulacion
        - En caso de haber trade, se realiza un unico Trade con TODO el Capital disponible o Lote asignado del Instrumento el dia del trade, en la direccion "mayoria_global_dir".
 - Estrategia Estricta:
        - Igual que la Estrategia Global, pero ahora la direccion la determina la "mayoria_estricta_dir"
 - Estrategia Capital ponderado:
        - Igual que todas al principio de la simulación el capital se divide entre los instrumentos.
        - Esta vez el monto disponible diario se reparte entre las librerias que hacen trades
        - Igual que en la estrategia de la mayoria estricta la mayoria de los que hacen trade determinan la direccion del unico trade, pero el monto o lote disponible se divide entre el numero de votantes y se multiplica por la diferencia entre Ganadores y Perdedores
        Ejemplo 4 Librerias que hacen trade ese dia a cada una le corrsponde 20% del disponible. 3 dicen Sube y 1 BAJA se hace un solo trade por el 50% del disponible   Formula --> monto / librerias con trade * (sumatoria de las direcciones de esas librerias).
        quedaria (monto / trades_dia * direccion_cuantificada_dia)
"""
import polars as pl
import numpy as np
import torch
from utils.utils import estrategias


def binariza_lineal(arr: np.ndarray, libreria: str) -> np.ndarray:
    '''
    sklearn    --> Valores entre [-1     1]
    lightgbm   --> solo `-1` ó `1`           modelo.predict_proba(X_test)
                                            [0.15, 0.85],  # 85% probabilidad de clase 1
                                            [0.60, 0.40],  # 60% probabilidad de clase -1

    xgboost    ---> Valores entre [entre -1.7  y   1.5]
    pytorch    ---> Valores entre [0 y 1]
    tensorflow ---> Valores entre [0 y 1]

    si la i esta en [0, 1, 2] esta binarizando una predicion de sklearn ó lightgbm ó xgboost resultados negativos -1 y positivos 1 punto_medio 0
    si la i esta en [3, 4] esta binarizando una predicion de pytorch o tensorflow donde 0.5 es el punto_medio
    '''
    punto_medio = 0.5 if libreria.lower() in ['pytorch', 'tensorflow'] else 0
    return np.where(arr > punto_medio, 1, np.where(arr < punto_medio, -1, 0))


def binariza_ponderada(arr: np.array, libreria: str, parametros: dict) -> np.array:
    '''
    Los parametros de Ponderacion vienen de esta forma
    parametros = {
        "sklearn": [-0.2, 0.2],
        "lightgbm": [0.66, 0.66],
        "xgboost": [-0.1, 0.1],
        "pytorch": [0.48, 0.52],
        "tensorflow": [0.45, 0.55]
    }
    '''
    if libreria.lower() == 'pytorch':
        return np.where(arr > parametros[libreria][1], 1, np.where(arr < parametros[libreria][0], -1, 0))

    if libreria.lower() == 'tensorflow':
        return np.where(arr > parametros[libreria][1], 1, np.where(arr < parametros[libreria][0], -1, 0))

    if libreria.lower() == 'sklearn':
        return np.where(arr > parametros[libreria][1], 1, np.where(arr < parametros[libreria][0], -1, 0))

    if libreria.lower() == 'xgboost':
        return np.where(arr > parametros[libreria][1], 1, np.where(arr < parametros[libreria][0], -1, 0))

    # Modelo lightgbm -> Viene un predict_proba
    lista = []
    for i in range(len(arr)):
        if arr[i][0] > parametros[libreria][0]:
            lista.append(-1)
            continue
        if arr[i][1] > parametros[libreria][1]:
            lista.append(1)
            continue
        lista.append(0)
    return np.array(lista)


def inferencias(modelos: dict, X: np.array, df: pl.DataFrame, parametros: dict, elegido: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    # Determinando el device para Pytorch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df_pred = df.select(['date', 'close', 'close_tomorrow', 'target_direction']).sort('date')
    df_pred_pond = df_pred.clone()

    librerias = ['sklearn', 'lightgbm', 'xgboost', 'pytorch', 'tensorflow']
    predicciones = {}
    predicciones_pond = {}
    for i, libreria in enumerate(librerias):
        modelo = modelos[libreria]
        if modelo is None:
            predicciones[librerias[i]] = None
            predicciones_pond[librerias[i]] = None
            continue

        if i != 3:
            if i == 1:
                pred_proba = modelo.predict_proba(X)
            pred = modelo.predict(X)
            if i == 4:  # Si el modelo es Tensorflow hay que hacer un reshape
                pred = pred.reshape((1, pred.shape[0])).flatten()

            # Se binariza la prediccion y se adjunta en la lista de predicciones
            pred_binarizada = binariza_lineal(arr=pred, libreria=librerias[i])
            predicciones[librerias[i]] = pred_binarizada

            # Se binariza la prediccion de manera ponderada y se adjunta en la lista de predicciones ponderadas
            pred_binarizada_ponderada = binariza_ponderada(arr=pred, libreria=librerias[i], parametros=parametros) if i != 1 else \
                binariza_ponderada(arr=pred_proba, libreria=librerias[i], parametros=parametros)
            predicciones_pond[librerias[i]] = pred_binarizada_ponderada
            continue

        # Si no entró en el if anterior se trata de un modelo de Pytorch
        modelo.eval()
        X_pytorch = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            pred_pytorch_t = modelo(X_pytorch.to(device))

        pred = pred_pytorch_t.cpu().detach().numpy().flatten()
        # Se binariza la prediccion y se adjunta en la lista de predicciones
        pred_binarizada = binariza_lineal(arr=pred, libreria=librerias[i])
        predicciones[librerias[i]] = pred_binarizada
        # Se binariza la prediccion de manera ponderada y se adjunta en la lista de predicciones ponderadas
        pred_binarizada_ponderada = binariza_ponderada(arr=pred, libreria=librerias[i], parametros=parametros)
        predicciones_pond[librerias[i]] = pred_binarizada_ponderada

    def determinar_trade(row: dict) -> int:
        for key in row.keys():
            if "estrategia" in key:
                col_estrategia = key
            else:
                libreria = key
        if row[libreria] == 0:
            return 0
        if (row[col_estrategia] == "SUBE" and row[libreria] == 1) or (row[col_estrategia] == "BAJA" and row[libreria] == -1):
            return 1
        if row[col_estrategia] == "AMBOS":
            return 1
        return 0

    for libreria in librerias:
        if predicciones[libreria] is None:
            df_pred_pond = df_pred_pond.sort("date").with_columns(pl.lit(0).alias(libreria))
        else:
            df_pred = df_pred.sort('date').with_columns(pl.Series(name=libreria, values=predicciones[libreria]))
            df_pred_pond = df_pred_pond.sort('date').with_columns(pl.Series(name=libreria, values=predicciones_pond[libreria]))

        df_pred_pond = df_pred_pond.with_columns(pl.lit(estrategias[elegido][libreria]).alias(f"{libreria}_estrategia"))
        df_pred_pond = df_pred_pond.with_columns(
            pl.struct([libreria, f"{libreria}_estrategia"]).map_elements(determinar_trade, return_dtype=pl.Int64).alias(f"{libreria}_trade")
        )
        df_pred_pond = df_pred_pond.with_columns(
            pl.when((pl.col("target_direction") == pl.col(libreria)) & (pl.col(f"{libreria}_trade") == 1))
              .then(1)
              .otherwise(0)
              .alias(f"{libreria}_win")
        )

    columns_trade = [f"{libreria}_trade" for libreria in librerias]
    columns_win = [f"{libreria}_win" for libreria in librerias]

    df_pred_pond = df_pred_pond.with_columns(
        pl.sum_horizontal(columns_trade).alias("trades_dia")
    )
    df_pred_pond = df_pred_pond.sort("date").with_columns(
        pl.sum_horizontal(columns_win).alias("wins_dia")
    )

    # BLOQUE MAYORIA ESTRICTA Y DIRECCION CUANTIFICADA DEL LOS TRADES DEL DIA *********************************************
    def mayoria_estricta(row: dict) -> dict:
        if row["trades_dia"] == 0:
            return {"direccion_cuantificada_dia": 0, "mayoria_estricta_dir": 0, "mayoria_estricta_estrategia": "", "mayoria_estricta_trade": 0}
        librerias_trade = [x.replace("_trade", "") for x in columns_trade if row[x] == 1]
        sumatoria_estrategias = sum(row[libreria] for libreria in librerias_trade)
        if sumatoria_estrategias > 0:
            return {"direccion_cuantificada_dia": sumatoria_estrategias, "mayoria_estricta_dir": 1, "mayoria_estricta_estrategia": "SUBE", "mayoria_estricta_trade": 1}
        elif sumatoria_estrategias < 0:
            return {"direccion_cuantificada_dia": sumatoria_estrategias, "mayoria_estricta_dir": -1, "mayoria_estricta_estrategia": "BAJA", "mayoria_estricta_trade": 1}
        else:
            return {"direccion_cuantificada_dia": 0, "mayoria_estricta_dir": 0, "mayoria_estricta_estrategia": "", "mayoria_estricta_trade": 0}

    df_pred_pond = df_pred_pond.with_columns(
        pl.struct(librerias + columns_trade + ["trades_dia"]).map_elements(mayoria_estricta).alias("mayoria_estricta")
    ).unnest("mayoria_estricta")

    df_pred_pond = df_pred_pond.with_columns(
        pl.when((pl.col("target_direction") == pl.col("mayoria_estricta_dir")) & (pl.col("mayoria_estricta_trade") == 1))
        .then(1)
        .otherwise(0)
        .alias("mayoria_estricta_win")
    )

    # BLOQUE MAYORIA GLOBAL *********************************************************************************************
    def mayoria_global(row: dict) -> dict:
        sumatoria_librerias = sum(row[libreria] for libreria in librerias)
        if sumatoria_librerias == 0:
            return {"mayoria_global_dir": 0, "mayoria_global_estrategia": "", "mayoria_global_trade": 0}
        if sumatoria_librerias > 0:
            return {"mayoria_global_dir": 1, "mayoria_global_estrategia": "SUBE", "mayoria_global_trade": 1}
        return {"mayoria_global_dir": -1, "mayoria_global_estrategia": "BAJA", "mayoria_global_trade": 1}

    df_pred_pond = df_pred_pond.with_columns(
        pl.struct(librerias).map_elements(mayoria_global).alias("mayoria_global")
    ).unnest("mayoria_global")
    df_pred_pond = df_pred_pond.with_columns(
        pl.when((pl.col("target_direction") == pl.col("mayoria_global_dir")) & (pl.col("mayoria_global_trade") == 1))
        .then(1)
        .otherwise(0)
        .alias("mayoria_global_win")
    )

    return df_pred, df_pred_pond
