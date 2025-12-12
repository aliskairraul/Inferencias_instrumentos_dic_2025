import polars as pl
import numpy as np
# from utils.utils import librerias, estrategias
# from sklearn.metrics import accuracy_score

import torch


def precision_direcciones(y: np.array, pred: np.array, iteracion: int) -> dict:
    """
    Precision: Es la proporción de verdaderos positivos entre todos los casos que el modelo clasificó como positivos.   VP / (VP + FP)
    Nota: un Modelo puede tener una Precision de 100% si solo considera que el precio subira 3 veces y esas 3 veces acierta. No significa
          que el modelo sirva.
    """

    def contar(arr: np.array, num: int) -> int:
        contador = 0
        for i in range(len(arr)):
            if arr[i] == num:
                contador += 1
        return contador

    subidas_real = contar(y, 1)
    subidas_pred = contar(pred, 1)
    bajadas_pred = contar(pred, -1)
    bajadas_real = contar(y, -1)
    subidas_acertadas = 0
    bajadas_acertadas = 0

    for i in range(len(pred)):
        if pred[i] == 1 and y[i] == 1:
            subidas_acertadas += 1
        if pred[i] == -1 and y[i] == -1:
            bajadas_acertadas += 1

    precision_subida = round((subidas_acertadas / subidas_pred), 4) if subidas_pred != 0 else 0
    precision_bajada = round((bajadas_acertadas / bajadas_pred), 4) if bajadas_pred != 0 else 0

    porc_bajadas_acertadas = round((bajadas_acertadas / bajadas_real), 4) if bajadas_real != 0 else 0
    porc_subidas_acertadas = round((subidas_acertadas / subidas_real), 4) if subidas_real != 0 else 0

    # subida_porc_acierto_X_precision = round((porc_subidas_acertadas * precision_subida), 4)
    # bajada_porc_acierto_X_precision = round((porc_bajadas_acertadas * precision_bajada), 4)

    porc_subidas = round((subidas_real / len(y)), 4) if len(y) != 0 else 0
    porc_bajadas = round((bajadas_real / len(y)), 4) if len(y) != 0 else 0

    return {
        "Accuracy": round(((subidas_acertadas + bajadas_acertadas) / (subidas_pred + bajadas_pred)), 4) if (subidas_pred + bajadas_pred) != 0 else 0,
        "iter": iteracion,
        "Total dias": len(y),
        "% Subidas": porc_subidas,
        "% Bajadas": porc_bajadas,
        "Dias Opera": subidas_pred + bajadas_pred,
        "Subidas": subidas_real,
        "Subidas-Pred": subidas_pred,
        "Subidas-Acert": subidas_acertadas,
        "Preci-Sub": precision_subida,
        "% Subidas-Acert": porc_subidas_acertadas,
        "Bajadas": bajadas_real,
        "Bajadas-Pred": bajadas_pred,
        "Bajadas-Acert": bajadas_acertadas,
        "Preci-Baj": precision_bajada,
        "% Bajadas-Acert": porc_bajadas_acertadas,
    }


def acumula_data_accuracy_modelos(best_general_model: pl.DataFrame,
                                  best_up_model: pl.DataFrame,
                                  best_down_model: pl.DataFrame,
                                  symbol: str,
                                  libreria: str,
                                  data: list) -> list:
    models = [best_general_model, best_up_model, best_down_model]
    cualidades = ['Mejor Accuracy General', 'Mejor Accuracy Subidas', 'Mejor Accuracy Bajadas']
    for i, model in enumerate(models):
        diccionario = {
            "symbol": symbol,
            "libreria": libreria,
            "Cualidad": cualidades[i],
            "Precisión General": model['Direction Accuracy'][0],
            "Precisión en Subidas": model['Precicion 1 (subirá)'][0],
            "Precisión em Bajada": model['Precision -1 (bajará)'][0]
        }
        data.append(diccionario)
    return data


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


def votaciones(df: pl.DataFrame) -> pl.DataFrame:
    def binariza(numero: int):
        return 1 if numero > 0 else -1 if numero < 0 else 0

    columns = [x for x in df.columns if x not in ['date', 'close', 'target_direction', 'close_tomorrow']]
    col_sklearn = [x for x in df.columns if x not in ['date', 'close', 'target_direction', 'close_tomorrow'] and 'sklearn' in x]
    col_lightgbm = [x for x in df.columns if x not in ['date', 'close', 'target_direction', 'close_tomorrow'] and 'lightgbm' in x]
    col_xgboost = [x for x in df.columns if x not in ['date', 'close', 'target_direction', 'close_tomorrow'] and 'xgboost' in x]
    col_pytorch = [x for x in df.columns if x not in ['date', 'close', 'target_direction', 'close_tomorrow'] and 'pytorch' in x]
    col_tensorflow = [x for x in df.columns if x not in ['date', 'close', 'target_direction', 'close_tomorrow'] and 'tensorflow' in x]

    matriz_columnas = [columns, col_sklearn, col_lightgbm, col_xgboost, col_pytorch, col_tensorflow]
    nombres_columnas_totalizadoras = ["TODOS", "SKLEARN", "LIGHTGBM", "XGBOOST", "PYTORCH", "TENSORFLOW"]
    for i, lista_columnas in enumerate(matriz_columnas):
        if len(lista_columnas) > 1:
            df = df.with_columns(
                pl.sum_horizontal([pl.col(c) for c in lista_columnas])
                  .map_elements(binariza, return_dtype=pl.Int8)
                  .alias(nombres_columnas_totalizadoras[i])
            )

    # df = df.with_columns([
    #     pl.sum_horizontal([pl.col(c) for c in columns]).map_elements(binariza, return_dtype=pl.Int8).alias("TODOS"),
    #     pl.sum_horizontal([pl.col(c) for c in col_sklearn]).map_elements(binariza, return_dtype=pl.Int8).alias("SKLEARN"),
    #     pl.sum_horizontal([pl.col(c) for c in col_lightgbm]).map_elements(binariza, return_dtype=pl.Int8).alias("LIGHTGBM"),
    #     pl.sum_horizontal([pl.col(c) for c in col_xgboost]).map_elements(binariza, return_dtype=pl.Int8).alias("XGBOOST"),
    #     pl.sum_horizontal([pl.col(c) for c in col_pytorch]).map_elements(binariza, return_dtype=pl.Int8).alias("PYTORCH"),
    #     pl.sum_horizontal([pl.col(c) for c in col_tensorflow]).map_elements(binariza, return_dtype=pl.Int8).alias("TENSORFLOW"),
    # ])
    return df


def inferencias(matriz_modelos: list, X: np.array, y: np.array, df: pl.DataFrame, parametros: dict):
    # Determinando el device para Pytorch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df_dates = df.select(['date', 'close', 'target_direction']).sort('date')
    df_dates_pond = df_dates.clone()

    librerias = ['sklearn', 'lightgbm', 'xgboost', 'pytorch', 'tensorflow']
    predicciones = [[], [], [], [], []]  # En la Primera lista meto los Arreglos Numpy de predicciones de los modelos Sklearn y asi sucesivamente
    predicciones_pond = [[], [], [], [], []]

    # Este ciclo crea las Predicciones de los 50 modelos (10 modelos por cada libreria)
    for i, list_modelos in enumerate(matriz_modelos):
        for modelo in list_modelos:
            if i != 3:
                if i == 1:
                    pred_proba = modelo.predict_proba(X)
                pred = modelo.predict(X)
                if i == 4:  # Si el modelo es Tensorflow hay que hacer un reshape
                    pred = pred.reshape((1, pred.shape[0])).flatten()

                # Se binariza la prediccion y se adjunta en la lista de predicciones
                pred_binarizada = binariza_lineal(arr=pred, libreria=librerias[i])
                predicciones[i].append(pred_binarizada)

                # Se binariza la prediccion de manera ponderada y se adjunta en la lista de predicciones ponderadas
                pred_binarizada_ponderada = binariza_ponderada(arr=pred, libreria=librerias[i], parametros=parametros) if i != 1 else \
                    binariza_ponderada(arr=pred_proba, libreria=librerias[i], parametros=parametros)
                predicciones_pond[i].append(pred_binarizada_ponderada)
                continue

            # Si no entró en el if anterior se trata de un modelo de Pytorch
            modelo.eval()
            X_pytorch = torch.tensor(X, dtype=torch.float32)
            with torch.no_grad():
                pred_pytorch_t = modelo(X_pytorch.to(device))

            pred = pred_pytorch_t.cpu().detach().numpy().flatten()
            # Se binariza la prediccion y se adjunta en la lista de predicciones
            pred_binarizada = binariza_lineal(arr=pred, libreria=librerias[i])
            predicciones[i].append(pred_binarizada)
            # Se binariza la prediccion de manera ponderada y se adjunta en la lista de predicciones ponderadas
            pred_binarizada_ponderada = binariza_ponderada(arr=pred, libreria=librerias[i], parametros=parametros)
            predicciones_pond[i].append(pred_binarizada_ponderada)

    for i, lista_predicciones in enumerate(predicciones):
        for j, predicciones_modelo in enumerate(lista_predicciones):
            nombre = librerias[i] + '_' + str(j + 1) if j + 1 >= 10 else librerias[i] + '_0' + str(j + 1)
            df_dates = df_dates.sort('date').with_columns(pl.Series(name=nombre, values=predicciones_modelo))
            df_dates_pond = df_dates_pond.sort('date').with_columns(pl.Series(name=nombre, values=predicciones_pond[i][j]))
    return votaciones(df=df_dates), votaciones(df=df_dates_pond)


def totaliza(df: pl.DataFrame) -> pl.DataFrame:
    columns = [x for x in df.columns if x not in ['date', 'close', 'target_direction']]
    columns_aciertos = []
    columns_aciertos_sub = []
    columns_aciertos_baj = []
    columns_pred_sub = []
    columns_pred_baj = []
    for column in columns:
        columns_aciertos.append(f'Acierto-{column}')
        columns_pred_sub.append(f'Pred_Sub-{column}')
        columns_pred_baj.append(f'Pred_Baj-{column}')
        columns_aciertos_sub.append(f'Acierto_Sub-{column}')
        columns_aciertos_baj.append(f'Acierto_Baj-{column}')
        df = df.with_columns([
            (pl.col(column) == 1).cast(pl.Int8).alias(f'Pred_Sub-{column}'),
            ((pl.col('target_direction') == pl.col(column)) & (pl.col(column) == 1)).cast(pl.Int8).alias(f'Acierto_Sub-{column}'),
            (pl.col(column) == -1).cast(pl.Int8).alias(f'Pred_Baj-{column}'),
            ((pl.col('target_direction') == pl.col(column)) & (pl.col(column) == -1)).cast(pl.Int8).alias(f'Acierto_Baj-{column}'),
            (pl.col('target_direction') == pl.col(column)).cast(pl.Int8).alias(f'Acierto-{column}'),
        ])

    dias_bajada = df.filter(df['target_direction'] == 1).shape[0]
    dias_subida = df.filter(df['target_direction'] == 1).shape[0]
    data = []
    for i, column in enumerate(columns):
        numero_predicciones = df[columns_pred_sub[i]].sum() + df[columns_pred_baj[i]].sum()
        diccionario = {}
        diccionario["Modelo"] = column
        diccionario["Directional-Accuracy"] = round((df[columns_aciertos[i]].sum() / numero_predicciones), 4) if numero_predicciones != 0 else 0
        diccionario["Dias de Operacion"] = df.shape[0]
        diccionario['Dias Operados'] = numero_predicciones
        diccionario['Dias Subida'] = dias_subida
        diccionario["Dias-Predijo-Sub"] = df[columns_pred_sub[i]].sum()
        diccionario["Acierto-Sub"] = df[columns_aciertos_sub[i]].sum()
        diccionario["Precision-Sub"] = round((df[columns_aciertos_sub[i]].sum() / df[columns_pred_sub[i]].sum()), 4) if df[columns_pred_sub[i]].sum() != 0 else 0
        diccionario['Dias Bajada'] = dias_bajada
        diccionario["Dias-Predijo-Baj"] = df[columns_pred_baj[i]].sum()
        diccionario["Acierto-Baj"] = df[columns_aciertos_baj[i]].sum()
        diccionario["Precision-Baj"] = round((df[columns_aciertos_baj[i]].sum() / df[columns_pred_baj[i]].sum()), 4) if df[columns_pred_baj[i]].sum() != 0 else 0
        data.append(diccionario)
    return pl.DataFrame(data)


def define_acierto(row: dict) -> int:
    target_direction = row["target_direction"]
    for key in row.keys():
        if "01" in key:
            model_direction = row[key]
            continue
        if "estrategia" in key:
            model_strategy = row[key]

    if model_direction == 0:
        return 0

    if model_strategy == "SUBE" and model_direction == -1:
        return 0

    if model_strategy == "BAJA" and model_direction == 1:
        return 0

    if model_direction == target_direction:
        return 1
    else:
        return 0


def define_modelos_sube_baja_ambos(df_votos: pl.DataFrame,
                                   elegido: str,
                                   librerias: list,
                                   estrategias: dict,
                                   indice_estrategias: dict) -> tuple[list, list, list]:
    numero_modelos_librerias = []

    for libreria in librerias:
        numero_modelos_librerias.append(len([x for x in df_votos.columns if libreria in x]))

    modelos_sube = []
    modelos_baja = []
    modelos_ambos = []
    # indice = instrumentos[elegido]

    for i, libreria in enumerate(librerias):
        if numero_modelos_librerias[i] == 0:
            continue

        if estrategias[indice_estrategias[elegido]][libreria] == "SUBE":
            if numero_modelos_librerias[i] > 1:
                modelos_sube.extend([libreria + "_01", libreria.upper()])
            else:
                modelos_sube.extend([libreria + "_01"])
        elif estrategias[indice_estrategias[elegido]][libreria] == "AMBOS":
            if numero_modelos_librerias[i] > 1:
                modelos_ambos.extend([libreria + "_01", libreria.upper()])
            else:
                modelos_ambos.extend([libreria + "_01"])
        else:
            if numero_modelos_librerias[i] > 1:
                modelos_baja.extend([libreria + "_01", libreria.upper()])
            else:
                modelos_baja.extend([libreria + "_01"])

    return (modelos_sube, modelos_baja, modelos_ambos)


def sumariza_data(df: pl.DataFrame,
                  df_votos_pond: pl.DataFrame,
                  df_votos: pl.DataFrame,
                  elegido: str,
                  librerias: list,
                  estrategias: dict,
                  indice_estrategias: dict) -> tuple[pl.DataFrame, list]:
    modelos_sube, modelos_baja, modelos_ambos = define_modelos_sube_baja_ambos(df_votos=df_votos, elegido=elegido,
                                                                               librerias=librerias, estrategias=estrategias,
                                                                               indice_estrategias=indice_estrategias)

    df = df.select(["date", "close_tomorrow"])
    df_votos_pond = df_votos_pond.join(df, on="date", how="left", coalesce=True)

    columns = ["date", "close", "close_tomorrow", "target_direction"]
    models_columns = [x for x in df_votos_pond.columns if "01" in x]

    for model_column in models_columns:
        columns.append(model_column)
        columna_estrategia = model_column.replace("01", "estrategia")
        columna_acierto = model_column.replace("01", "acierto")
        columna_opero = model_column.replace("01", "opero")
        columns.append(columna_estrategia)
        columns.append(columna_opero)
        columns.append(columna_acierto)
        if model_column in modelos_sube:
            valor_columna_estrategia = "SUBE"
        elif model_column in modelos_ambos:
            valor_columna_estrategia = "AMBOS"
        else:
            valor_columna_estrategia = "BAJA"
        df_votos_pond = df_votos_pond.with_columns(pl.lit(valor_columna_estrategia).alias(columna_estrategia))

        df_votos_pond = df_votos_pond.with_columns(
            pl.when(((pl.col(columna_estrategia) == "SUBE") & (pl.col(model_column) == 1))
                    | ((pl.col(columna_estrategia) == "BAJA") & (pl.col(model_column) == -1))
                    | ((pl.col(columna_estrategia) == "AMBOS") & (pl.col(model_column) != 0))
                    )
            .then(1)
            .otherwise(0)
            .alias(columna_opero)
        )

        df_votos_pond = df_votos_pond.with_columns(
            pl.struct(["target_direction", model_column, columna_estrategia])
            .map_elements(define_acierto, return_dtype=pl.Int8)
            .alias(columna_acierto)
        )

    df_votos_pond = df_votos_pond.select(columns)
    return (df_votos_pond, models_columns)
