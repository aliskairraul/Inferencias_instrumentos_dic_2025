import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import accuracy_score, confusion_matrix
from functions.other_functions import precision_direcciones, binariza_lineal, binariza_ponderada
import warnings

warnings.filterwarnings("ignore")


def entrena_evalua_histgb(parametros_ponderacion: dict,
                          max_iter: int,
                          max_leaf_nodes: int,
                          learning_rate: float,
                          l2_regularization: float,
                          X_train: np.array,
                          y_train: np.array,
                          X_test: np.array,
                          y_test: np.array,
                          X_prueba: np.array,
                          y_prueba: np.array,
                          iteracion: int = 0) -> tuple:

    param_grid = {
        "max_iter": [max_iter],
        "max_leaf_nodes": [max_leaf_nodes],
        "learning_rate": [learning_rate],
        "l2_regularization": [l2_regularization]
    }

    model = HistGradientBoostingRegressor(random_state=42)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="neg_mean_absolute_error",
        verbose=0,
        n_jobs=-1
    )

    grid_search.fit(X_train, y_train)
    modelo = grid_search.best_estimator_

    pred_test = modelo.predict(X_test)
    pred_prueba = modelo.predict(X_prueba)

    pred_test_lineal = binariza_lineal(arr=pred_test, libreria="sklearn")
    pred_prueba_lineal = binariza_lineal(arr=pred_prueba, libreria="sklearn")

    pred_test_pond = binariza_ponderada(arr=pred_test, libreria="sklearn", parametros=parametros_ponderacion)
    pred_prueba_pond = binariza_ponderada(arr=pred_prueba, libreria="sklearn", parametros=parametros_ponderacion)

    evalua_test = precision_direcciones(y=y_test, pred=pred_test_lineal, iteracion=iteracion)
    evalua_prueba = precision_direcciones(y=y_prueba, pred=pred_prueba_lineal, iteracion=iteracion)

    evalua_test_pond = precision_direcciones(y=y_test, pred=pred_test_pond, iteracion=iteracion)
    evalua_prueba_pond = precision_direcciones(y=y_prueba, pred=pred_prueba_pond, iteracion=iteracion)

    parametros_entrenamiento = {
        "max_iter": max_iter,
        "max_leaf_nodes": max_leaf_nodes,
        "learning_rate": learning_rate,
        "l2_regularization": l2_regularization
    }

    return modelo, evalua_prueba | parametros_entrenamiento, evalua_test, evalua_prueba_pond, evalua_test_pond
