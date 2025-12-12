import xgboost as xgb
import numpy as np
from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import accuracy_score, confusion_matrix
from functions.other_functions import precision_direcciones, binariza_lineal, binariza_ponderada


def entrena_evalua_xgboost(parametros_ponderacion: dict,
                           estimators: int,
                           depth: int,
                           learning_rate: float,
                           subsample: float,
                           colsample: float,
                           X_train: np.array,
                           y_train: np.array,
                           X_test: np.array,
                           y_test: np.array,
                           X_prueba: np.array,
                           y_prueba: np.array,
                           iteracion: int = 0) -> tuple:

    param_grid = {
        "n_estimators": [estimators],
        "max_depth": [depth],
        "learning_rate": [learning_rate],
        "subsample": [subsample],
        "colsample_bytree": [colsample]
    }

    model = xgb.XGBRegressor(random_state=42)

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

    pred_test_lineal = binariza_lineal(arr=pred_test, libreria="xgboost")
    pred_prueba_lineal = binariza_lineal(arr=pred_prueba, libreria="xgboost")

    pred_test_pond = binariza_ponderada(arr=pred_test, libreria="xgboost", parametros=parametros_ponderacion)
    pred_prueba_pond = binariza_ponderada(arr=pred_prueba, libreria="xgboost", parametros=parametros_ponderacion)

    evalua_test = precision_direcciones(y=y_test, pred=pred_test_lineal, iteracion=iteracion)
    evalua_prueba = precision_direcciones(y=y_prueba, pred=pred_prueba_lineal, iteracion=iteracion)

    evalua_test_pond = precision_direcciones(y=y_test, pred=pred_test_pond, iteracion=iteracion)
    evalua_prueba_pond = precision_direcciones(y=y_prueba, pred=pred_prueba_pond, iteracion=iteracion)

    parametros_entrenamiento = {
        "estimators": estimators,
        "depth": depth,
        "learning_rate": learning_rate,
        "colsample": colsample,
        "subsample": subsample
    }

    return (modelo, evalua_prueba | parametros_entrenamiento, evalua_test, evalua_prueba_pond, evalua_test_pond)
