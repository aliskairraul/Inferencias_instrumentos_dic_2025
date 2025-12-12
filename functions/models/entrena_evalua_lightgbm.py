from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from functions.other_functions import precision_direcciones, binariza_lineal, binariza_ponderada


def entrena_evalua_lightgbm(parametros_ponderacion: dict,
                            parametros_modelo: dict,
                            X_train: np.array,
                            y_train: np.array,
                            X_test: np.array,
                            y_test: np.array,
                            X_prueba: np.array,
                            y_prueba: np.array,
                            iteracion: int) -> tuple:

    modelo = LGBMClassifier(random_state=42, **parametros_modelo)
    modelo.fit(X_train, y_train)

    pred_test_lineal = modelo.predict(X_test)
    pred_prueba_lineal = modelo.predict(X_prueba)

    pred_test_proba = modelo.predict_proba(X_test)
    pred_prueba_proba = modelo.predict_proba(X_prueba)

    pred_test_pond = binariza_ponderada(arr=pred_test_proba, libreria="lightgbm", parametros=parametros_ponderacion)
    pred_prueba_pond = binariza_ponderada(arr=pred_prueba_proba, libreria="lightgbm", parametros=parametros_ponderacion)

    evalua_test = precision_direcciones(y=y_test, pred=pred_test_lineal, iteracion=iteracion)
    evalua_prueba = precision_direcciones(y=y_prueba, pred=pred_prueba_lineal, iteracion=iteracion)

    evalua_test_pond = precision_direcciones(y=y_test, pred=pred_test_pond, iteracion=iteracion)
    evalua_prueba_pond = precision_direcciones(y=y_prueba, pred=pred_prueba_pond, iteracion=iteracion)

    parametros_entrenamiento = {
        "estimators": parametros_modelo["n_estimators"],
        "depth": parametros_modelo["max_depth"],
        "learning_rate": parametros_modelo["learning_rate"],
        "colsample": parametros_modelo["colsample_bytree"],
        "subsample": parametros_modelo["subsample"]
    }

    return (modelo, evalua_prueba | parametros_entrenamiento, evalua_test, evalua_prueba_pond, evalua_test_pond)
