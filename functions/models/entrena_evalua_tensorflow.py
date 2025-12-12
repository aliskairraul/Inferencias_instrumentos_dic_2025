import tensorflow as tf
import numpy as np
from functions.other_functions import precision_direcciones, binariza_lineal, binariza_ponderada


def entrena_evalua_tensorflow(parametros_ponderacion: dict,
                              hidden_dims: list,
                              learning_rate: float,
                              dropout: float,
                              epochs: int,
                              batch_size: int,
                              X_train: np.ndarray,
                              y_train: np.ndarray,
                              X_test: np.ndarray,
                              y_test: np.ndarray,
                              X_prueba: np.ndarray,
                              y_prueba: np.ndarray,
                              iteracion: int = 0) -> tuple:

    # hidden_dims = [hidden_dim, hidden_dim // 2, hidden_dim // 4]
    # hidden_dims = [hidden_dim, hidden_dim, hidden_dim]
    # hidden_dims = [hidden_dim * 2, hidden_dim, hidden_dim // 2]

    y_train_bin = (y_train > 0).astype(np.float32)
    y_test_bin = (y_test > 0).astype(np.float32)

    # Construcción del modelo
    model = tf.keras.Sequential()
    for dim in hidden_dims:
        model.add(tf.keras.layers.Dense(dim))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.LeakyReLU(alpha=0.1))
        model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']  # <-- AÑADE MÉTRICAS
    )

    # Callback de EarlyStopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',         # <-- CAMBIA A 'val_loss'
        patience=10,
        restore_best_weights=True
    )

    model.fit(
        X_train,
        y_train_bin,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        callbacks=[early_stop],
        validation_data=(X_test, y_test_bin)  # <-- AÑADE VALIDATION_DATA
    )

    # Inferencia
    pred_test = model.predict(X_test, batch_size=batch_size, verbose=0).flatten()
    pred_prueba = model.predict(X_prueba, batch_size=batch_size, verbose=0).flatten()

    pred_test_lineal = binariza_lineal(arr=pred_test, libreria="tensorflow")
    pred_prueba_lineal = binariza_lineal(arr=pred_prueba, libreria="tensorflow")

    pred_test_pond = binariza_ponderada(arr=pred_test, libreria="tensorflow", parametros=parametros_ponderacion)
    pred_prueba_pond = binariza_ponderada(arr=pred_prueba, libreria="tensorflow", parametros=parametros_ponderacion)

    evalua_test = precision_direcciones(y_test, pred_test_lineal, iteracion=iteracion)
    evalua_prueba = precision_direcciones(y_prueba, pred_prueba_lineal, iteracion=iteracion)

    evalua_test_pond = precision_direcciones(y_test, pred_test_pond, iteracion=iteracion)
    evalua_prueba_pond = precision_direcciones(y_prueba, pred_prueba_pond, iteracion=iteracion)

    parametros_entrenamiento = {
        "hidden_dims": hidden_dims,
        "learning_rate": learning_rate,
        "dropout": dropout,
        "epochs": epochs,
        "batch_size": batch_size
    }

    return model, evalua_prueba | parametros_entrenamiento, evalua_test, evalua_prueba_pond, evalua_test_pond
