from tensorflow.keras.models import load_model
import torch
import os
from pathlib import Path
import json
import joblib
from functions.models.entrena_evalua_pytorch import InstrumentosFinancierosDirectionNet


def cargar_modelos(instrumento: str) -> list[list, list, list, list, list]:
    modelos_sklearn = []
    modelos_xgboost = []
    modelos_lightgbm = []
    modelos_tensorflow = []
    modelos_pytorch = []

    carpeta = Path("models/") / instrumento / "general"
    archivos_json = [f for f in os.listdir(carpeta) if f.endswith('.json')]
    archivos_pth = [f for f in os.listdir(carpeta) if f.endswith('.pth')]
    archivos_keras = [f for f in os.listdir(carpeta) if f.endswith('.keras')]
    archivos_pkl = [f for f in os.listdir(carpeta) if f.endswith('.pkl')]

    # CARGANDO LOS MODELOS PYTORCH
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for i, archivo in enumerate(archivos_json):
        # Recupero la Configuracion
        ruta_instancia_dict = carpeta / archivo
        with open(ruta_instancia_dict, "r") as f:
            instancia_dict = json.load(f)

        # Recreo el modelo con la Class, el device, La Configuracion de la Instancia y el State_dict
        ruta_state_dict_pytorch = carpeta / archivos_pth[i]
        model_pytorch = InstrumentosFinancierosDirectionNet(**instancia_dict).to(device)
        model_pytorch.load_state_dict(torch.load(ruta_state_dict_pytorch))

        modelos_pytorch.append(model_pytorch)

    # CARGANDO LOS MODELOS SKLEARN, XGBOOST y LIGHTGBM
    for archivo in archivos_pkl:
        ruta_modelo = carpeta / archivo
        modelo = joblib.load(ruta_modelo)
        if archivo.startswith("sklearn"):
            modelos_sklearn.append(modelo)
            continue
        if archivo.startswith("xgboost"):
            modelos_xgboost.append(modelo)
            continue
        if archivo.startswith("lightgbm"):
            modelos_lightgbm.append(modelo)

    # CARGANDO LOS MODELOS TENSORFLOW
    for archivo in archivos_keras:
        ruta_modelo = carpeta / archivo
        modelo_tensorflow = load_model(ruta_modelo)
        modelos_tensorflow.append(modelo_tensorflow)

    matriz_modelos = [modelos_sklearn, modelos_lightgbm, modelos_xgboost, modelos_pytorch, modelos_tensorflow]

    return matriz_modelos
