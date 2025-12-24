from tensorflow.keras.models import load_model
import torch
import os
from pathlib import Path
import json
import joblib
from functions.models.entrena_evalua_pytorch import InstrumentosFinancierosDirectionNet


def cargar_modelos(instrumento: str) -> dict:
    modelos = {
        "sklearn": None,
        "lightgbm": None,
        "xgboost": None,
        "pytorch": None,
        "tensorflow": None
    }

    carpeta = Path("models/") / instrumento / "general"
    archivos_json = [f for f in os.listdir(carpeta) if f.endswith('01.json')]
    archivos_pth = [f for f in os.listdir(carpeta) if f.endswith('01.pth')]
    archivos_keras = [f for f in os.listdir(carpeta) if f.endswith('01.keras')]
    archivos_pkl = [f for f in os.listdir(carpeta) if f.endswith('01.pkl')]

    # CARGANDO  MODELOS SKLEARN, XGBOOST y LIGHTGBM
    for archivo in archivos_pkl:
        ruta_modelo = carpeta / archivo
        modelo = joblib.load(ruta_modelo)
        if archivo.startswith("sklearn"):
            modelos["sklearn"] = modelo
            continue
        if archivo.startswith("xgboost"):
            modelos["xgboost"] = modelo
            continue
        if archivo.startswith("lightgbm"):
            modelos["lightgbm"] = modelo

    # CARGANDO MODELO PYTORCH
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
        modelos["pytorch"] = model_pytorch

    # CARGANDO MODELO TENSORFLOW
    for archivo in archivos_keras:
        ruta_modelo = carpeta / archivo
        modelo_tensorflow = load_model(ruta_modelo)
        modelos["tensorflow"] = modelo_tensorflow

    return modelos
