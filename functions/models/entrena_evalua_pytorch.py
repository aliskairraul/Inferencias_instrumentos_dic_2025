import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from functions.other_functions import precision_direcciones, binariza_lineal, binariza_ponderada


class InstrumentosFinancierosDirectionNet(nn.Module):
    """Red neuronal optimizada para predecir dirección del precio de los Instrumentos Financieros."""

    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_prob=0.3):
        super(InstrumentosFinancierosDirectionNet, self).__init__()

        layers = []
        prev_dim = input_dim

        # Construcción dinámica de capas
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),  # BatchNorm para mejor estabilidad
                nn.LeakyReLU(0.1),  # LeakyReLU para evitar neuronas muertas
                nn.Dropout(dropout_prob)
            ])
            prev_dim = hidden_dim

        # Capa final de clasificación
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # Sigmoid para probabilidades [0,1]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


def entrena_evalua_pytorch(parametros_ponderacion: dict,
                           hidden_dims: list,
                           learning_rate: float,
                           dropout: float,
                           epochs: int,
                           batch_size: int,
                           train_loader: DataLoader,
                           X_test_t: torch.Tensor,
                           X_prueba_t: torch.Tensor,
                           y_test: np.array,
                           y_prueba: np.array,
                           device: torch.device,
                           iteracion: int) -> tuple:

    input_dim = next(iter(train_loader))[0].shape[1]
    # hidden_dims = [hidden_dim, hidden_dim // 2, hidden_dim // 4]

    modelo = InstrumentosFinancierosDirectionNet(input_dim, hidden_dims, dropout).to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(modelo.parameters(), lr=learning_rate, weight_decay=1e-4)

    for epoch in range(epochs):
        modelo.train()
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            binary_labels = (labels > 0).float().unsqueeze(1)

            outputs = modelo(features)
            # Aca una correccion local       # print("Output range:", outputs.min().item(), outputs.max().item())
            loss = criterion(outputs, binary_labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(modelo.parameters(), max_norm=1.0)
            optimizer.step()
    modelo.eval()

    def evalua(X_pytorch: torch.Tensor, y_numpy: np.array):
        with torch.no_grad():
            pred_pytorch_t = modelo(X_pytorch.to(device))

        pred = pred_pytorch_t.cpu().detach().numpy().flatten()
        pred_binariza_lineal = binariza_lineal(arr=pred, libreria="pytorch")
        pred_binariza_pond = binariza_ponderada(arr=pred, libreria="pytorch", parametros=parametros_ponderacion)

        return (precision_direcciones(y=y_numpy, pred=pred_binariza_lineal, iteracion=iteracion),
                precision_direcciones(y=y_numpy, pred=pred_binariza_pond, iteracion=iteracion))

    evalua_test, evalua_test_pond = evalua(X_test_t, y_test)
    evalua_prueba, evalua_prueba_pond = evalua(X_prueba_t, y_prueba)

    parametros_entrenamiento = {
        'input_dim': input_dim,
        'hidden_dims': hidden_dims,
        'learning_rate': learning_rate,
        'dropout': dropout,
        'epochs': epochs,
        'batch_size': batch_size,
    }

    return modelo, evalua_prueba | parametros_entrenamiento, evalua_test, evalua_prueba_pond, evalua_test_pond
