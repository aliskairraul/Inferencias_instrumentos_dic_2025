# 🚀 Inferencias de Predicción de Mercado con Deep Learning

![Deep Learning](https://img.shields.io/badge/Tech-Deep%20Learning-blueviolet)
![Finance](https://img.shields.io/badge/Field-Financial%20Markets-gold)
![Python](https://img.shields.io/badge/Language-Python%203.x-blue)
![Status](https://img.shields.io/badge/Status-Scaleable-green)

Este proyecto implementa un sistema robusto de **Inferencias para la Predicción de Dirección del Precio** en diversos instrumentos financieros. Utilizando arquitecturas avanzadas de Aprendizaje Profundo (Deep Learning) y modelos de ensamble, el sistema analiza datos históricos y en tiempo real para proporcionar señales de mercado con una capa inteligente de ponderación de riesgos.

## 📈 Instrumentos Financieros Cubiertos

El sistema realiza inferencias precisas sobre los activos más líquidos y representativos del mercado global:

*   **S&P 500 (SPX):** El principal índice bursátil que agrupa las 500 empresas más grandes de EE.UU.
*   **EUR/USD:** El par de divisas (Forex) con mayor volumen de negociación a nivel mundial.
*   **BTC/USD:** La criptomoneda líder (Bitcoin) frente al dólar estadounidense.
*   **XAU/USD:** El valor del Oro por onza, el activo refugio por excelencia.

---

## 🛠️ Stack Tecnológico y Modelos

Hemos seleccionado cuidadosamente las librerías más potentes para garantizar precisión y escalabilidad:

*   **Deep Learning Frameworks**: 
    *   **PyTorch**: Implementación de redes neuronales de clasificación binaria personalizadas. `Class con clasificación binaria`.   
    *   **TensorFlow/Keras**: Modelos secuenciales optimizados. `keras.Sequential`.
*   **Machine Learning & Ensembles**:
    *   **XGBoost**: Regresión y clasificación de alto rendimiento `XGBRegressor`.
    *   **LightGBM**: Gradiente descentrado rápido y eficiente `LGBMRegressor`.
    *   **Sklearn**: Regresión basado en Gradient Boosting con histogramas `HistGradientBoostingRegressor`.
    *   **Sklearn**: Optimización de hiperparámetros mediante `GridSearchCV`.

> [!NOTE]
> Se realizaron pruebas con arquitecturas Convolucionales (CNN) y Recurrentes (RNN); sin embargo, para la naturaleza de estos datos específicos, los modelos elegidos demostraron una superioridad estadística en los resultados de validación.

---

## ⚙️ Proceso de Inferencia y Pipeline

El flujo de trabajo está diseñado para ser modular y eficiente:

1.  **Preparación de Entorno**: Carga de *scalers* (normalización) y parámetros de ponderación específicos por instrumento.
2.  **Pipeline de Datos**: Ingesta de data inicial, aplicación de Ingeniería de Características (*Feature Engineering*) y segmentación de datos (*Splitting*).
3.  **Ejecución de Modelos**: Proceso de inferencia cruzada utilizando una matriz de modelos entrenados (una combinación única por cada par Librería/Instrumento).
4.  **Consolidación de Resultados**: Interpretación y persistencia de las predicciones en formatos de alto rendimiento como Parquet.

---

## 🧠 Inteligencia de Ponderación (Decision Logic)

Predecir el mercado es un desafío complejo. Por ello, hemos implementado una **Capa de Ponderación** basada en porcentajes de certeza. Esto transforma una predicción binaria tradicional en una decisión estratégica de tres estados:

*   🟢 **SUBE**: Señal de compra con alta probabilidad de acierto.
*   🔴 **BAJA**: Señal de venta con alta probabilidad de acierto.
*   ⚪ **NO OPERAR**: Filtro de seguridad cuando el consenso de los modelos o la certeza no alcanzan el umbral óptimo.

---

## 📂 Estructura del Proyecto

```text
├── .github/              # Configuraciones de GitHub Actions
├── db/                   # Almacenamiento local de datos persistidos
├── functions/            # Funciones core para procesamiento de datos
├── models/               # Modelos entrenados, scalers y parámetros JSON
├── utils/                # Utilidades de logging y herramientas auxiliares
├── main.py               # Punto de entrada principal para inferencias
└── requirements.txt      # Dependencias del proyecto
```

---

## 🚀 Pasos para la Ejecución

Para poner en marcha el sistema de inferencias en su entorno local, siga estos pasos:

### 1. Clonar el repositorio
```bash
git clone https://github.com/aliskairraul/Inferencias_instrumentos_dic_2025.git
cd Inferencias_instrumentos_dic_2025
```

### 2. Configurar el entorno virtual
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 4. Ejecutar el sistema de inferencias
```bash
python main.py
```

*El sistema se encargará automáticamente de descargar la última data disponible desde el repositorio de actualización de datos.*

---

## 📊 Fuentes de Datos y Estructura

La data se sincroniza automáticamente desde el repositorio [Actualiza-Data-Instrumentos](https://github.com/aliskairraul/Actualiza-Data-Instrumentos), asegurando que las inferencias se realicen siempre sobre el histórico más reciente.

### Enlaces Directos a los Datasets:
*   **SPX**: [Parquet File](https://raw.githubusercontent.com/aliskairraul/Actualiza-Data-Instrumentos/main/db/sp500-D1_2000-01-03_actualidad.parquet)
*   **EURUSD**: [Parquet File](https://raw.githubusercontent.com/aliskairraul/Actualiza-Data-Instrumentos/main/db/eurusd-D1_2000-01-03_actualidad.parquet)
*   **BTCUSD**: [Parquet File](https://raw.githubusercontent.com/aliskairraul/Actualiza-Data-Instrumentos/main/db/btcusd-D1_2010-07-17_actualidad.parquet)
*   **XAUUSD**: [Parquet File](https://raw.githubusercontent.com/aliskairraul/Actualiza-Data-Instrumentos/main/db/xauusd-D1_2000-01-03_actualidad.parquet)

### Estructura de la Data:
Los archivos contienen las siguientes columnas técnicas:
*   `date`: Fecha de operación (`datetime.date`).
*   `open`: Precio de apertura del día (`Float`).
*   `high`: Precio máximo alcanzado (`Float`).
*   `low`: Precio mínimo alcanzado (`Float`).
*   `close`: Precio de cierre final (`Float`).
*   `symbol`: Símbolo identificador del instrumento (`String`).

---

## 🤝 Contacto y Portafolio

¡Conectemos! Estoy abierto a colaboraciones y discusiones sobre IA aplicada a Finanzas.

*   **LinkedIn**: [Aliskair Rodriguez](https://www.linkedin.com/in/aliskair-rodriguez-782b3641/)
*   **Email**: [aliskairraul@gmail.com](mailto:aliskairraul@gmail.com)
*   **Web/Portfolio**: [aliskairraul.github.io](https://aliskairraul.github.io)

---
*Desarrollado con ❤️ para el mundo del Trading Algorítmico.*
