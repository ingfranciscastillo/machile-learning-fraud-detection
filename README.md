# 🔍 Detector de Fraudes en Transacciones

Un sistema completo de Machine Learning para detectar fraudes en transacciones de tarjetas de crédito usando múltiples algoritmos y una API REST con FastAPI.

## 🌟 Características

- **Múltiples Modelos**: Entrena y compara Regresión Logística, Random Forest y Redes Neuronales
- **Manejo de Desbalance**: Utiliza SMOTE para balancear las clases
- **Evaluación Completa**: Métricas detalladas (Precision, Recall, F1-Score, Matriz de Confusión)
- **API REST**: Interfaz FastAPI para predicciones en tiempo real
- **Código Modular**: Arquitectura limpia y bien documentada

## 📋 Requisitos

- Python
- Dataset "Credit Card Fraud Detection" de Kaggle (opcional, se genera uno sintético si no está disponible)

## 🚀 Instalación

1. **Clonar o descargar el proyecto**

2. **Instalar dependencias**:

```bash
pip install -r requirements.txt
```

3. **Descargar dataset (opcional)**:
   - Descargar el dataset de [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
   - Colocar el archivo `creditcard.csv` en la raíz del proyecto
   - Si no tienes el dataset, el sistema generará uno sintético automáticamente

## 💻 Uso

### 1. Entrenar el Modelo

```bash
python train_model.py
```

Este script:

- Carga el dataset (real o sintético)
- Preprocesa los datos y aplica SMOTE
- Entrena múltiples modelos
- Evalúa y selecciona el mejor modelo
- Guarda el modelo entrenado en la carpeta `models/`

### 2. Iniciar la API

```bash
python api.py
```

La API estará disponible en:

- **Servidor**: http://localhost:8000
- **Documentación**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

### 3. Probar la API

```bash
python test_api.py
```

## 🔧 Endpoints de la API

### `GET /health`

Verifica el estado de la API y si el modelo está cargado.

**Respuesta**:

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_name": "random_forest",
  "timestamp": "2024-01-15T10:30:00"
}
```

### `POST /predict`

Predice si una transacción es fraudulenta.

**Request**:

```json
{
  "V1": -1.359807,
  "V2": -0.072781,
  // ... resto de características V3-V28
  "Amount": 149.62
}
```

**Respuesta**:

```json
{
  "prediction": 0,
  "probability": 0.0234,
  "risk_level": "BAJO",
  "timestamp": "2024-01-15T10:30:00"
}
```

### `GET /model-info`

Obtiene información del modelo cargado y sus métricas.

## 📊 Estructura del Proyecto

```
fraud-detection/
├── train_model.py          # Script de entrenamiento
├── api.py                  # API FastAPI
├── test_api.py            # Script de pruebas
├── requirements.txt       # Dependencias
├── README.md             # Documentación
├── models/               # Modelos entrenados
│   ├── best_fraud_model.pkl
│   ├── scaler.pkl
│   └── model_info.pkl
└── creditcard.csv        # Dataset (opcional)
```

## 🧠 Modelos Implementados

### 1. Regresión Logística

- **Ventajas**: Rápido, interpretable
- **Uso**: Baseline y casos donde la interpretabilidad es importante

### 2. Random Forest

- **Ventajas**: Robusto, maneja bien características no lineales
- **Uso**: Buen balance entre rendimiento y velocidad

### 3. Red Neuronal (MLP)

- **Ventajas**: Puede capturar patrones complejos
- **Uso**: Cuando hay suficientes datos y se necesita máxima precisión

## 📈 Métricas de Evaluación

- **Accuracy**: Precisión general
- **Precision**: Proporción de fraudes detectados correctamente
- **Recall**: Proporción de fraudes reales detectados
- **F1-Score**: Media armónica de Precision y Recall
- **Matriz de Confusión**: Distribución detallada de predicciones

## 🛠️ Preprocesamiento

1. **Normalización**: Estandarización de todas las características numéricas
2. **SMOTE**: Sobremuestreo sintético para balancear clases
3. **División**: 80% entrenamiento, 20% prueba
4. **Validación**: Estratificada para mantener proporción de clases

## 🔒 Niveles de Riesgo

La API clasifica las transacciones en tres niveles:

- **BAJO** (< 30%): Transacción segura
- **MEDIO** (30-70%): Requiere revisión
- **ALTO** (> 70%): Probable fraude

## 📝 Ejemplos de Uso

### Ejemplo 1: Transacción Normal

```python
import requests

transaction = {
    "V1": -1.359807, "V2": -0.072781, "V3": 2.536347,
    # ... resto de características
    "Amount": 149.62
}

response = requests.post("http://localhost:8000/predict", json=transaction)
result = response.json()
# Resultado esperado: prediction=0, probability=0.02, risk_level="BAJO"
```

### Ejemplo 2: Transacción Sospechosa

```python
suspicious_transaction = {
    "V1": 2.5, "V2": 3.2, "V3": -1.8,
    # ... valores alterados para simular fraude
    "Amount": 2500.0
}

response = requests.post("http://localhost:8000/predict", json=suspicious_transaction)
result = response.json()
# Resultado esperado: prediction=1, probability=0.85, risk_level="ALTO"
```

## 🐛 Solución de Problemas

### Error: "Modelo no disponible"

1. Verificar que existe la carpeta `models/`
2. Ejecutar `python train_model.py` para entrenar el modelo
3. Verificar que se generaron los archivos `.pkl`

### Error: "Dataset no encontrado"

1. El sistema generará un dataset sintético automáticamente
2. Para usar el dataset real, descargarlo de Kaggle

### Error de dependencias

```bash
pip install --upgrade -r requirements.txt
```

## 🚀 Despliegue en Producción

### Con Docker (recomendado)

```dockerfile
FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    gcc \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p models

EXPOSE 8000

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Comando para entrenar modelo (opcional, comentar si ya tienes modelo)
# RUN python train_model.py

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]

```

### Con Uvicorn

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --workers 4
```

## 📊 Monitoreo y Logging

La API incluye:

- Logging detallado de todas las predicciones
- Timestamps para auditoría
- Métricas de rendimiento del modelo
- Health checks automáticos

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Ver `LICENSE` para más detalles.
