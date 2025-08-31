"""
API FastAPI para el detector de fraudes en transacciones.
Expone endpoints para predicciones y health check.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
import joblib
import numpy as np
import os
import logging
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variables globales para el modelo y scaler
model = None
scaler = None
model_info = None

def load_models():
    """Carga el modelo entrenado y el scaler."""
    global model, scaler, model_info
    
    try:
        model_dir = 'models'
        
        # Cargar modelo
        model_path = os.path.join(model_dir, 'best_fraud_model.pkl')
        if not os.path.exists(model_path):
            logger.error(f"No se encontró el modelo en: {model_path}")
            return False
        
        model = joblib.load(model_path)
        logger.info("Modelo cargado exitosamente")
        
        # Cargar scaler
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        if not os.path.exists(scaler_path):
            logger.error(f"No se encontró el scaler en: {scaler_path}")
            return False
        
        scaler = joblib.load(scaler_path)
        logger.info("Scaler cargado exitosamente")
        
        # Cargar información del modelo
        info_path = os.path.join(model_dir, 'model_info.pkl')
        if os.path.exists(info_path):
            model_info = joblib.load(info_path)
            logger.info(f"Información del modelo cargada: {model_info['model_name']}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error cargando modelos: {str(e)}")
        return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manejador de eventos de lifespan para FastAPI."""
    # Startup
    logger.info("Iniciando Fraud Detection API...")
    if not load_models():
        logger.error("Error al cargar los modelos. La API puede no funcionar correctamente.")
    else:
        logger.info("API iniciada exitosamente")
    
    yield
    
    # Shutdown
    logger.info("Cerrando Fraud Detection API...")

# Inicializar FastAPI con lifespan
app = FastAPI(
    title="Fraud Detection API",
    description="API para detectar fraudes en transacciones usando Machine Learning",
    version="1.0.0",
    lifespan=lifespan
)

class TransactionData(BaseModel):
    """Modelo Pydantic para validar los datos de entrada de una transacción."""
    
    V1: float = Field(..., description="Característica V1 de la transacción")
    V2: float = Field(..., description="Característica V2 de la transacción")
    V3: float = Field(..., description="Característica V3 de la transacción")
    V4: float = Field(..., description="Característica V4 de la transacción")
    V5: float = Field(..., description="Característica V5 de la transacción")
    V6: float = Field(..., description="Característica V6 de la transacción")
    V7: float = Field(..., description="Característica V7 de la transacción")
    V8: float = Field(..., description="Característica V8 de la transacción")
    V9: float = Field(..., description="Característica V9 de la transacción")
    V10: float = Field(..., description="Característica V10 de la transacción")
    V11: float = Field(..., description="Característica V11 de la transacción")
    V12: float = Field(..., description="Característica V12 de la transacción")
    V13: float = Field(..., description="Característica V13 de la transacción")
    V14: float = Field(..., description="Característica V14 de la transacción")
    V15: float = Field(..., description="Característica V15 de la transacción")
    V16: float = Field(..., description="Característica V16 de la transacción")
    V17: float = Field(..., description="Característica V17 de la transacción")
    V18: float = Field(..., description="Característica V18 de la transacción")
    V19: float = Field(..., description="Característica V19 de la transacción")
    V20: float = Field(..., description="Característica V20 de la transacción")
    V21: float = Field(..., description="Característica V21 de la transacción")
    V22: float = Field(..., description="Característica V22 de la transacción")
    V23: float = Field(..., description="Característica V23 de la transacción")
    V24: float = Field(..., description="Característica V24 de la transacción")
    V25: float = Field(..., description="Característica V25 de la transacción")
    V26: float = Field(..., description="Característica V26 de la transacción")
    V27: float = Field(..., description="Característica V27 de la transacción")
    V28: float = Field(..., description="Característica V28 de la transacción")
    Amount: float = Field(..., description="Monto de la transacción", ge=0)
    
    model_config = {
        "json_schema_extra": {
            "example": {
                "V1": -1.359807,
                "V2": -0.072781,
                "V3": 2.536347,
                "V4": 1.378155,
                "V5": -0.338321,
                "V6": 0.462388,
                "V7": 0.239599,
                "V8": 0.098698,
                "V9": 0.363787,
                "V10": 0.090794,
                "V11": -0.551600,
                "V12": -0.617801,
                "V13": -0.991390,
                "V14": -0.311169,
                "V15": 1.468177,
                "V16": -0.470401,
                "V17": 0.207971,
                "V18": 0.025791,
                "V19": 0.403993,
                "V20": 0.251412,
                "V21": -0.018307,
                "V22": 0.277838,
                "V23": -0.110474,
                "V24": 0.066928,
                "V25": 0.128539,
                "V26": -0.189115,
                "V27": 0.133558,
                "V28": -0.021053,
                "Amount": 149.62
            }
        }
    }

class PredictionResponse(BaseModel):
    """Modelo Pydantic para la respuesta de predicción."""
    
    prediction: int = Field(..., description="Predicción: 0 (No Fraude) o 1 (Fraude)")
    probability: float = Field(..., description="Probabilidad de que sea fraude")
    risk_level: str = Field(..., description="Nivel de riesgo: BAJO, MEDIO, ALTO")
    timestamp: str = Field(..., description="Timestamp de la predicción")

class HealthResponse(BaseModel):
    """Modelo Pydantic para la respuesta de health check."""
    
    status: str = Field(..., description="Estado de la API")
    model_loaded: bool = Field(..., description="Si el modelo está cargado")
    model_name: Optional[str] = Field(None, description="Nombre del modelo")
    timestamp: str = Field(..., description="Timestamp del health check")

def get_risk_level(probability: float) -> str:
    """
    Determina el nivel de riesgo basado en la probabilidad de fraude.
    
    Args:
        probability (float): Probabilidad de fraude (0-1)
        
    Returns:
        str: Nivel de riesgo (BAJO, MEDIO, ALTO)
    """
    if probability < 0.3:
        return "BAJO"
    elif probability < 0.7:
        return "MEDIO"
    else:
        return "ALTO"

@app.get("/", response_model=dict)
async def root():
    """Endpoint raíz con información básica de la API."""
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "description": "API para detectar fraudes en transacciones",
        "endpoints": {
            "/predict": "POST - Realizar predicción de fraude",
            "/health": "GET - Health check de la API",
            "/docs": "GET - Documentación interactiva"
        }
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Endpoint para verificar el estado de la API.
    
    Returns:
        HealthResponse: Estado actual de la API
    """
    model_loaded = model is not None and scaler is not None
    model_name = model_info.get('model_name') if model_info else None
    
    return HealthResponse(
        status="healthy" if model_loaded else "unhealthy",
        model_loaded=model_loaded,
        model_name=model_name,
        timestamp=datetime.now().isoformat()
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(transaction: TransactionData):
    """
    Endpoint para predecir si una transacción es fraudulenta.
    
    Args:
        transaction (TransactionData): Datos de la transacción
        
    Returns:
        PredictionResponse: Resultado de la predicción
        
    Raises:
        HTTPException: Si hay un error en la predicción
    """
    if model is None or scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Modelo no disponible. Verificar que los archivos del modelo existan."
        )
    
    try:
        # Convertir los datos de entrada a array numpy
        feature_names = [
            'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
            'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount'
        ]
        
        # Extraer valores en el orden correcto
        features = np.array([[getattr(transaction, feature) for feature in feature_names]])
        
        # Normalizar características
        features_scaled = scaler.transform(features)
        
        # Realizar predicción
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]  # Probabilidad de fraude
        
        # Determinar nivel de riesgo
        risk_level = get_risk_level(probability)
        
        logger.info(f"Predicción realizada: {prediction}, Probabilidad: {probability:.4f}")
        
        return PredictionResponse(
            prediction=int(prediction),
            probability=round(float(probability), 4),
            risk_level=risk_level,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error interno en la predicción: {str(e)}"
        )

@app.get("/model-info", response_model=dict)
async def get_model_info():
    """
    Endpoint para obtener información del modelo cargado.
    
    Returns:
        dict: Información del modelo y sus métricas
    """
    if model_info is None:
        raise HTTPException(
            status_code=404,
            detail="Información del modelo no disponible"
        )
    
    return {
        "model_name": model_info.get('model_name'),
        "model_scores": model_info.get('model_scores'),
        "model_type": str(type(model)).split('.')[-1].replace("'>", ""),
        "scaler_type": str(type(scaler)).split('.')[-1].replace("'>", ""),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    
    print("Iniciando servidor FastAPI...")
    print("Documentación disponible en: http://localhost:8000/docs")
    print("Health check en: http://localhost:8000/health")
    
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )