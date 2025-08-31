"""
Script para probar la API de detección de fraudes.
Incluye ejemplos de transacciones normales y fraudulentas.
"""

import requests
import json
import time

# URL base de la API (cambiar si es necesario)
BASE_URL = "http://localhost:8000"

def test_health_endpoint():
    """Prueba el endpoint de health check."""
    print("🏥 Probando endpoint /health...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_predict_endpoint():
    """Prueba el endpoint de predicción con datos de ejemplo."""
    print("\n🔍 Probando endpoint /predict...")
    
    # Ejemplo de transacción normal (valores típicos del dataset)
    normal_transaction = {
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
    
    # Ejemplo de transacción sospechosa (valores alterados para simular fraude)
    suspicious_transaction = {
        "V1": 2.5,
        "V2": 3.2,
        "V3": -1.8,
        "V4": 4.1,
        "V5": -2.3,
        "V6": 1.9,
        "V7": -3.1,
        "V8": 2.8,
        "V9": -1.5,
        "V10": 3.7,
        "V11": -2.9,
        "V12": 1.6,
        "V13": -4.2,
        "V14": 2.1,
        "V15": -3.8,
        "V16": 1.3,
        "V17": -2.7,
        "V18": 3.4,
        "V19": -1.2,
        "V20": 4.5,
        "V21": -2.8,
        "V22": 1.7,
        "V23": -3.3,
        "V24": 2.9,
        "V25": -1.4,
        "V26": 3.6,
        "V27": -2.1,
        "V28": 1.8,
        "Amount": 2500.0
    }
    
    transactions = [
        ("Transacción Normal", normal_transaction),
        ("Transacción Sospechosa", suspicious_transaction)
    ]
    
    for name, transaction in transactions:
        print(f"\n--- {name} ---")
        try:
            response = requests.post(
                f"{BASE_URL}/predict",
                json=transaction,
                headers={"Content-Type": "application/json"}
            )
            
            print(f"Status Code: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Predicción: {'FRAUDE' if result['prediction'] == 1 else 'NO FRAUDE'}")
                print(f"Probabilidad de fraude: {result['probability']:.4f}")
                print(f"Nivel de riesgo: {result['risk_level']}")
                print(f"Timestamp: {result['timestamp']}")
            else:
                print(f"Error: {response.text}")
        except Exception as e:
            print(f"Error: {e}")

def test_model_info_endpoint():
    """Prueba el endpoint de información del modelo."""
    print("\n📊 Probando endpoint /model-info...")
    try:
        response = requests.get(f"{BASE_URL}/model-info")
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {json.dumps(response.json(), indent=2)}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error: {e}")

def test_root_endpoint():
    """Prueba el endpoint raíz."""
    print("\n🏠 Probando endpoint raíz /...")
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    except Exception as e:
        print(f"Error: {e}")

def main():
    """Función principal para ejecutar todas las pruebas."""
    print("🚀 Iniciando pruebas de la API de Detección de Fraudes")
    print("=" * 60)
    
    # Esperar un momento para asegurar que la API esté lista
    print("Esperando que la API esté lista...")
    time.sleep(2)
    
    # Ejecutar pruebas
    health_ok = test_health_endpoint()
    
    if health_ok:
        test_root_endpoint()
        test_predict_endpoint()
        test_model_info_endpoint()
    else:
        print("\n❌ La API no está disponible. Verificar que esté ejecutándose.")
    
    print("\n" + "=" * 60)
    print("✅ Pruebas completadas")

if __name__ == "__main__":
    main()