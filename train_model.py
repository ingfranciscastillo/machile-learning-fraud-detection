"""
Script para entrenar el modelo de detección de fraudes en transacciones.
Entrena múltiples modelos y guarda el mejor según la métrica F1-Score.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                            precision_score, recall_score, f1_score, accuracy_score)
from imblearn.over_sampling import SMOTE
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

class FraudDetectionTrainer:
    """Clase para entrenar modelos de detección de fraudes."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=42)
        self.models = {}
        self.model_scores = {}
        self.best_model_name = None
        self.best_model = None
        
    def load_data(self, filepath):
        """
        Carga el dataset de transacciones.
        
        Args:
            filepath (str): Ruta al archivo CSV del dataset.
            
        Returns:
            pd.DataFrame: Dataset cargado.
        """
        print("Cargando dataset...")
        try:
            df = pd.read_csv(filepath)
            print(f"Dataset cargado exitosamente. Shape: {df.shape}")
            return df
        except FileNotFoundError:
            print("Error: No se encontró el archivo. Descarga el dataset de Kaggle:")
            print("https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
            return None
    
    def preprocess_data(self, df):
        """
        Preprocesa los datos del dataset.
        
        Args:
            df (pd.DataFrame): Dataset original.
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        print("Preprocesando datos...")
        
        # Separar características y variable objetivo
        X = df.drop(['Class'], axis=1)
        y = df['Class']
        
        print(f"Distribución de clases antes de SMOTE:")
        print(f"No Fraude (0): {sum(y == 0)} ({sum(y == 0)/len(y)*100:.2f}%)")
        print(f"Fraude (1): {sum(y == 1)} ({sum(y == 1)/len(y)*100:.2f}%)")
        
        # Dividir en entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Normalizar características
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Aplicar SMOTE para balancear las clases
        X_train_balanced, y_train_balanced = self.smote.fit_resample(
            X_train_scaled, y_train
        )
        
        print(f"Distribución de clases después de SMOTE:")
        print(f"No Fraude (0): {sum(y_train_balanced == 0)}")
        print(f"Fraude (1): {sum(y_train_balanced == 1)}")
        
        return X_train_balanced, X_test_scaled, y_train_balanced, y_test
    
    def initialize_models(self):
        """Inicializa los modelos a entrenar."""
        print("Inicializando modelos...")
        
        self.models = {
            'logistic_regression': LogisticRegression(
                random_state=42, max_iter=1000
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100, random_state=42, n_jobs=-1
            ),
            'neural_network': MLPClassifier(
                hidden_layer_sizes=(100, 50), random_state=42,
                max_iter=300, early_stopping=True
            )
        }
    
    def train_models(self, X_train, y_train):
        """
        Entrena todos los modelos.
        
        Args:
            X_train: Características de entrenamiento.
            y_train: Etiquetas de entrenamiento.
        """
        print("Entrenando modelos...")
        
        for model_name, model in self.models.items():
            print(f"Entrenando {model_name}...")
            model.fit(X_train, y_train)
            print(f"{model_name} entrenado exitosamente.")
    
    def evaluate_models(self, X_test, y_test):
        """
        Evalúa todos los modelos entrenados.
        
        Args:
            X_test: Características de prueba.
            y_test: Etiquetas de prueba.
        """
        print("\nEvaluando modelos...")
        print("="*60)
        
        for model_name, model in self.models.items():
            print(f"\n{model_name.upper().replace('_', ' ')}")
            print("-" * 40)
            
            # Predicciones
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            # Métricas
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Guardar scores
            self.model_scores[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            }
            
            # Imprimir métricas
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1-Score: {f1:.4f}")
            
            # Matriz de confusión
            cm = confusion_matrix(y_test, y_pred)
            print(f"\nMatriz de Confusión:")
            print(f"TN: {cm[0][0]}, FP: {cm[0][1]}")
            print(f"FN: {cm[1][0]}, TP: {cm[1][1]}")
            
            # Reporte de clasificación
            print(f"\nReporte de Clasificación:")
            print(classification_report(y_test, y_pred))
    
    def select_best_model(self):
        """Selecciona el mejor modelo basado en F1-Score."""
        print("\nSeleccionando el mejor modelo...")
        
        best_f1 = 0
        for model_name, scores in self.model_scores.items():
            if scores['f1_score'] > best_f1:
                best_f1 = scores['f1_score']
                self.best_model_name = model_name
                self.best_model = self.models[model_name]
        
        print(f"Mejor modelo: {self.best_model_name}")
        print(f"F1-Score: {best_f1:.4f}")
        
    def save_model(self, model_dir='models'):
        """
        Guarda el mejor modelo y el scaler.
        
        Args:
            model_dir (str): Directorio donde guardar los modelos.
        """
        # Crear directorio si no existe
        os.makedirs(model_dir, exist_ok=True)
        
        # Guardar el mejor modelo
        model_path = os.path.join(model_dir, 'best_fraud_model.pkl')
        joblib.dump(self.best_model, model_path)
        print(f"Modelo guardado en: {model_path}")
        
        # Guardar el scaler
        scaler_path = os.path.join(model_dir, 'scaler.pkl')
        joblib.dump(self.scaler, scaler_path)
        print(f"Scaler guardado en: {scaler_path}")
        
        # Guardar información del modelo
        model_info = {
            'model_name': self.best_model_name,
            'model_scores': self.model_scores[self.best_model_name]
        }
        info_path = os.path.join(model_dir, 'model_info.pkl')
        joblib.dump(model_info, info_path)
        print(f"Información del modelo guardada en: {info_path}")


def main():
    """Función principal para entrenar el modelo."""
    # Inicializar trainer
    trainer = FraudDetectionTrainer()
    
    # Cargar datos
    # NOTA: Debes descargar el dataset de Kaggle y colocarlo en la ruta correcta
    data_path = 'creditcard.csv'  # Cambia esta ruta según tu archivo
    df = trainer.load_data(data_path)
    
    if df is None:
        print("Creando dataset de ejemplo para demostración...")
        # Crear dataset sintético para demostración
        np.random.seed(42)
        n_samples = 10000
        n_features = 30
        
        # Generar características normalizadas
        X_normal = np.random.normal(0, 1, (int(n_samples * 0.998), n_features))
        y_normal = np.zeros(int(n_samples * 0.998))
        
        # Generar transacciones fraudulentas con patrones diferentes
        X_fraud = np.random.normal(2, 1.5, (int(n_samples * 0.002), n_features))
        y_fraud = np.ones(int(n_samples * 0.002))
        
        # Combinar datos
        X = np.vstack([X_normal, X_fraud])
        y = np.hstack([y_normal, y_fraud])
        
        # Crear DataFrame
        columns = [f'V{i}' for i in range(1, n_features)] + ['Amount']
        df = pd.DataFrame(X, columns=columns)
        df['Class'] = y
        
        print(f"Columnas creadas: {columns}")
        print(f"Número de características: {len(columns)}")
        
        print(f"Dataset sintético creado. Shape: {df.shape}")
    
    # Preprocesar datos
    X_train, X_test, y_train, y_test = trainer.preprocess_data(df)
    
    # Inicializar y entrenar modelos
    trainer.initialize_models()
    trainer.train_models(X_train, y_train)
    
    # Evaluar modelos
    trainer.evaluate_models(X_test, y_test)
    
    # Seleccionar mejor modelo
    trainer.select_best_model()
    
    # Guardar modelo
    trainer.save_model()
    
    print("\n¡Entrenamiento completado exitosamente!")


if __name__ == "__main__":
    main()