import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (classification_report,
                            confusion_matrix,
                            precision_recall_curve,
                            average_precision_score,
                            roc_auc_score)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import json
import logging
from pathlib import Path
from typing import Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FraudDetectionModel:
    def __init__(self, model_path: str = 'model.pkl'):
        self.model_path = Path(model_path)
        self.scaler = StandardScaler()
        self.model = None
        self.feature_names = ['avg_keystroke_latency', 'avg_mouse_speed']
        self.params = {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'bootstrap': True,
            'class_weight': 'balanced'
        }

    def create_sample_data(self, n_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Generate synthetic training data with realistic patterns"""
        np.random.seed(42)
        
        # Normal transactions (shorter latencies, moderate speed)
        normal_latency = np.random.exponential(scale=0.1, size=n_samples//2)
        normal_speed = np.random.normal(loc=150, scale=50, size=n_samples//2)
        normal_data = np.column_stack((normal_latency, normal_speed))
        normal_labels = np.zeros(n_samples//2)
        
        # Fraudulent transactions (longer latencies, either very slow or fast)
        fraud_latency = np.random.exponential(scale=0.3, size=n_samples//4)
        fraud_speed_slow = np.random.normal(loc=80, scale=20, size=n_samples//4)
        fraud_speed_fast = np.random.normal(loc=400, scale=100, size=n_samples//4)
        fraud_data = np.vstack((
            np.column_stack((fraud_latency, fraud_speed_slow)),
            np.column_stack((fraud_latency, fraud_speed_fast))
        ))
        fraud_labels = np.ones(n_samples//2)
        
        X = np.vstack((normal_data, fraud_data))
        y = np.concatenate((normal_labels, fraud_labels))
        
        return X, y
    
    def preprocess_data(self, X: np.ndarray) -> np.ndarray:
        """Standardize features by removing mean and scaling to unit variance"""
        return self.scaler.fit_transform(X)
    
    def handle_imbalance(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply SMOTE oversampling to handle class imbalance"""
        smote = SMOTE(random_state=42)
        return smote.fit_resample(X, y)
    
    def train_model(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train and optimize the Random Forest model with cross-validation"""
        logging.info("Starting model training...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Preprocess
        X_train = self.preprocess_data(X_train)
        X_test = self.scaler.transform(X_test)  # Use same scaler as training
        
        # Handle class imbalance
        X_train_smote, y_train_smote = self.handle_imbalance(X_train, y_train)
        
        # Hyperparameter tuning
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        self.model = RandomForestClassifier(
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        
        grid_search = GridSearchCV(
            estimator=self.model,
            param_grid=param_grid,
            cv=5,
            scoring='average_precision',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train_smote, y_train_smote)
        self.model = grid_search.best_estimator_
        logging.info(f"Best params: {grid_search.best_params_}")
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_proba = self.model.predict_proba(X_test)[:, 1]
        
        logging.info("\nClassification Report:")
        logging.info(classification_report(y_test, y_pred))
        logging.info("\nConfusion Matrix:")
        logging.info(confusion_matrix(y_test, y_pred))
        logging.info(f"\nROC AUC: {roc_auc_score(y_test, y_proba):.4f}")
        logging.info(f"Average Precision: {average_precision_score(y_test, y_proba):.4f}")
        
        # Feature Importance
        feature_importances = dict(zip(self.feature_names, self.model.feature_importances_))
        logging.info(f"\nFeature Importances: {feature_importances}")
        
    def save_model(self) -> None:
        """Save the trained model and metadata"""
        model_metadata = {
            'model_name': 'fraud_detection_rf',
            'version': '1.0',
            'features': self.feature_names,
            'class_names': ['normal', 'fraud'],
            'performance': {
                'description': 'Evaluation metrics on test set'
            }
        }
        
        # Create directory if it doesn't exist
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model and metadata
        joblib.dump(self.model, self.model_path)
        with open(self.model_path.parent / 'model_metadata.json', 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        logging.info(f"Model saved to {self.model_path}")

if __name__ == '__main__':
    # Initialize and train the model
    fraud_model = FraudDetectionModel()
    
    # Generate synthetic data (replace with your real data)
    X, y = fraud_model.create_sample_data(n_samples=10000)
    
    # Preprocess and train
    fraud_model.train_model(X, y)
    fraud_model.save_model()
