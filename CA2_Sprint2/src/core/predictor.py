"""
Predictor Module
Handles ML model loading and predictions using Phase 1C model
"""

import pandas as pd
import numpy as np
import joblib
import json
from typing import Dict, Tuple
from pathlib import Path

from .. import config


class Predictor:
    """
    Handles machine learning predictions for trip safety
    Integrates Phase 1C trained model (75% recall, 35% precision)
    """
    
    def __init__(self):
        """Initialize the predictor and load Phase 1C model"""
        self.model = None
        self.scaler = None
        self.selected_features = []
        self.threshold = config.PREDICTION_THRESHOLD
        self.model_config = None
        self.is_loaded = False
        
        # Automatically load model on initialization
        self._load_model()
        
    def _load_model(self):
        """Load the Phase 1C trained model and dependencies"""
        try:
            print("=" * 60)
            print("🔄 Loading Phase 1C Model...")
            print("=" * 60)
            
            # Check if model files exist
            if not config.MODEL_PATH.exists():
                raise FileNotFoundError(f"Model not found: {config.MODEL_PATH}")
            
            # Load the trained Logistic Regression model
            self.model = joblib.load(config.MODEL_PATH)
            print(f"✅ Model loaded: {config.MODEL_PATH.name}")
            
            # Load the feature scaler
            if config.SCALER_PATH.exists():
                self.scaler = joblib.load(config.SCALER_PATH)
                print(f"✅ Scaler loaded: {config.SCALER_PATH.name}")
            else:
                print(f"⚠️  No scaler found (features may not be scaled)")
                self.scaler = None
            
            # Load selected features list
            if config.FEATURES_PATH.exists():
                with open(config.FEATURES_PATH, 'r') as f:
                    self.selected_features = [line.strip() for line in f.readlines()]
                print(f"✅ Features loaded: {len(self.selected_features)} features")
            else:
                # Fallback to config
                self.selected_features = config.MODEL_FEATURES
                print(f"⚠️  Using features from config: {len(self.selected_features)} features")
            
            # Load model configuration
            if config.CONFIG_PATH.exists():
                with open(config.CONFIG_PATH, 'r') as f:
                    self.model_config = json.load(f)
                    self.threshold = self.model_config.get('threshold', config.PREDICTION_THRESHOLD)
                print(f"✅ Config loaded: threshold={self.threshold}")
            else:
                print(f"⚠️  Using default threshold: {self.threshold}")
            
            self.is_loaded = True
            
            print("=" * 60)
            print("🎉 Model Ready for Predictions!")
            print("=" * 60)
            print(f"   Model Type: Logistic Regression")
            print(f"   Features: {len(self.selected_features)}")
            print(f"   Threshold: {self.threshold}")
            if self.model_config:
                print(f"   Expected Recall: {self.model_config.get('validation_metrics', {}).get('recall', 'N/A'):.2%}")
                print(f"   Expected Precision: {self.model_config.get('validation_metrics', {}).get('precision', 'N/A'):.2%}")
            print("=" * 60)
            
        except Exception as e:
            print(f"❌ ERROR loading model: {e}")
            print(f"   Make sure Phase 1C model is saved in: {config.MODEL_DIR}")
            self.is_loaded = False
            raise
    
    def predict_batch(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict safety for batch of trips
        
        Args:
            features_df: DataFrame with trip features (must include selected features)
            
        Returns:
            DataFrame with predictions and probabilities
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded! Cannot make predictions.")
        
        print(f"\n🔮 Making predictions for {len(features_df)} trips...")
        
        # Validate features
        missing_features = [f for f in self.selected_features if f not in features_df.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Select only the model features (in correct order)
        X = features_df[self.selected_features].copy()
        
        # Handle any missing values
        X = X.fillna(0)
        
        # Scale features if scaler is available
        if self.scaler is not None:
            X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        # Get probability predictions
        probabilities = self.model.predict_proba(X_scaled)
        proba_safe = probabilities[:, 0]
        proba_dangerous = probabilities[:, 1]
        
        # Apply custom threshold (from Phase 1C)
        predictions = (proba_dangerous >= self.threshold).astype(int)
        
        # Create results DataFrame
        results = pd.DataFrame()
        results['bookingID'] = features_df['bookingID']
        results['prediction'] = predictions
        results['prediction_label'] = results['prediction'].map({
            0: 'SAFE',
            1: 'DANGEROUS'
        })
        results['probability_safe'] = proba_safe
        results['probability_dangerous'] = proba_dangerous
        
        # Confidence = probability of predicted class
        results['confidence'] = results.apply(
            lambda row: row['probability_safe'] if row['prediction'] == 0 
                       else row['probability_dangerous'],
            axis=1
        )
        
        # Summary statistics
        n_dangerous = (predictions == 1).sum()
        n_safe = len(predictions) - n_dangerous
        
        print(f"✅ Predictions complete:")
        print(f"   Dangerous: {n_dangerous} ({n_dangerous/len(predictions)*100:.1f}%)")
        print(f"   Safe: {n_safe} ({n_safe/len(predictions)*100:.1f}%)")
        
        return results
    
    def predict_single(self, features_df: pd.DataFrame) -> Tuple[str, float]:
        """
        Predict safety for a single trip
        
        Args:
            features_df: DataFrame with one row of trip features
            
        Returns:
            Tuple of (prediction_label, confidence)
        """
        results = self.predict_batch(features_df)
        prediction = results['prediction_label'].iloc[0]
        confidence = results['confidence'].iloc[0]
        
        return prediction, confidence
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance from the Logistic Regression model
        
        Returns:
            Dictionary mapping feature names to importance scores (coefficients)
        """
        if not self.is_loaded or self.model is None:
            return {}
        
        # For Logistic Regression, coefficients represent feature importance
        try:
            coefficients = self.model.coef_[0]
            importance = {
                feature: abs(coef) 
                for feature, coef in zip(self.selected_features, coefficients)
            }
            
            # Sort by importance
            importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True))
            
            return importance
        except:
            return {}
    
    def model_info(self) -> Dict:
        """
        Get information about the loaded model
        
        Returns:
            Dictionary with model metadata
        """
        info = {
            'loaded': self.is_loaded,
            'model_path': str(config.MODEL_PATH) if config.MODEL_PATH else None,
            'model_type': 'Logistic Regression (Phase 1C)',
            'n_features': len(self.selected_features),
            'features': self.selected_features,
            'threshold': self.threshold,
        }
        
        if self.model_config:
            info['validation_metrics'] = self.model_config.get('validation_metrics', {})
            info['hyperparameters'] = self.model_config.get('hyperparameters', {})
        
        return info
