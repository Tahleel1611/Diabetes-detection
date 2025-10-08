#!/usr/bin/env python3
"""
Simple BRFSS Diabetes Prediction Demo
Uses pre-trained models to make predictions
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import warnings
warnings.filterwarnings('ignore')

class CleanDiabetesModel(nn.Module):
    """Clean, efficient neural network for diabetes detection (multi-class)"""
    
    def __init__(self, input_size, hidden_sizes=[128, 64, 32], dropout_rate=0.3, num_classes=4):
        super(CleanDiabetesModel, self).__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        
        # Output layer for multi-class
        layers.append(nn.Linear(prev_size, num_classes))
        # No activation here; will use softmax for probabilities
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def predict_diabetes_sample():
    """Simple prediction demo using pre-trained models"""
    try:
        print("\n" + "="*60)
        print("🏥 BRFSS DIABETES PREDICTION DEMO")
        print("="*60)
        print("Classes: 0=No Diabetes, 1=Pre-diabetes, 2=Pre-diabetes/Borderline, 3=Diabetes")
        print("="*60)
        
        # Load models and feature names
        print("📂 Loading trained models...")
        
        # Load feature names and get input size
        feature_names = joblib.load('models/feature_names.pkl')
        input_size = len(feature_names)
        print(f"✅ Feature count: {input_size}")
        
        # Load scaler
        scaler = joblib.load('models/clean_scaler.pkl')
        print("✅ Scaler loaded")
        
        # Load Random Forest
        rf_model = joblib.load('models/clean_diabetes_rf.pkl')
        print("✅ Random Forest loaded")
        
        # Load Neural Network
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        nn_model = CleanDiabetesModel(input_size, num_classes=4)
        nn_model.load_state_dict(torch.load('models/clean_diabetes_nn.pth', map_location=device))
        nn_model.eval()
        print("✅ Neural Network loaded")
        
        # Use test data for demonstration
        print("\n📊 Loading test data for demonstration...")
        df = pd.read_csv('data/2023_BRFSS_CLEANED.csv')
        
        # Sample a few patients for demonstration
        sample_patients = df.sample(n=5, random_state=42)
        
        # Get target values for comparison
        true_labels = sample_patients['DIABETES_STATUS'].values
        
        # Drop target and apply same preprocessing as training
        if 'YEAR' in sample_patients.columns:
            sample_patients = sample_patients.drop('YEAR', axis=1)
        
        y_sample = sample_patients['DIABETES_STATUS']
        X_sample = sample_patients.drop('DIABETES_STATUS', axis=1)
        
        # Apply preprocessing
        categorical_cols = [col for col in X_sample.columns if X_sample[col].dtype == 'object' or X_sample[col].nunique() < 10]
        numerical_cols = [col for col in X_sample.columns if col not in categorical_cols]
        
        # Impute missing values
        for col in categorical_cols:
            X_sample[col] = X_sample[col].fillna(X_sample[col].mode()[0] if not X_sample[col].mode().empty else 'Unknown')
        for col in numerical_cols:
            X_sample[col] = X_sample[col].fillna(X_sample[col].median())
        
        # One-hot encode
        X_encoded = pd.get_dummies(X_sample, columns=categorical_cols, drop_first=True)
        
        # Align with training features
        for col in feature_names:
            if col not in X_encoded.columns:
                X_encoded[col] = 0
        
        # Reorder columns to match training
        X_encoded = X_encoded.reindex(columns=feature_names, fill_value=0)
        
        # Scale features
        X_scaled = scaler.transform(X_encoded)
        
        print(f"✅ Preprocessed {len(X_scaled)} samples")
        
        # Make predictions
        print("\n🤖 Making Predictions...")
        print("-" * 60)
        
        class_labels = {0: 'No Diabetes', 1: 'Pre-diabetes', 2: 'Pre-diabetes/Borderline', 3: 'Diabetes'}
        risk_levels = {0: 'Low', 1: 'Medium', 2: 'Medium-High', 3: 'High'}
        
        for i in range(len(X_scaled)):
            print(f"\nPatient {i+1}:")
            print(f"  True Status: {class_labels[true_labels[i]]}")
            
            # Neural Network prediction
            input_tensor = torch.FloatTensor(X_scaled[i:i+1])
            with torch.no_grad():
                logits = nn_model(input_tensor)
                probabilities = torch.softmax(logits, dim=1)
                nn_pred_class = torch.argmax(probabilities, dim=1).item()
                nn_class_probs = probabilities.squeeze().numpy()
            
            print(f"  🤖 Neural Network: {class_labels[nn_pred_class]} ({risk_levels[nn_pred_class]} Risk)")
            print(f"     Probabilities: {', '.join([f'{class_labels[j]}: {nn_class_probs[j]:.1%}' for j in range(4)])}")
            
            # Random Forest prediction
            rf_pred_class = rf_model.predict(X_scaled[i:i+1])[0]
            rf_class_probs = rf_model.predict_proba(X_scaled[i:i+1])[0]
            
            print(f"  🌲 Random Forest: {class_labels[rf_pred_class]} ({risk_levels[rf_pred_class]} Risk)")
            print(f"     Probabilities: {', '.join([f'{class_labels[j]}: {rf_class_probs[j]:.1%}' for j in range(4)])}")
            
            # Check if predictions match
            match_nn = "✅" if nn_pred_class == true_labels[i] else "❌"
            match_rf = "✅" if rf_pred_class == true_labels[i] else "❌"
            print(f"  Accuracy: NN {match_nn} | RF {match_rf}")
        
        print("\n" + "="*60)
        print("✅ Prediction demo completed successfully!")
        print("💡 This system can predict 4 classes of diabetes status using BRFSS health data")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    predict_diabetes_sample()