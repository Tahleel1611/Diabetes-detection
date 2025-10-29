#!/usr/bin/env python3
"""
Simple Diabetes Prediction Demo
Uses the pre-trained and optimized binary classification model.
"""

import numpy as np
import pandas as pd
import torch
import joblib
import warnings
from models import EnhancedDiabetesModel # Import the correct model

warnings.filterwarnings('ignore')

def predict_diabetes_sample():
    """Simple prediction demo using the optimized binary classification model."""
    try:
        print("\n" + "="*60)
        print(" OPTIMIZED DIABETES PREDICTION DEMO")
        print("="*60)
        print("Classes: 0=No Diabetes, 1=Diabetes")
        print("="*60)
        
        # --- Load Models and Feature Names ---
        print(" Loading trained model, scaler, and feature names...")
        
        # Load feature names to determine model input size
        feature_names = joblib.load('models/pima_feature_names.pkl')
        input_size = len(feature_names)
        print(f" Feature count: {input_size}")
        
        # Load the scaler used during training
        scaler = joblib.load('models/combined_scaler.pkl')
        print(" Scaler loaded")
        
        # Load the optimized Neural Network
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # We need to instantiate the model with the architecture that was found by Optuna
        # These are the best parameters from the last successful run.
        best_params = {
            'hidden_dim1': 235,
            'hidden_dim2': 95,
            'dropout_rate': 0.28723006045434735
        }
        
        nn_model = EnhancedDiabetesModel(
            input_dim=input_size,
            hidden_dim1=best_params['hidden_dim1'],
            hidden_dim2=best_params['hidden_dim2'],
            dropout_rate=best_params['dropout_rate']
        )
        nn_model.load_state_dict(torch.load('models/enhanced_diabetes_nn.pth', map_location=device))
        nn_model.eval()
        print(" Optimized Neural Network loaded")
        
        # --- Load and Preprocess Sample Data ---
        print("\n Loading sample data from 'diabetes.csv' for demonstration...")
        df = pd.read_csv('data/diabetes.csv')
        
        # Sample a few patients for demonstration
        sample_patients = df.sample(n=5, random_state=50)
        
        # Get true labels for comparison
        true_labels = sample_patients['Outcome'].values
        X_sample = sample_patients.drop('Outcome', axis=1)
        
        # --- Apply Same Preprocessing as in Training ---
        # 1. Clean zero values
        zero_replace_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for col in zero_replace_cols:
            X_sample[col] = X_sample[col].replace(0, np.nan)
            # Use the median from the full training data for imputation if needed
            # For this demo, we'll just fill with the sample's median.
            X_sample[col] = X_sample[col].fillna(X_sample[col].median())

        # 2. Feature Engineering
        X_sample['BMI_Age'] = X_sample['BMI'] * X_sample['Age']
        X_sample['Glucose_BMI'] = X_sample['Glucose'] * X_sample['BMI']
        X_sample['Insulin_Glucose'] = X_sample['Insulin'] * X_sample['Glucose']
        X_sample['BMI_squared'] = X_sample['BMI'] ** 2
        X_sample['Age_squared'] = X_sample['Age'] ** 2
        X_sample['BMI_Age_ratio'] = X_sample['BMI'] / (X_sample['Age'] + 1)
        X_sample['Glucose_squared'] = X_sample['Glucose'] ** 2
        X_sample['Glucose_Age_ratio'] = X_sample['Glucose'] / (X_sample['Age'] + 1)
        
        # 3. Align columns with the training feature set
        X_sample = X_sample.reindex(columns=feature_names, fill_value=0)
        
        # 4. Scale features using the loaded scaler
        X_scaled = scaler.transform(X_sample)
        
        print(f" Preprocessed {len(X_scaled)} samples")
        
        # --- Make Predictions ---
        print("\n Making Predictions...")
        print("-" * 60)
        
        class_labels = {0: 'No Diabetes', 1: 'Diabetes'}
        risk_levels = {0: 'Low', 1: 'High'}
        
        for i in range(len(X_scaled)):
            print(f"\nPatient {i+1}:")
            print(f"  True Status: {class_labels[true_labels[i]]}")
            
            # Neural Network prediction for binary classification
            input_tensor = torch.FloatTensor(X_scaled[i:i+1]).to(device)
            with torch.no_grad():
                logits = nn_model(input_tensor)
                probability = torch.sigmoid(logits).item() # Get single probability
                prediction = 1 if probability > 0.5 else 0
            
            print(f"   Prediction: {class_labels[prediction]} ({risk_levels[prediction]} Risk)")
            print(f"     Confidence (Probability of Diabetes): {probability:.2%}")
            
        
        print("\n" + "="*60)
        print(" Prediction demo completed successfully!")
        print(" This system uses an optimized model to predict diabetes risk.")
        print("="*60)
        
    except FileNotFoundError as e:
        print(f" Demo failed: A required file was not found.")
        print(f"   Error: {e}")
        print("   Please ensure you have run the training script (`train.py`) successfully first.")
    except Exception as e:
        print(f" Demo failed with an unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    predict_diabetes_sample()