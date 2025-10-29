#!/usr/bin/env python3
"""
Model Interpretability Script
Uses SHAP to explain the predictions of the optimized neural network.
"""

import pandas as pd
import torch
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt
from models import EnhancedDiabetesModel

def explain_model():
    """
    Generates and saves SHAP explanations for the trained binary classification model.
    """
    try:
        print("\n" + "="*60)
        print("🔍 MODEL INTERPRETABILITY (SHAP ANALYSIS)")
        print("="*60)

        # --- 1. Load Model, Data, and Scaler ---
        print("📂 Loading model, test data, and scaler...")
        
        # Load feature names to determine model input size
        feature_names = joblib.load('models/pima_feature_names.pkl')
        input_size = len(feature_names)
        
        # Load the scaler
        scaler = joblib.load('models/combined_scaler.pkl')
        
        # Load the test data
        X_test = pd.read_csv('output/data/X_test_pima.csv')
        y_test = pd.read_csv('output/data/y_test_pima.csv')
        
        # Load the optimized Neural Network
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        best_params = {
            'hidden_dim1': 204,
            'hidden_dim2': 65,
            'dropout_rate': 0.49656753020767197
        }
        nn_model = EnhancedDiabetesModel(
            input_dim=input_size,
            hidden_dim1=best_params['hidden_dim1'],
            hidden_dim2=best_params['hidden_dim2'],
            dropout_rate=best_params['dropout_rate']
        )
        nn_model.load_state_dict(torch.load('models/enhanced_diabetes_nn.pth', map_location=device))
        nn_model.eval()
        nn_model.to(device)
        
        print("✅ All artifacts loaded successfully.")

        # --- 2. Prepare Data for SHAP ---
        # SHAP needs a background dataset to represent the "absence" of a feature.
        # A common practice is to use a smaller, representative subset of the training data.
        # Let's load the training data for this purpose.
        X_train = pd.read_csv('output/data/X_train_pima.csv')
        
        # Using a random sample of 100 data points for the background distribution
        background_data = X_train.sample(n=100, random_state=42)
        
        # We need to convert the test data to tensors for the model
        X_test_tensor = torch.FloatTensor(X_test.values).to(device)

        print("📊 Prepared background and test data for SHAP.")

        # --- 3. Create SHAP Explainer ---
        # For PyTorch models, we use the DeepExplainer.
        # It needs the model and a background data distribution.
        explainer = shap.DeepExplainer(nn_model, torch.FloatTensor(background_data.values).to(device))
        
        print("🧠 SHAP explainer created. Calculating SHAP values...")
        
        # --- 4. Calculate SHAP Values ---
        # This can take a moment as it computes the contribution of each feature for each prediction.
        shap_values = explainer.shap_values(X_test_tensor)
        
        print("✅ SHAP values calculated.")

        # --- 5. Generate and Save Summary Plot ---
        print("🎨 Generating SHAP summary plot...")
        
        # The summary plot shows the most important features for the model's predictions.
        # Red dots indicate a high feature value, blue dots a low feature value.
        # The x-axis is the SHAP value, showing the impact on the model's output.
        plt.figure()
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        
        # Adjust layout and save the plot
        plt.gcf().tight_layout()
        save_path = 'output/plots/shap_summary_plot.png'
        plt.savefig(save_path, dpi=300)
        plt.close()
        
        print(f"✅ SHAP summary plot saved to: {save_path}")
        
        print("\n" + "="*60)
        print("✅ Interpretability analysis completed successfully!")
        print("="*60)

    except FileNotFoundError as e:
        print(f"❌ Analysis failed: A required file was not found.")
        print(f"   Error: {e}")
        print("   Please ensure you have run the training script (`train.py`) successfully first.")
    except Exception as e:
        print(f"❌ Analysis failed with an unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    explain_model()
