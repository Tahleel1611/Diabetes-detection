#!/usr/bin/env python3
"""
Clean Diabetes Prediction Script
Simple interface for making predictions with trained models
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

class DiabetesPredictor:
    """Clean diabetes prediction interface"""
    
    def __init__(self, model_path=None, rf_path=None, scaler_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Default paths
        if model_path is None:
            model_path = 'models/clean_diabetes_nn.pth'
        if rf_path is None:
            rf_path = 'models/clean_diabetes_rf.pkl'
        if scaler_path is None:
            scaler_path = 'models/clean_scaler.pkl'
        
        # Load models
        self.load_models(model_path, rf_path, scaler_path)
    
    def load_models(self, model_path, rf_path, scaler_path):
        """Load trained models and scaler"""
        try:
            # Load scaler
            self.scaler = joblib.load(scaler_path)
            
            # Load Random Forest
            self.rf_model = joblib.load(rf_path)
            
            # Load Neural Network
            # First need to determine input size from scaler
            input_size = len(self.scaler.feature_names_in_) if hasattr(self.scaler, 'feature_names_in_') else 14
            self.nn_model = CleanDiabetesModel(input_size)
            self.nn_model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.nn_model.eval()
            self.nn_model.to(self.device)
            
            print("✅ Models loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            print("Please ensure you've trained the models first by running: python train_clean.py")
            raise
    
    def engineer_features(self, data):
        """Apply feature engineering to input data"""
        df = data.copy()
        
        # Add engineered features
        df['BMI_Age'] = df['BMI'] * df['Age']
        df['Glucose_BMI'] = df['Glucose'] * df['BMI']
        df['Insulin_Glucose'] = df['Insulin'] * df['Glucose']
        df['BMI_squared'] = df['BMI'] ** 2
        df['Age_squared'] = df['Age'] ** 2
        df['Glucose_Age_ratio'] = df['Glucose'] / (df['Age'] + 1)
        
        return df
    
    def predict_single(self, patient_data, use_ensemble=True):
        """Make prediction for a single patient"""
        # Convert to DataFrame if it's a dict
        if isinstance(patient_data, dict):
            df = pd.DataFrame([patient_data])
        else:
            df = patient_data.copy()
        
        # Apply feature engineering
        df_engineered = self.engineer_features(df)
        
        # Scale features
        X_scaled = self.scaler.transform(df_engineered)
        
        # Get predictions from both models
        # Random Forest prediction
        rf_prob = self.rf_model.predict_proba(X_scaled)[0, 1]
        
        # Neural Network prediction
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        with torch.no_grad():
            nn_prob = self.nn_model(X_tensor).cpu().numpy()[0, 0]
        
        if use_ensemble:
            # Ensemble prediction (average)
            final_prob = (rf_prob + nn_prob) / 2
            prediction = int(final_prob > 0.5)
        else:
            # Use Neural Network as primary
            final_prob = nn_prob
            prediction = int(final_prob > 0.5)
        
        return {
            'prediction': prediction,
            'probability': float(final_prob),
            'risk_level': self.get_risk_level(final_prob),
            'rf_probability': float(rf_prob),
            'nn_probability': float(nn_prob)
        }
    
    def get_risk_level(self, probability):
        """Convert probability to risk level"""
        if probability < 0.3:
            return "Low Risk"
        elif probability < 0.7:
            return "Moderate Risk"
        else:
            return "High Risk"
    
    def predict_batch(self, patient_data_list):
        """Make predictions for multiple patients"""
        results = []
        for patient_data in patient_data_list:
            result = self.predict_single(patient_data)
            results.append(result)
        return results
    
    def explain_prediction(self, patient_data):
        """Provide explanation for the prediction"""
        result = self.predict_single(patient_data)
        
        # Get feature importances from Random Forest
        feature_names = self.rf_model.feature_names_in_
        importances = self.rf_model.feature_importances_
        
        # Sort by importance
        importance_pairs = list(zip(feature_names, importances))
        importance_pairs.sort(key=lambda x: x[1], reverse=True)
        
        explanation = {
            'prediction_result': result,
            'top_risk_factors': importance_pairs[:5],
            'patient_values': patient_data
        }
        
        return explanation

def create_sample_patients():
    """Create sample patient data for testing"""
    return [
        {
            'Pregnancies': 1,
            'Glucose': 85,
            'BloodPressure': 66,
            'SkinThickness': 29,
            'Insulin': 0,
            'BMI': 26.6,
            'DiabetesPedigreeFunction': 0.351,
            'Age': 31
        },
        {
            'Pregnancies': 6,
            'Glucose': 148,
            'BloodPressure': 72,
            'SkinThickness': 35,
            'Insulin': 0,
            'BMI': 33.6,
            'DiabetesPedigreeFunction': 0.627,
            'Age': 50
        },
        {
            'Pregnancies': 0,
            'Glucose': 137,
            'BloodPressure': 40,
            'SkinThickness': 35,
            'Insulin': 168,
            'BMI': 43.1,
            'DiabetesPedigreeFunction': 2.288,
            'Age': 33
        }
    ]

def demo_predictions():
    """Demonstrate the prediction system"""
    print("\n" + "="*60)
    print("🏥 DIABETES PREDICTION DEMO")
    print("="*60)
    
    try:
        # Initialize predictor
        predictor = DiabetesPredictor()
        
        # Get sample patients
        sample_patients = create_sample_patients()
        
        print("\n📊 Sample Predictions:")
        print("-" * 60)
        
        for i, patient in enumerate(sample_patients, 1):
            print(f"\nPatient {i}:")
            print(f"  Age: {patient['Age']}, BMI: {patient['BMI']}, Glucose: {patient['Glucose']}")
            
            result = predictor.predict_single(patient)
            
            print(f"  🎯 Prediction: {'Diabetic' if result['prediction'] else 'Non-Diabetic'}")
            print(f"  📈 Probability: {result['probability']:.1%}")
            print(f"  ⚠️  Risk Level: {result['risk_level']}")
            print(f"  🤖 NN Prob: {result['nn_probability']:.1%}, 🌲 RF Prob: {result['rf_probability']:.1%}")
        
        # Detailed explanation for first patient
        print(f"\n📋 Detailed Analysis for Patient 1:")
        print("-" * 60)
        explanation = predictor.explain_prediction(sample_patients[0])
        
        print(f"Top Risk Factors:")
        for factor, importance in explanation['top_risk_factors']:
            print(f"  • {factor}: {importance:.3f}")
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("Please run training first: python train_clean.py")

def interactive_prediction():
    """Interactive prediction interface"""
    print("\n" + "="*60)
    print("🏥 INTERACTIVE DIABETES PREDICTION")
    print("="*60)
    
    try:
        predictor = DiabetesPredictor()
        
        print("\nEnter patient information:")
        patient_data = {}
        
        fields = [
            ('Pregnancies', 'Number of pregnancies'),
            ('Glucose', 'Glucose level'),
            ('BloodPressure', 'Blood pressure'),
            ('SkinThickness', 'Skin thickness'),
            ('Insulin', 'Insulin level'),
            ('BMI', 'Body Mass Index'),
            ('DiabetesPedigreeFunction', 'Diabetes pedigree function'),
            ('Age', 'Age')
        ]
        
        for field, description in fields:
            while True:
                try:
                    value = float(input(f"{description} ({field}): "))
                    patient_data[field] = value
                    break
                except ValueError:
                    print("Please enter a valid number.")
        
        # Make prediction
        result = predictor.predict_single(patient_data)
        
        print("\n" + "="*60)
        print("🎯 PREDICTION RESULTS")
        print("="*60)
        print(f"Prediction: {'DIABETIC' if result['prediction'] else 'NON-DIABETIC'}")
        print(f"Probability: {result['probability']:.1%}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Neural Network: {result['nn_probability']:.1%}")
        print(f"Random Forest: {result['rf_probability']:.1%}")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    """Main function - BRFSS Multi-Class Diabetes Prediction Demo"""
    print("\n" + "="*60)
    print("🏥 BRFSS DIABETES PREDICTION SYSTEM (MULTI-CLASS)")
    print("="*60)
    print("Classes: 0=No Diabetes, 1=Pre-diabetes, 2=Diabetes")
    print("="*60)
    
    try:
        # Load models
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load scaler
        scaler = joblib.load('models/clean_scaler.pkl')
        
        # Load Random Forest
        rf_model = joblib.load('models/clean_diabetes_rf.pkl')
        
        # Load Neural Network
        # Get input size from a sample of the BRFSS data with proper preprocessing
        df_sample = pd.read_csv('data/2023_BRFSS_CLEANED.csv', nrows=1000)
        if 'YEAR' in df_sample.columns:
            df_sample = df_sample.drop('YEAR', axis=1)
        target_col = 'DIABETES_STATUS'
        X_sample = df_sample.drop(target_col, axis=1)
        
        # Apply same preprocessing as training to get correct feature count
        categorical_cols = [col for col in X_sample.columns if X_sample[col].dtype == 'object' or X_sample[col].nunique() < 10]
        numerical_cols = [col for col in X_sample.columns if col not in categorical_cols]
        
        # Impute missing values
        for col in categorical_cols:
            X_sample[col] = X_sample[col].fillna(X_sample[col].mode()[0])
        for col in numerical_cols:
            X_sample[col] = X_sample[col].fillna(X_sample[col].median())
        
        # One-hot encode categorical variables
        X_sample = pd.get_dummies(X_sample, columns=categorical_cols, drop_first=True)
        input_size = X_sample.shape[1]
        
        nn_model = CleanDiabetesModel(input_size, num_classes=4)
        nn_model.load_state_dict(torch.load('models/clean_diabetes_nn.pth', map_location=device))
        nn_model.eval()
        
        print(f"✅ Models loaded successfully!")
        print(f"📊 Feature dimensions: {input_size}")
        
        # Sample BRFSS patient data (simplified for demo)
        sample_patients = [
            {
                'AGE': 35, 'BMI': 25.5, 'GENERAL_HEALTH': 'Good',
                'PHYSICAL_HEALTH_DAYS': 0, 'MENTAL_HEALTH_DAYS': 1,
                'HEALTH_CARE_ACCESS': 'Yes', 'EXERCISE': 'Yes',
                'HEART_DISEASE': 'No', 'HIGH_BLOOD_PRESSURE': 'No',
                'HIGH_CHOLESTEROL': 'No', 'STROKE': 'No', 'SMOKING': 'Never'
            },
            {
                'AGE': 55, 'BMI': 32.1, 'GENERAL_HEALTH': 'Fair',
                'PHYSICAL_HEALTH_DAYS': 5, 'MENTAL_HEALTH_DAYS': 3,
                'HEALTH_CARE_ACCESS': 'Yes', 'EXERCISE': 'No',
                'HEART_DISEASE': 'No', 'HIGH_BLOOD_PRESSURE': 'Yes',
                'HIGH_CHOLESTEROL': 'Yes', 'STROKE': 'No', 'SMOKING': 'Former'
            },
            {
                'AGE': 65, 'BMI': 35.8, 'GENERAL_HEALTH': 'Poor',
                'PHYSICAL_HEALTH_DAYS': 15, 'MENTAL_HEALTH_DAYS': 8,
                'HEALTH_CARE_ACCESS': 'Limited', 'EXERCISE': 'No',
                'HEART_DISEASE': 'Yes', 'HIGH_BLOOD_PRESSURE': 'Yes',
                'HIGH_CHOLESTEROL': 'Yes', 'STROKE': 'No', 'SMOKING': 'Current'
            }
        ]
        
        class_labels = {0: 'No Diabetes', 1: 'Pre-diabetes', 2: 'Pre-diabetes/Borderline', 3: 'Diabetes'}
        risk_levels = {0: 'Low', 1: 'Medium', 2: 'Medium-High', 3: 'High'}
        
        print("\n📊 Sample Predictions:")
        print("-" * 60)
        
        for i, patient in enumerate(sample_patients, 1):
            print(f"\nPatient {i}:")
            print(f"  Age: {patient['AGE']}, BMI: {patient['BMI']}")
            print(f"  Health: {patient['GENERAL_HEALTH']}, Exercise: {patient['EXERCISE']}")
            print(f"  Conditions: BP={patient['HIGH_BLOOD_PRESSURE']}, Chol={patient['HIGH_CHOLESTEROL']}")
            
            try:
                # Simple preprocessing for demo (not complete BRFSS pipeline)
                # Create a DataFrame
                df = pd.DataFrame([patient])
                
                # Fill missing categorical columns with defaults
                for col in categorical_cols:
                    if col not in df.columns:
                        df[col] = 'Unknown'
                
                # Fill missing numerical columns with defaults
                numerical_cols = [col for col in X_sample.columns if col not in categorical_cols]
                for col in numerical_cols:
                    if col not in df.columns:
                        df[col] = 0
                
                # One-hot encode
                df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
                
                # Align with training features
                for col in X_sample.columns:
                    if col not in df_encoded.columns:
                        df_encoded[col] = 0
                
                # Reorder columns to match training
                df_encoded = df_encoded.reindex(columns=X_sample.columns, fill_value=0)
                
                # Scale features
                features_scaled = scaler.transform(df_encoded)
                
                # Neural Network prediction
                input_tensor = torch.FloatTensor(features_scaled)
                with torch.no_grad():
                    logits = nn_model(input_tensor)
                    probabilities = torch.softmax(logits, dim=1)
                    nn_pred_class = torch.argmax(probabilities, dim=1).item()
                    nn_class_probs = probabilities.squeeze().numpy()
                
                print(f"  🤖 Neural Network:")
                print(f"     Prediction: {class_labels[nn_pred_class]}")
                print(f"     Risk Level: {risk_levels[nn_pred_class]}")
                print(f"     Probabilities: No Diabetes: {nn_class_probs[0]:.1%}, Pre-diabetes: {nn_class_probs[1]:.1%}, Diabetes: {nn_class_probs[2]:.1%}")
                
                # Random Forest prediction
                rf_pred_class = rf_model.predict(features_scaled)[0]
                rf_class_probs = rf_model.predict_proba(features_scaled)[0]
                
                print(f"  🌲 Random Forest:")
                print(f"     Prediction: {class_labels[rf_pred_class]}")
                print(f"     Risk Level: {risk_levels[rf_pred_class]}")
                print(f"     Probabilities: No Diabetes: {rf_class_probs[0]:.1%}, Pre-diabetes: {rf_class_probs[1]:.1%}, Diabetes: {rf_class_probs[2]:.1%}")
                
            except Exception as e:
                print(f"  ❌ Prediction error: {e}")
                print(f"     Note: This is a simplified demo. Full BRFSS preprocessing required.")
        
        print("\n" + "="*60)
        print("💡 To use this system properly:")
        print("   1. Train models with: python train.py")
        print("   2. Provide complete BRFSS feature set")
        print("   3. Use proper feature preprocessing pipeline")
        print("="*60)
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        print("Please run training first: python train.py")

if __name__ == "__main__":
    main()