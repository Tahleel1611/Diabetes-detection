#!/usr/bin/env python3
"""
Individual Patient Explanation Tool
Provides SHAP and LIME explanations for individual diabetes predictions
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import torch
import os
import warnings
warnings.filterwarnings('ignore')

# Import interpretability libraries
import shap
import lime
from lime import lime_tabular

class PatientExplainer:
    """
    Individual patient explanation using SHAP and LIME
    """
    
    def __init__(self):
        """Initialize the patient explainer"""
        self.rf_model = None
        self.scaler = None
        self.feature_names = None
        self.load_models()
        
        # Sample background data for SHAP
        self.background_data = None
        self.setup_background()
        
        # Initialize explainers
        self.shap_explainer = None
        self.lime_explainer = None
        self.setup_explainers()
    
    def load_models(self):
        """Load the trained models and preprocessing"""
        try:
            self.rf_model = joblib.load('models/clean_diabetes_rf.pkl')
            self.scaler = joblib.load('models/clean_scaler.pkl')
            
            # Load feature names
            if os.path.exists('models/brfss_feature_names.pkl'):
                self.feature_names = joblib.load('models/brfss_feature_names.pkl')
            else:
                self.feature_names = [f'Feature_{i}' for i in range(55)]
            
            print("✅ Models and preprocessing loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def setup_background(self, n_samples=100):
        """Setup background data for SHAP"""
        try:
            # Load and process a small sample of BRFSS data
            df = pd.read_csv('data/2023_BRFSS_CLEANED.csv')
            df_sample = df.sample(n=n_samples, random_state=42)
            
            if 'YEAR' in df_sample.columns:
                df_sample = df_sample.drop('YEAR', axis=1)
            
            X_sample = df_sample.drop('DIABETES_STATUS', axis=1)
            
            # Process categorical and numerical columns
            categorical_cols = [col for col in X_sample.columns if X_sample[col].dtype == 'object' or X_sample[col].nunique() < 10]
            numerical_cols = [col for col in X_sample.columns if col not in categorical_cols]
            
            for col in categorical_cols:
                X_sample[col] = X_sample[col].fillna(X_sample[col].mode()[0])
            for col in numerical_cols:
                X_sample[col] = X_sample[col].fillna(X_sample[col].median())
            
            X_sample = pd.get_dummies(X_sample, columns=categorical_cols, drop_first=True)
            
            # Ensure same features as training
            missing_cols = set(self.feature_names) - set(X_sample.columns)
            for col in missing_cols:
                X_sample[col] = 0
            X_sample = X_sample[self.feature_names]
            
            self.background_data = self.scaler.transform(X_sample)
            print(f"✅ Background data setup: {self.background_data.shape}")
            
        except Exception as e:
            print(f"❌ Error setting up background data: {e}")
    
    def setup_explainers(self):
        """Setup SHAP and LIME explainers"""
        try:
            # SHAP explainer for Random Forest
            self.shap_explainer = shap.TreeExplainer(self.rf_model)
            
            # LIME explainer
            self.lime_explainer = lime_tabular.LimeTabularExplainer(
                self.background_data,
                feature_names=self.feature_names,
                class_names=['No Diabetes', 'Pre-diabetes', 'Pre-diabetes/Borderline', 'Diabetes'],
                mode='classification',
                discretize_continuous=True
            )
            
            print("✅ SHAP and LIME explainers initialized")
            
        except Exception as e:
            print(f"❌ Error setting up explainers: {e}")
    
    def create_sample_patient(self, risk_level='moderate'):
        """Create sample patient data for demonstration"""
        
        sample_patients = {
            'low_risk': {
                'AGE': 35, 'SEX': 1, 'BMI': 22.5, 'SMOKE100': 0, 'CVDINFR4': 0,
                'CVDCRHD4': 0, 'CVDSTRK3': 0, 'ASTHMA3': 0, 'CHCSCNCR': 0,
                'CHCOCNCR': 0, 'CHCCOPD2': 0, 'HAVARTH4': 0, 'ADDEPEV3': 0,
                'CHCKDNY2': 0, 'TOTINDA': 1, 'METVL11_': 600, 'MAXVO2_': 45,
                'FC60_': 120, 'ACTIN11_': 180, 'ACTIN21_': 90, 'PADUR1_': 30,
                'PADUR2_': 15, 'PAFREQ1_': 5, 'PAFREQ2_': 3, '_PACAT1': 1,
                '_PAINDX1': 1, '_PA150R2': 1, '_PA300R2': 1, '_PA65PLR': 0,
                '_PASTRNG': 1, '_PAREC1': 1, '_PASTAE1': 1, '_LMTACT1': 0,
                '_LMTSCL1': 0, '_LMTWRK1': 0, '_LMTSOC1': 0, '_DRDXAR2': 0,
                '_PRACE2': 1, '_MRACE2': 1, '_HISPANC': 0, '_RACE': 1,
                '_RACEG21': 1, '_RACEGR3': 1, '_RACE_G1': 1, '_AGEG5YR': 7,
                '_AGE65YR': 1, '_AGE_G': 2, 'HTIN4': 68, 'HTM4': 173,
                'WTKG3': 68, '_BMI5': 2300, '_BMI5CAT': 2, '_RFBMI5': 1
            },
            'moderate_risk': {
                'AGE': 55, 'SEX': 0, 'BMI': 28.5, 'SMOKE100': 1, 'CVDINFR4': 0,
                'CVDCRHD4': 0, 'CVDSTRK3': 0, 'ASTHMA3': 1, 'CHCSCNCR': 0,
                'CHCOCNCR': 0, 'CHCCOPD2': 0, 'HAVARTH4': 1, 'ADDEPEV3': 0,
                'CHCKDNY2': 0, 'TOTINDA': 0, 'METVL11_': 400, 'MAXVO2_': 35,
                'FC60_': 140, 'ACTIN11_': 120, 'ACTIN21_': 60, 'PADUR1_': 20,
                'PADUR2_': 10, 'PAFREQ1_': 3, 'PAFREQ2_': 2, '_PACAT1': 2,
                '_PAINDX1': 2, '_PA150R2': 2, '_PA300R2': 2, '_PA65PLR': 1,
                '_PASTRNG': 2, '_PAREC1': 2, '_PASTAE1': 2, '_LMTACT1': 1,
                '_LMTSCL1': 0, '_LMTWRK1': 0, '_LMTSOC1': 0, '_DRDXAR2': 1,
                '_PRACE2': 1, '_MRACE2': 1, '_HISPANC': 0, '_RACE': 1,
                '_RACEG21': 1, '_RACEGR3': 1, '_RACE_G1': 1, '_AGEG5YR': 11,
                '_AGE65YR': 1, '_AGE_G': 4, 'HTIN4': 65, 'HTM4': 165,
                'WTKG3': 78, '_BMI5': 2850, '_BMI5CAT': 3, '_RFBMI5': 2
            },
            'high_risk': {
                'AGE': 68, 'SEX': 1, 'BMI': 35.2, 'SMOKE100': 1, 'CVDINFR4': 1,
                'CVDCRHD4': 1, 'CVDSTRK3': 0, 'ASTHMA3': 1, 'CHCSCNCR': 0,
                'CHCOCNCR': 0, 'CHCCOPD2': 1, 'HAVARTH4': 1, 'ADDEPEV3': 1,
                'CHCKDNY2': 0, 'TOTINDA': 0, 'METVL11_': 200, 'MAXVO2_': 25,
                'FC60_': 160, 'ACTIN11_': 60, 'ACTIN21_': 30, 'PADUR1_': 10,
                'PADUR2_': 5, 'PAFREQ1_': 1, 'PAFREQ2_': 1, '_PACAT1': 4,
                '_PAINDX1': 3, '_PA150R2': 2, '_PA300R2': 2, '_PA65PLR': 1,
                '_PASTRNG': 2, '_PAREC1': 2, '_PASTAE1': 2, '_LMTACT1': 1,
                '_LMTSCL1': 1, '_LMTWRK1': 1, '_LMTSOC1': 0, '_DRDXAR2': 1,
                '_PRACE2': 2, '_MRACE2': 2, '_HISPANC': 0, '_RACE': 2,
                '_RACEG21': 2, '_RACEGR3': 2, '_RACE_G1': 2, '_AGEG5YR': 13,
                '_AGE65YR': 2, '_AGE_G': 5, 'HTIN4': 70, 'HTM4': 178,
                'WTKG3': 110, '_BMI5': 3520, '_BMI5CAT': 4, '_RFBMI5': 3
            }
        }
        
        return sample_patients[risk_level]
    
    def explain_patient(self, patient_data, patient_id="Unknown"):
        """
        Provide comprehensive explanation for a patient
        
        Args:
            patient_data: Dictionary with patient features
            patient_id: Patient identifier
        """
        print(f"\n🏥 EXPLAINING DIABETES PREDICTION FOR PATIENT: {patient_id}")
        print("="*60)
        
        try:
            # Convert to DataFrame and ensure all features are present
            patient_df = pd.DataFrame([patient_data])
            
            # Add missing features with default values
            for feature in self.feature_names:
                if feature not in patient_df.columns:
                    patient_df[feature] = 0
            
            # Reorder columns to match training
            patient_df = patient_df[self.feature_names]
            
            # Scale the features
            patient_scaled = self.scaler.transform(patient_df)
            
            # Make prediction
            prediction_proba = self.rf_model.predict_proba(patient_scaled)[0]
            prediction = self.rf_model.predict(patient_scaled)[0]
            
            class_names = ['No Diabetes', 'Pre-diabetes', 'Pre-diabetes/Borderline', 'Diabetes']
            predicted_class = class_names[prediction]
            confidence = prediction_proba[prediction] * 100
            
            print(f"🎯 PREDICTION: {predicted_class}")
            print(f"🔍 CONFIDENCE: {confidence:.1f}%")
            print(f"📊 CLASS PROBABILITIES:")
            for i, (class_name, prob) in enumerate(zip(class_names, prediction_proba)):
                print(f"   {class_name}: {prob*100:.1f}%")
            
            # SHAP Explanation
            print(f"\n🔬 SHAP EXPLANATION:")
            shap_values = self.shap_explainer.shap_values(patient_scaled)
            
            # Handle different SHAP output formats
            if isinstance(shap_values, list):
                # Multi-class case - use predicted class
                shap_values_patient = shap_values[prediction]
                expected_value = self.shap_explainer.expected_value[prediction]
            else:
                # Single output case
                if len(shap_values.shape) > 1 and shap_values.shape[0] == 1:
                    shap_values_patient = shap_values[0]
                else:
                    shap_values_patient = shap_values
                expected_value = self.shap_explainer.expected_value
            
            # Ensure we have a 1D array for feature contributions
            if len(shap_values_patient.shape) > 1:
                shap_values_patient = shap_values_patient.flatten()
            
            # Get top contributing features
            feature_contributions = []
            for i, (feature, contribution) in enumerate(zip(self.feature_names, shap_values_patient)):
                feature_contributions.append((feature, float(contribution)))
            
            feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            
            print(f"   Top 10 Contributing Factors:")
            for i, (feature, contribution) in enumerate(feature_contributions[:10]):
                direction = "↑ Increases" if contribution > 0 else "↓ Decreases"
                print(f"   {i+1:2d}. {feature}: {contribution:+.3f} ({direction} risk)")
            
            # LIME Explanation
            print(f"\n🔍 LIME EXPLANATION:")
            lime_explanation = self.lime_explainer.explain_instance(
                patient_scaled[0], 
                self.rf_model.predict_proba,
                num_features=10,
                top_labels=1
            )
            
            # Get LIME explanation for predicted class
            lime_features = lime_explanation.as_list(label=prediction)
            print(f"   Key Factors (Local Interpretation):")
            for i, (feature, importance) in enumerate(lime_features):
                direction = "↑ Increases" if importance > 0 else "↓ Decreases"
                print(f"   {i+1:2d}. {feature}: {importance:+.3f} ({direction} risk)")
            
            # Create visualizations
            self.create_patient_visualizations(
                patient_scaled[0], shap_values_patient, lime_explanation, 
                patient_id, predicted_class, confidence
            )
            
            # Clinical interpretation
            self.provide_clinical_interpretation(feature_contributions[:5], predicted_class)
            
            return {
                'prediction': predicted_class,
                'confidence': confidence,
                'probabilities': prediction_proba,
                'shap_values': shap_values_patient,
                'lime_explanation': lime_explanation
            }
            
        except Exception as e:
            print(f"❌ Error explaining patient: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def create_patient_visualizations(self, patient_data, shap_values_patient, lime_explanation, 
                                    patient_id, prediction, confidence):
        """Create visualizations for patient explanation"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            
            # SHAP feature importance
            feature_contributions = list(zip(self.feature_names, shap_values_patient))
            feature_contributions.sort(key=lambda x: abs(x[1]), reverse=True)
            
            top_features = feature_contributions[:15]
            features, contributions = zip(*top_features)
            colors = ['red' if c < 0 else 'green' for c in contributions]
            
            axes[0, 0].barh(range(len(features)), contributions, color=colors, alpha=0.7)
            axes[0, 0].set_yticks(range(len(features)))
            axes[0, 0].set_yticklabels(features, fontsize=10)
            axes[0, 0].set_xlabel('SHAP Value')
            axes[0, 0].set_title('SHAP Feature Contributions', fontweight='bold')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].axvline(x=0, color='black', linestyle='-', alpha=0.5)
            
            # LIME feature importance
            lime_features = lime_explanation.as_list(label=lime_explanation.available_labels()[0])
            if lime_features:
                lime_feat_names, lime_importances = zip(*lime_features)
                lime_colors = ['red' if imp < 0 else 'green' for imp in lime_importances]
                
                axes[0, 1].barh(range(len(lime_feat_names)), lime_importances, 
                               color=lime_colors, alpha=0.7)
                axes[0, 1].set_yticks(range(len(lime_feat_names)))
                axes[0, 1].set_yticklabels(lime_feat_names, fontsize=10)
                axes[0, 1].set_xlabel('LIME Importance')
                axes[0, 1].set_title('LIME Feature Importance', fontweight='bold')
                axes[0, 1].grid(True, alpha=0.3)
                axes[0, 1].axvline(x=0, color='black', linestyle='-', alpha=0.5)
            
            # Prediction probabilities
            class_names = ['No Diabetes', 'Pre-diabetes', 'Pre-diabetes/Borderline', 'Diabetes']
            probabilities = self.rf_model.predict_proba([patient_data])[0]
            
            bars = axes[1, 0].bar(class_names, probabilities, 
                                 color=['green', 'yellow', 'orange', 'red'], alpha=0.7)
            axes[1, 0].set_ylabel('Probability')
            axes[1, 0].set_title('Prediction Probabilities', fontweight='bold')
            axes[1, 0].tick_params(axis='x', rotation=45)
            
            # Add probability labels on bars
            for bar, prob in zip(bars, probabilities):
                height = bar.get_height()
                axes[1, 0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                               f'{prob:.2f}', ha='center', va='bottom')
            
            # Summary text
            summary_text = f"""Patient ID: {patient_id}
            
Prediction: {prediction}
Confidence: {confidence:.1f}%

Key Risk Factors:
• {features[0]}: {contributions[0]:+.3f}
• {features[1]}: {contributions[1]:+.3f}
• {features[2]}: {contributions[2]:+.3f}

Recommendation:
{self.get_recommendation(prediction, confidence)}"""
            
            axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes,
                           fontsize=11, verticalalignment='top', fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
            axes[1, 1].set_xlim(0, 1)
            axes[1, 1].set_ylim(0, 1)
            axes[1, 1].axis('off')
            axes[1, 1].set_title('Patient Summary', fontweight='bold')
            
            plt.suptitle(f'Diabetes Risk Explanation - Patient {patient_id}', 
                        fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save visualization
            import os
            os.makedirs('explanations/patients', exist_ok=True)
            plt.savefig(f'explanations/patients/patient_{patient_id}_explanation.png', 
                       dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            print(f"❌ Error creating visualizations: {e}")
    
    def provide_clinical_interpretation(self, top_features, prediction):
        """Provide clinical interpretation of the results"""
        print(f"\n🏥 CLINICAL INTERPRETATION:")
        print("-" * 40)
        
        # Risk factor interpretation
        high_risk_factors = [f for f, v in top_features if v > 0]
        protective_factors = [f for f, v in top_features if v < 0]
        
        if high_risk_factors:
            print("⚠️ RISK-INCREASING FACTORS:")
            for factor in high_risk_factors:
                interpretation = self.interpret_feature(factor, positive=True)
                print(f"   • {interpretation}")
        
        if protective_factors:
            print("✅ PROTECTIVE FACTORS:")
            for factor in protective_factors:
                interpretation = self.interpret_feature(factor, positive=False)
                print(f"   • {interpretation}")
        
        # Overall recommendation
        recommendation = self.get_recommendation(prediction, 0)
        print(f"\n💡 RECOMMENDATION:")
        print(f"   {recommendation}")
    
    def interpret_feature(self, feature_name, positive=True):
        """Provide clinical interpretation of features"""
        interpretations = {
            'BMI': 'Body Mass Index - weight status indicator',
            'AGE': 'Age - older age increases diabetes risk',
            'CVDINFR4': 'History of heart attack',
            'CVDCRHD4': 'History of coronary heart disease',
            'TOTINDA': 'Physical activity level',
            'SMOKE100': 'Smoking history',
            'HAVARTH4': 'Arthritis diagnosis',
            '_RFBMI5': 'BMI risk category',
            'CHCKDNY2': 'Kidney disease',
            '_PACAT1': 'Physical activity category'
        }
        
        base_interpretation = interpretations.get(feature_name, feature_name)
        direction = "contributes to higher" if positive else "reduces"
        
        return f"{base_interpretation} {direction} diabetes risk"
    
    def get_recommendation(self, prediction, confidence):
        """Get clinical recommendation based on prediction"""
        recommendations = {
            'No Diabetes': "Continue healthy lifestyle habits. Regular checkups recommended.",
            'Pre-diabetes': "Lifestyle modifications recommended. Increase physical activity and improve diet. Regular monitoring needed.",
            'Pre-diabetes/Borderline': "Medical consultation advised. Lifestyle changes and possible medication may be needed.",
            'Diabetes': "Medical attention required. Comprehensive diabetes management plan needed including medication, diet, and monitoring."
        }
        
        return recommendations.get(prediction, "Consult healthcare provider for personalized advice.")

def demo_patient_explanations():
    """Demonstrate patient explanations with sample cases"""
    print("🚀 Starting Patient Explanation Demo...")
    
    # Initialize explainer
    explainer = PatientExplainer()
    
    # Demo with different risk levels
    risk_levels = ['low_risk', 'moderate_risk', 'high_risk']
    
    for risk_level in risk_levels:
        print(f"\n{'='*80}")
        patient_data = explainer.create_sample_patient(risk_level)
        explainer.explain_patient(patient_data, patient_id=f"{risk_level.replace('_', '-')}-demo")
        
        input("Press Enter to continue to next patient...")

if __name__ == "__main__":
    demo_patient_explanations()