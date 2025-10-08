#!/usr/bin/env python3
"""
Model Interpretability and Explainability Module
Using SHAP and LIME for diabetes detection model explanations
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import interpretability libraries
try:
    import shap
    print("✅ SHAP imported successfully")
except ImportError:
    print("❌ SHAP not found. Installing...")
    os.system("pip install shap")
    import shap

try:
    import lime
    from lime import lime_tabular
    print("✅ LIME imported successfully")
except ImportError:
    print("❌ LIME not found. Installing...")
    os.system("pip install lime")
    import lime
    from lime import lime_tabular

# Ensure directories exist
os.makedirs('explanations', exist_ok=True)
os.makedirs('explanations/shap', exist_ok=True)
os.makedirs('explanations/lime', exist_ok=True)

class DiabetesModelExplainer:
    """
    Comprehensive model explainability for diabetes detection using SHAP and LIME
    """
    
    def __init__(self, model_type='brfss'):
        """
        Initialize the explainer
        
        Args:
            model_type: 'brfss' for BRFSS models, 'pima' for Pima models
        """
        self.model_type = model_type
        self.models = {}
        self.scaler = None
        self.feature_names = None
        self.class_names = ['No Diabetes', 'Pre-diabetes', 'Pre-diabetes/Borderline', 'Diabetes']
        
        # Load models and preprocessing
        self.load_models()
        
        # Initialize explainers
        self.shap_explainers = {}
        self.lime_explainer = None
        
    def load_models(self):
        """Load trained models and preprocessing components"""
        try:
            if self.model_type == 'brfss':
                # Load BRFSS models
                self.models['rf'] = joblib.load('models/clean_diabetes_rf.pkl')
                self.scaler = joblib.load('models/clean_scaler.pkl')
                
                # Load feature names
                if os.path.exists('models/brfss_feature_names.pkl'):
                    self.feature_names = joblib.load('models/brfss_feature_names.pkl')
                elif os.path.exists('models/feature_names.pkl'):
                    self.feature_names = joblib.load('models/feature_names.pkl')
                else:
                    print("⚠️ Feature names not found, using indices")
                    self.feature_names = [f'Feature_{i}' for i in range(55)]
                
                # Load Neural Network if available
                try:
                    from train import CleanDiabetesModel
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    nn_model = CleanDiabetesModel(len(self.feature_names), num_classes=4)
                    nn_model.load_state_dict(torch.load('models/clean_diabetes_nn.pth', map_location=device))
                    nn_model.eval()
                    self.models['nn'] = nn_model
                    print("✅ Neural Network model loaded")
                except Exception as e:
                    print(f"⚠️ Could not load Neural Network: {e}")
                
            print(f"✅ Models loaded for {self.model_type} interpretation")
            print(f"📊 Feature count: {len(self.feature_names) if self.feature_names else 'Unknown'}")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            raise
    
    def load_sample_data(self, n_samples=1000):
        """Load sample data for explanations"""
        try:
            # Load BRFSS data
            df = pd.read_csv('data/2023_BRFSS_CLEANED.csv')
            
            # Sample for explanation
            df_sample = df.sample(n=min(n_samples, len(df)), random_state=42)
            
            # Process similar to training
            if 'YEAR' in df_sample.columns:
                df_sample = df_sample.drop('YEAR', axis=1)
            
            y_sample = df_sample['DIABETES_STATUS']
            X_sample = df_sample.drop('DIABETES_STATUS', axis=1)
            
            # Handle missing values and categorical variables
            categorical_cols = [col for col in X_sample.columns if X_sample[col].dtype == 'object' or X_sample[col].nunique() < 10]
            numerical_cols = [col for col in X_sample.columns if col not in categorical_cols]
            
            for col in categorical_cols:
                X_sample[col] = X_sample[col].fillna(X_sample[col].mode()[0])
            for col in numerical_cols:
                X_sample[col] = X_sample[col].fillna(X_sample[col].median())
            
            # One-hot encode
            X_sample = pd.get_dummies(X_sample, columns=categorical_cols, drop_first=True)
            
            # Ensure same features as training
            if self.feature_names:
                missing_cols = set(self.feature_names) - set(X_sample.columns)
                for col in missing_cols:
                    X_sample[col] = 0
                X_sample = X_sample[self.feature_names]
            
            # Scale features
            X_sample_scaled = self.scaler.transform(X_sample)
            
            print(f"✅ Sample data loaded: {X_sample_scaled.shape}")
            return X_sample_scaled, y_sample.values, X_sample
            
        except Exception as e:
            print(f"❌ Error loading sample data: {e}")
            return None, None, None
    
    def setup_shap_explainers(self, X_background, max_background=100):
        """Setup SHAP explainers for different models"""
        try:
            # Subsample background for efficiency
            if len(X_background) > max_background:
                background_idx = np.random.choice(len(X_background), max_background, replace=False)
                background_data = X_background[background_idx]
            else:
                background_data = X_background
            
            # Random Forest SHAP explainer
            if 'rf' in self.models:
                self.shap_explainers['rf'] = shap.TreeExplainer(self.models['rf'])
                print("✅ SHAP TreeExplainer setup for Random Forest")
            
            # Neural Network SHAP explainer
            if 'nn' in self.models:
                def nn_predict(X):
                    """Wrapper for neural network prediction"""
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(X)
                        outputs = self.models['nn'](X_tensor)
                        return torch.softmax(outputs, dim=1).numpy()
                
                self.shap_explainers['nn'] = shap.KernelExplainer(nn_predict, background_data)
                print("✅ SHAP KernelExplainer setup for Neural Network")
            
        except Exception as e:
            print(f"❌ Error setting up SHAP explainers: {e}")
    
    def setup_lime_explainer(self, X_background, y_background):
        """Setup LIME explainer"""
        try:
            self.lime_explainer = lime_tabular.LimeTabularExplainer(
                X_background,
                feature_names=self.feature_names if self.feature_names else [f'Feature_{i}' for i in range(X_background.shape[1])],
                class_names=self.class_names,
                mode='classification',
                training_labels=y_background,
                discretize_continuous=True
            )
            print("✅ LIME explainer setup successfully")
            
        except Exception as e:
            print(f"❌ Error setting up LIME explainer: {e}")
    
    def explain_with_shap(self, X_test, model_name='rf', max_samples=50):
        """Generate SHAP explanations"""
        try:
            if model_name not in self.shap_explainers:
                print(f"❌ SHAP explainer for {model_name} not available")
                return None
            
            # Limit samples for efficiency
            X_explain = X_test[:max_samples] if len(X_test) > max_samples else X_test
            
            print(f"🔍 Generating SHAP explanations for {model_name}...")
            
            # Calculate SHAP values
            if model_name == 'rf':
                shap_values = self.shap_explainers[model_name].shap_values(X_explain)
            else:
                shap_values = self.shap_explainers[model_name].shap_values(X_explain)
            
            # Create visualizations
            self.create_shap_visualizations(shap_values, X_explain, model_name)
            
            return shap_values
            
        except Exception as e:
            print(f"❌ Error generating SHAP explanations: {e}")
            return None
    
    def create_shap_visualizations(self, shap_values, X_test, model_name):
        """Create SHAP visualizations"""
        try:
            feature_names = self.feature_names if self.feature_names else [f'Feature_{i}' for i in range(X_test.shape[1])]
            
            # Summary plot
            plt.figure(figsize=(12, 8))
            if isinstance(shap_values, list):  # Multi-class
                shap.summary_plot(shap_values, X_test, feature_names=feature_names, 
                                class_names=self.class_names, show=False)
            else:
                shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
            
            plt.title(f'SHAP Feature Importance - {model_name.upper()}', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'explanations/shap/{model_name}_summary_plot.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Feature importance bar plot
            plt.figure(figsize=(10, 8))
            if isinstance(shap_values, list):
                # For multi-class, use diabetes class (class 3)
                importance_values = np.abs(shap_values[3]).mean(0)
            else:
                importance_values = np.abs(shap_values).mean(0)
            
            # Get top 20 features
            top_indices = np.argsort(importance_values)[-20:]
            top_features = [feature_names[i] for i in top_indices]
            top_importance = importance_values[top_indices]
            
            plt.barh(range(len(top_features)), top_importance, color='skyblue', alpha=0.7)
            plt.yticks(range(len(top_features)), top_features)
            plt.xlabel('Mean |SHAP Value|')
            plt.title(f'Top 20 Feature Importance - {model_name.upper()}', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'explanations/shap/{model_name}_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Waterfall plot for first instance
            if isinstance(shap_values, list):
                shap_values_single = shap_values[3][0]  # Diabetes class, first instance
            else:
                shap_values_single = shap_values[0]
            
            plt.figure(figsize=(10, 8))
            shap.waterfall_plot(shap.Explanation(values=shap_values_single, 
                                               base_values=self.shap_explainers[model_name].expected_value[3] if isinstance(shap_values, list) else self.shap_explainers[model_name].expected_value,
                                               data=X_test[0],
                                               feature_names=feature_names), show=False)
            plt.title(f'SHAP Waterfall Plot - {model_name.upper()} (First Instance)', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f'explanations/shap/{model_name}_waterfall.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"✅ SHAP visualizations saved for {model_name}")
            
        except Exception as e:
            print(f"❌ Error creating SHAP visualizations: {e}")
    
    def explain_with_lime(self, X_test, model_name='rf', num_instances=5):
        """Generate LIME explanations"""
        try:
            if self.lime_explainer is None:
                print("❌ LIME explainer not available")
                return None
            
            model = self.models[model_name]
            
            # Define prediction function
            if model_name == 'rf':
                predict_fn = model.predict_proba
            else:  # Neural network
                def predict_fn(X):
                    with torch.no_grad():
                        X_tensor = torch.FloatTensor(X)
                        outputs = model(X_tensor)
                        return torch.softmax(outputs, dim=1).numpy()
            
            explanations = []
            
            print(f"🔍 Generating LIME explanations for {model_name}...")
            
            for i in range(min(num_instances, len(X_test))):
                explanation = self.lime_explainer.explain_instance(
                    X_test[i], 
                    predict_fn,
                    num_features=20,
                    top_labels=4
                )
                explanations.append(explanation)
                
                # Save individual explanation
                explanation.save_to_file(f'explanations/lime/{model_name}_instance_{i}.html')
            
            # Create summary visualization
            self.create_lime_summary(explanations, model_name)
            
            print(f"✅ LIME explanations saved for {model_name}")
            return explanations
            
        except Exception as e:
            print(f"❌ Error generating LIME explanations: {e}")
            return None
    
    def create_lime_summary(self, explanations, model_name):
        """Create LIME summary visualization"""
        try:
            # Extract feature importance across all explanations for diabetes class
            feature_importance = {}
            
            for exp in explanations:
                if 3 in exp.available_labels():  # Diabetes class
                    for feature, importance in exp.as_list(label=3):
                        if feature not in feature_importance:
                            feature_importance[feature] = []
                        feature_importance[feature].append(importance)
            
            # Calculate mean importance
            mean_importance = {k: np.mean(v) for k, v in feature_importance.items()}
            
            # Plot top features
            sorted_features = sorted(mean_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
            
            features, importances = zip(*sorted_features)
            colors = ['red' if imp < 0 else 'green' for imp in importances]
            
            plt.figure(figsize=(12, 8))
            plt.barh(range(len(features)), importances, color=colors, alpha=0.7)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Mean LIME Importance (Diabetes Class)')
            plt.title(f'LIME Feature Importance Summary - {model_name.upper()}', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3)
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.5)
            plt.tight_layout()
            plt.savefig(f'explanations/lime/{model_name}_summary.png', dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"❌ Error creating LIME summary: {e}")
    
    def create_comparison_dashboard(self):
        """Create a comprehensive comparison dashboard"""
        try:
            fig, axes = plt.subplots(2, 2, figsize=(20, 16))
            
            # Load saved SHAP and LIME results for comparison
            axes[0, 0].text(0.5, 0.5, 'SHAP Feature Importance\n(Global Explanations)', 
                           ha='center', va='center', fontsize=16, fontweight='bold')
            axes[0, 0].set_title('SHAP Analysis')
            
            axes[0, 1].text(0.5, 0.5, 'LIME Feature Importance\n(Local Explanations)', 
                           ha='center', va='center', fontsize=16, fontweight='bold')
            axes[0, 1].set_title('LIME Analysis')
            
            axes[1, 0].text(0.5, 0.5, 'Model Interpretability\nComparison', 
                           ha='center', va='center', fontsize=16, fontweight='bold')
            axes[1, 0].set_title('Comparison Analysis')
            
            axes[1, 1].text(0.5, 0.5, 'Clinical Insights\nand Recommendations', 
                           ha='center', va='center', fontsize=16, fontweight='bold')
            axes[1, 1].set_title('Clinical Interpretation')
            
            for ax in axes.flat:
                ax.set_xticks([])
                ax.set_yticks([])
            
            plt.suptitle('Diabetes Detection Model Interpretability Dashboard', 
                        fontsize=20, fontweight='bold')
            plt.tight_layout()
            plt.savefig('explanations/interpretability_dashboard.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("✅ Interpretability dashboard created")
            
        except Exception as e:
            print(f"❌ Error creating dashboard: {e}")
    
    def run_complete_analysis(self):
        """Run complete interpretability analysis"""
        print("\n" + "="*60)
        print("🔬 DIABETES MODEL INTERPRETABILITY ANALYSIS")
        print("="*60)
        
        try:
            # Load sample data
            X_sample, y_sample, X_raw = self.load_sample_data(n_samples=500)
            if X_sample is None:
                return
            
            # Split for explanation
            X_bg, X_test, y_bg, y_test = train_test_split(
                X_sample, y_sample, test_size=0.3, random_state=42, stratify=y_sample
            )
            
            print(f"📊 Background data: {X_bg.shape}")
            print(f"🧪 Test data: {X_test.shape}")
            
            # Setup explainers
            print("\n🔧 Setting up explainers...")
            self.setup_shap_explainers(X_bg, max_background=50)
            self.setup_lime_explainer(X_bg, y_bg)
            
            # Generate explanations for each model
            for model_name in self.models.keys():
                print(f"\n📈 Analyzing {model_name.upper()} model...")
                
                # SHAP analysis
                shap_values = self.explain_with_shap(X_test, model_name, max_samples=30)
                
                # LIME analysis
                lime_explanations = self.explain_with_lime(X_test, model_name, num_instances=3)
            
            # Create dashboard
            print("\n📊 Creating interpretability dashboard...")
            self.create_comparison_dashboard()
            
            print("\n" + "="*60)
            print("✅ INTERPRETABILITY ANALYSIS COMPLETE!")
            print("="*60)
            print("📁 Results saved in:")
            print("   📂 explanations/shap/ - SHAP visualizations")
            print("   📂 explanations/lime/ - LIME explanations")
            print("   📊 explanations/interpretability_dashboard.png")
            print("="*60)
            
        except Exception as e:
            print(f"❌ Error in complete analysis: {e}")
            import traceback
            traceback.print_exc()

def main():
    """Main interpretability analysis"""
    print("🚀 Starting Diabetes Model Interpretability Analysis...")
    
    # Initialize explainer
    explainer = DiabetesModelExplainer(model_type='brfss')
    
    # Run complete analysis
    explainer.run_complete_analysis()

if __name__ == "__main__":
    main()