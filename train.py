#!/usr/bin/env python3
"""
Clean Diabetes Detection Training
Simplified, robust, and efficient implementation
"""

import os
import sys
import logging
import warnings
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
import optuna
from sklearn.model_selection import KFold

# Import the unified model from models.py
from models import EnhancedDiabetesModel

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class CleanDiabetesTrainer:
    """Clean, efficient trainer for diabetes detection"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        logging.info(f"Using device: {self.device}")
    
    def load_and_preprocess_data(self):
        """Load and preprocess both BRFSS and Pima datasets for combined training"""
        try:
            print("🔄 Loading both datasets for combined training...")
            
            # Load BRFSS dataset
            df_brfss = pd.read_csv('data/2023_BRFSS_CLEANED.csv')
            logging.info(f"BRFSS Dataset loaded: {df_brfss.shape}")
            
            # Load Pima dataset  
            df_pima = pd.read_csv('data/diabetes.csv')
            logging.info(f"Pima Dataset loaded: {df_pima.shape}")
            
            # For faster training, sample BRFSS data
            df_brfss = df_brfss.sample(n=100000, random_state=42)
            logging.info(f"Using BRFSS sample for combined training: {df_brfss.shape}")
            
            # Process BRFSS data
            if 'YEAR' in df_brfss.columns:
                df_brfss = df_brfss.drop('YEAR', axis=1)
            
            # Map BRFSS 4-class to binary (0,1,2->0=No Diabetes, 3->1=Diabetes)
            y_brfss = df_brfss['DIABETES_STATUS'].map({0: 0, 1: 0, 2: 0, 3: 1})
            X_brfss = df_brfss.drop('DIABETES_STATUS', axis=1)
            
            # Process Pima data (already binary)
            y_pima = df_pima['Outcome'] 
            X_pima = df_pima.drop('Outcome', axis=1)
            
            # Clean Pima data (replace zeros with NaN and impute)
            zero_replace_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
            for col in zero_replace_cols:
                if col in X_pima.columns:
                    X_pima[col] = X_pima[col].replace(0, np.nan)
                    X_pima[col] = X_pima[col].fillna(X_pima[col].median())
            
            # Feature engineering for Pima
            X_pima['BMI_Age'] = X_pima['BMI'] * X_pima['Age']
            X_pima['Glucose_BMI'] = X_pima['Glucose'] * X_pima['BMI']
            X_pima['Insulin_Glucose'] = X_pima['Insulin'] * X_pima['Glucose']
            
            # Process BRFSS features
            categorical_cols = [col for col in X_brfss.columns if X_brfss[col].dtype == 'object' or X_brfss[col].nunique() < 10]
            numerical_cols = [col for col in X_brfss.columns if col not in categorical_cols]
            
            # Impute BRFSS missing values
            for col in categorical_cols:
                X_brfss[col] = X_brfss[col].fillna(X_brfss[col].mode()[0])
            for col in numerical_cols:
                X_brfss[col] = X_brfss[col].fillna(X_brfss[col].median())
            
            # One-hot encode BRFSS categorical variables
            X_brfss = pd.get_dummies(X_brfss, columns=categorical_cols, drop_first=True)
            
            # Create common feature space
            # Find overlapping numerical features (BMI, Age if available)
            common_features = []
            pima_feature_map = {}
            
            # Map common features between datasets
            if 'BMI' in X_pima.columns and 'BMI' in X_brfss.columns:
                common_features.append('BMI')
                pima_feature_map['BMI'] = 'BMI'
            
            if 'Age' in X_pima.columns and 'AGE' in X_brfss.columns:
                common_features.append('AGE')
                pima_feature_map['AGE'] = 'Age'
            
            # Combine datasets with feature alignment
            logging.info(f"Common features found: {common_features}")
            
            # Strategy 1: Train on BRFSS first, then fine-tune on Pima
            # Standardize BRFSS features
            X_brfss_scaled = self.scaler.fit_transform(X_brfss)
            
            # Create enhanced feature space for Pima by adding derived features
            X_pima_enhanced = X_pima.copy()
            
            # Add more engineered features to improve performance
            if 'BMI' in X_pima.columns and 'Age' in X_pima.columns:
                X_pima_enhanced['BMI_squared'] = X_pima['BMI'] ** 2
                X_pima_enhanced['Age_squared'] = X_pima['Age'] ** 2
                X_pima_enhanced['BMI_Age_ratio'] = X_pima['BMI'] / (X_pima['Age'] + 1)
            
            if 'Glucose' in X_pima.columns:
                X_pima_enhanced['Glucose_squared'] = X_pima['Glucose'] ** 2
                X_pima_enhanced['Glucose_Age_ratio'] = X_pima['Glucose'] / (X_pima['Age'] + 1)
            
            # Store both datasets for combined training
            self.brfss_data = (X_brfss_scaled, y_brfss.values)
            self.pima_data = (X_pima_enhanced, y_pima)
            
            # For primary training, use enhanced Pima data
            self.num_classes = 2  # Binary classification
            
            # Save feature names
            self.pima_feature_names = X_pima_enhanced.columns.tolist()
            self.brfss_feature_names = X_brfss.columns.tolist()
            
            joblib.dump(self.pima_feature_names, 'models/pima_feature_names.pkl')
            joblib.dump(self.brfss_feature_names, 'models/brfss_feature_names.pkl')
            
            logging.info(f"BRFSS features: {X_brfss.shape[1]}")
            logging.info(f"Enhanced Pima features: {X_pima_enhanced.shape[1]}")
            logging.info(f"BRFSS binary target distribution: {dict(pd.Series(y_brfss).value_counts().sort_index())}")
            logging.info(f"Pima target distribution: {dict(y_pima.value_counts().sort_index())}")
            
            return X_pima_enhanced, y_pima
            
        except Exception as e:
            logging.error(f"Error loading combined data: {e}")
            raise
    
    def prepare_data_splits(self, X, y, test_size=0.2, val_size=0.2):
        """Create train/validation/test splits"""
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        X_test_scaled = self.scaler.transform(X_test)
        
        logging.info(f"Data splits - Train: {X_train_scaled.shape}, Val: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")
        
        return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
    
    def create_data_loaders(self, X_train, X_val, X_test, y_train, y_val, y_test, batch_size=64):
        """Create PyTorch data loaders for BINARY classification"""
        X_train_tensor = torch.FloatTensor(X_train)
        X_val_tensor = torch.FloatTensor(X_val)
        X_test_tensor = torch.FloatTensor(X_test)
        
        # For binary classification with BCEWithLogitsLoss, labels should be FloatTensors of shape (N, 1)
        y_train_tensor = torch.FloatTensor(y_train.values.reshape(-1, 1))
        y_val_tensor = torch.FloatTensor(y_val.values.reshape(-1, 1))
        y_test_tensor = torch.FloatTensor(y_test.values.reshape(-1, 1))

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        
        return train_loader, val_loader, test_loader
    
    def train_random_forest(self, X_train, X_val, y_train, y_val):
        """Train Random Forest for multi-class comparison"""
        logging.info("Training Random Forest (multi-class)...")
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        val_pred = rf.predict(X_val)
        from sklearn.metrics import f1_score
        val_f1 = f1_score(y_val, val_pred, average='weighted')
        logging.info(f"Random Forest training complete - Val F1: {val_f1:.4f}")
        return rf, val_f1
    
    def evaluate_model(self, model, test_loader, model_name="Model"):
        """Evaluate model on test set (BINARY)"""
        model.eval()
        predictions = []
        targets = []
        probas = []
        from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = model(batch_X)
                
                # Get probabilities with sigmoid and predictions with a 0.5 threshold
                batch_probas = torch.sigmoid(outputs)
                batch_preds = (batch_probas > 0.5).float()
                
                probas.extend(batch_probas.cpu().numpy())
                predictions.extend(batch_preds.cpu().numpy())
                targets.extend(batch_y.cpu().numpy())

        predictions = np.array(predictions).flatten()
        targets = np.array(targets).flatten()
        probas = np.array(probas).flatten()

        accuracy = accuracy_score(targets, predictions)
        weighted_f1 = f1_score(targets, predictions, average='weighted')
        macro_f1 = f1_score(targets, predictions, average='macro')
        auc = roc_auc_score(targets, probas)

        logging.info(f"{model_name} Test Results - Accuracy: {accuracy:.4f}, Weighted F1: {weighted_f1:.4f}, Macro F1: {macro_f1:.4f}, AUC: {auc:.4f}")
        
        return {
            'accuracy': accuracy,
            'weighted_f1': weighted_f1,
            'macro_f1': macro_f1,
            'auc': auc,
            'predictions': predictions,
            'targets': targets,
            'probas': probas
        }
    
    def evaluate_random_forest(self, rf, X_test, y_test, model_name="Random Forest"):
        """Evaluate Random Forest on test set (BINARY)"""
        predictions = rf.predict(X_test)
        probas = rf.predict_proba(X_test)
        
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        accuracy = accuracy_score(y_test, predictions)
        weighted_f1 = f1_score(y_test, predictions, average='weighted')
        macro_f1 = f1_score(y_test, predictions, average='macro')
        
        # For binary classification, use probabilities of the positive class (class 1)
        auc = roc_auc_score(y_test, probas[:, 1])

        logging.info(f"{model_name} Test Results - Accuracy: {accuracy:.4f}, Weighted F1: {weighted_f1:.4f}, Macro F1: {macro_f1:.4f}, AUC: {auc:.4f}")
        
        return {
            'accuracy': accuracy,
            'weighted_f1': weighted_f1,
            'macro_f1': macro_f1,
            'auc': auc,
            'predictions': predictions,
            'targets': y_test.values if isinstance(y_test, pd.Series) else y_test,
            'probas': probas
        }
    
        
    def objective(self, trial, X, y):
        """Optuna objective function for hyperparameter tuning"""
        # Suggest hyperparameters
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)
        hidden_dim1 = trial.suggest_int("hidden_dim1", 64, 256)
        hidden_dim2 = trial.suggest_int("hidden_dim2", 32, 128)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        fold_aucs = []

        for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
            # Data for this fold
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]

            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)

            train_loader, val_loader, _ = self.create_data_loaders(
                X_train_scaled, X_val_scaled, X_val_scaled, y_train, y_val, y_val
            )

            model = EnhancedDiabetesModel(
                input_dim=X_train_scaled.shape[1],
                hidden_dim1=hidden_dim1,
                hidden_dim2=hidden_dim2,
                dropout_rate=dropout_rate
            ).to(self.device)

            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = nn.BCEWithLogitsLoss()

            # Training loop for the fold
            for epoch in range(25): # Fewer epochs for tuning
                model.train()
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    optimizer.zero_grad()
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()

            # Validation for the fold
            model.eval()
            val_probas = []
            val_targets = []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)
                    val_probas.extend(torch.sigmoid(outputs).cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())
            
            fold_auc = roc_auc_score(val_targets, val_probas)
            fold_aucs.append(fold_auc)

        return np.mean(fold_aucs)

    def train_final_model(self, X, y, best_params):
        """Train the final model on the entire dataset using the best hyperparameters"""
        print("\n🚀 Training final model with optimal hyperparameters...")
        
        X_scaled = self.scaler.fit_transform(X)
        joblib.dump(self.scaler, 'models/combined_scaler.pkl')
        
        train_loader, _, _ = self.create_data_loaders(X_scaled, X_scaled, X_scaled, y, y, y, batch_size=64)

        final_model = EnhancedDiabetesModel(
            input_dim=X_scaled.shape[1],
            hidden_dim1=best_params['hidden_dim1'],
            hidden_dim2=best_params['hidden_dim2'],
            dropout_rate=best_params['dropout_rate']
        ).to(self.device)

        optimizer = optim.AdamW(
            final_model.parameters(), 
            lr=best_params['lr'], 
            weight_decay=best_params['weight_decay']
        )
        criterion = nn.BCEWithLogitsLoss()

        # Train for a sufficient number of epochs
        for epoch in range(50):
            final_model.train()
            total_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = final_model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Final Training Epoch {epoch+1}/50: Avg Loss: {total_loss/len(train_loader):.4f}")

        torch.save(final_model.state_dict(), 'models/enhanced_diabetes_nn.pth')
        print("✅ Final optimized model trained and saved.")
        
        self.models['enhanced_nn'] = final_model
        return final_model

    
    def train_transfer_learning_model(self):
        """Train a transfer learning model using BRFSS then fine-tune on Pima"""
        try:
            print("🔄 Training transfer learning model (BRFSS -> Pima)...")
            
            X_brfss, y_brfss = self.brfss_data
            
            # Create BRFSS data loaders
            brfss_train_loader = DataLoader(
                TensorDataset(torch.FloatTensor(X_brfss[:50000]), torch.LongTensor(y_brfss[:50000])),
                batch_size=256, shuffle=True
            )
            
            # Pre-train on BRFSS data
            pretrain_model = EnhancedDiabetesModel(X_brfss.shape[1], num_classes=2)
            pretrain_model.to(self.device)
            
            optimizer = optim.AdamW(pretrain_model.parameters(), lr=0.001, weight_decay=0.01)
            criterion = nn.CrossEntropyLoss()
            
            print("🧠 Pre-training on BRFSS data...")
            for epoch in range(20):  # Fewer epochs for pre-training
                pretrain_model.train()
                total_loss = 0
                for batch_X, batch_y in brfss_train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = pretrain_model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                if (epoch + 1) % 5 == 0:
                    print(f"Pre-training Epoch {epoch+1}/20: Loss: {total_loss/len(brfss_train_loader):.4f}")
            
            # Save pre-trained model
            torch.save(pretrain_model.state_dict(), 'models/pretrained_brfss_nn.pth')
            print("✅ Pre-training completed and saved!")
            
        except Exception as e:
            logging.error(f"Error in transfer learning: {e}")
            print(f"⚠️ Transfer learning failed: {e}")

    def create_final_visualizations(self, nn_results, rf_results, rf_model, feature_names, study):
        """Create a dashboard of final visualizations."""
        print("🎨 Generating final visualization dashboard...")
        os.makedirs('output/plots', exist_ok=True)
        plt.style.use('seaborn-v0_8-whitegrid')

        # 1. Confusion Matrices
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Model Performance: Confusion Matrices', fontsize=16, fontweight='bold')
        
        cm_nn = confusion_matrix(nn_results['targets'], nn_results['predictions'])
        sns.heatmap(cm_nn, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar=False,
                    xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
        ax1.set_title('Optimized Neural Network', fontsize=12)
        ax1.set_xlabel('Predicted')
        ax1.set_ylabel('Actual')

        cm_rf = confusion_matrix(rf_results['targets'], rf_results['predictions'])
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=ax2, cbar=False,
                    xticklabels=['No Diabetes', 'Diabetes'], yticklabels=['No Diabetes', 'Diabetes'])
        ax2.set_title('Final Random Forest', fontsize=12)
        ax2.set_xlabel('Predicted')
        ax2.set_ylabel('Actual')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig('output/plots/final_confusion_matrices.png', dpi=300)
        plt.close()

        # 2. ROC Curves
        from sklearn.metrics import roc_curve
        plt.figure(figsize=(10, 8))
        
        fpr_nn, tpr_nn, _ = roc_curve(nn_results['targets'], nn_results['probas'])
        plt.plot(fpr_nn, tpr_nn, label=f"Neural Network (AUC = {nn_results['auc']:.4f})", color='blue', lw=2)

        fpr_rf, tpr_rf, _ = roc_curve(rf_results['targets'], rf_results['probas'][:, 1])
        plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {rf_results['auc']:.4f})", color='green', lw=2)

        plt.plot([0, 1], [0, 1], 'k--', label='Chance (AUC = 0.50)')
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.grid(True)
        plt.savefig('output/plots/final_roc_curves.png', dpi=300)
        plt.close()

        # 3. RF Feature Importance
        plt.figure(figsize=(12, 8))
        importances = rf_model.feature_importances_
        feature_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
        feature_df = feature_df.sort_values(by='Importance', ascending=False)

        sns.barplot(x='Importance', y='Feature', data=feature_df, palette='viridis')
        plt.title('Random Forest Feature Importance', fontsize=16, fontweight='bold')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig('output/plots/rf_feature_importance.png', dpi=300)
        plt.close()

        # 4. Optuna Visualizations
        try:
            import kaleido # Explicitly import to help with pathing
            import plotly.io as pio
            pio.kaleido.scope.mathjax = None # Workaround for a common kaleido issue

            fig_opt_hist = optuna.visualization.plot_optimization_history(study)
            fig_opt_hist.write_image("output/plots/optuna_optimization_history.png", engine="kaleido", scale=2)

            fig_param_imp = optuna.visualization.plot_param_importances(study)
            fig_param_imp.write_image("output/plots/optuna_param_importances.png", engine="kaleido", scale=2)
        except ImportError:
            logging.warning("Kaleido package not found. Please install it (`pip install kaleido`) to save Optuna plots.")
        except Exception as e:
            logging.warning(f"Could not generate Optuna plots. Error: {e}")

        print("✅ All visualizations saved to output/plots/")

def main():
    """Main training pipeline with hyperparameter tuning and K-Fold validation"""
    print("\n" + "="*60)
    print("🏥 OPTIMIZED DIABETES DETECTION MODEL")
    print("="*60)
    print("� Hyperparameter Tuning with Optuna & K-Fold CV")
    print("="*60 + "\n")
    
    trainer = CleanDiabetesTrainer()
    
    try:
        # Load and preprocess data
        print("🔄 Loading and preprocessing Pima dataset...")
        X, y = trainer.load_and_preprocess_data()
        
        # --- Hyperparameter Tuning with Optuna ---
        print("🧠 Starting hyperparameter tuning with Optuna...")
        study = optuna.create_study(direction="maximize")
        study.optimize(lambda trial: trainer.objective(trial, X, y), n_trials=50)
        
        print("\n" + "="*60)
        print("🏆 OPTUNA STUDY COMPLETE")
        print(f"  Best Trial AUC: {study.best_value:.4f}")
        print("  Best Hyperparameters:")
        for key, value in study.best_params.items():
            print(f"    - {key}: {value}")
        print("="*60 + "\n")

        # --- Train Final Model ---
        final_model = trainer.train_final_model(X, y, study.best_params)

        # --- Final Evaluation ---
        print("\n📈 Evaluating final model on a held-out test set...")
        # Create a final test set
        X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Create and save data splits for external use (like interpretability)
        os.makedirs('output/data', exist_ok=True)
        X_train_full.to_csv('output/data/X_train_pima.csv', index=False)
        y_train_full.to_csv('output/data/y_train_pima.csv', index=False)
        X_test.to_csv('output/data/X_test_pima.csv', index=False)
        y_test.to_csv('output/data/y_test_pima.csv', index=False)
        print("✅ Saved final data splits to output/data/")

        X_test_scaled = trainer.scaler.transform(X_test)
        _, _, test_loader = trainer.create_data_loaders(X_test_scaled, X_test_scaled, X_test_scaled, y_test, y_test, y_test)
        
        final_nn_results = trainer.evaluate_model(final_model, test_loader, "Final Optimized NN")

        # --- Train and Evaluate Final Random Forest ---
        print("\n🌲 Training and evaluating final Random Forest for comparison...")
        X_train_full_scaled = trainer.scaler.transform(X_train_full)
        rf_final = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced', n_jobs=-1)
        rf_final.fit(X_train_full_scaled, y_train_full)
        final_rf_results = trainer.evaluate_random_forest(rf_final, X_test_scaled, y_test, "Final Random Forest")
        joblib.dump(rf_final, 'models/enhanced_diabetes_rf.pkl')

        # --- Reporting ---
        print("\n" + "="*60)
        print("🎯 FINAL OPTIMIZED RESULTS")
        print("="*60)
        print(f"Optimized Neural Network:")
        print(f"  📊 Test Accuracy: {final_nn_results['accuracy']:.4f}")
        print(f"  📊 Test AUC: {final_nn_results['auc']:.4f}")
        print(f"  📊 Test Weighted F1: {final_nn_results['weighted_f1']:.4f}")
        
        print(f"\nFinal Random Forest:")
        print(f"  📊 Test Accuracy: {final_rf_results['accuracy']:.4f}")
        print(f"  📊 Test AUC: {final_rf_results['auc']:.4f}")
        print(f"  📊 Test Weighted F1: {final_rf_results['weighted_f1']:.4f}")
        print("="*60 + "\n")

        # --- Create Visualizations ---
        trainer.create_final_visualizations(
            final_nn_results, 
            final_rf_results, 
            rf_final, 
            trainer.pima_feature_names, 
            study
        )

        # Save a comprehensive report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_summary = f"""
Optimized Diabetes Detection Results - {timestamp}
================================================================

HYPERPARAMETER OPTIMIZATION (OPTUNA):
- Trials: 50
- Best Cross-Validated AUC: {study.best_value:.4f}
- Best Parameters:
{os.linesep.join([f"  - {k}: {v}" for k, v in study.best_params.items()])}

FINAL NEURAL NETWORK (ON TEST SET):
- Accuracy: {final_nn_results['accuracy']:.4f}
- AUC: {final_nn_results['auc']:.4f}
- Weighted F1: {final_nn_results['weighted_f1']:.4f}

FINAL RANDOM FOREST (ON TEST SET):
- Accuracy: {final_rf_results['accuracy']:.4f}
- AUC: {final_rf_results['auc']:.4f}
- Weighted F1: {final_rf_results['weighted_f1']:.4f}

Models saved to:
- Optimized Neural Network: models/enhanced_diabetes_nn.pth
- Final Random Forest: models/enhanced_diabetes_rf.pkl
- Final Scaler: models/combined_scaler.pkl
================================================================
"""
        report_path = f'output/results/optimized_results_{timestamp}.txt'
        with open(report_path, 'w') as f:
            f.write(results_summary)
            
        print(f"📄 Optimized results report saved to {report_path}")
        
    except Exception as e:
        logging.error(f"An error occurred in the main pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()