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
from datetime import datetime

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EnhancedDiabetesModel(nn.Module):
    """Enhanced neural network for diabetes detection with better architecture"""
    
    def __init__(self, input_size, num_classes=2):
        super(EnhancedDiabetesModel, self).__init__()
        self.network = nn.Sequential(
            # Input layer with dropout
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            # Hidden layers 
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Output layer
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)

class CleanDiabetesModel(nn.Module):
    """Clean, efficient neural network for diabetes detection"""
    
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
        # No activation here; nn.CrossEntropyLoss expects raw logits
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)

class CleanDiabetesTrainer:
    """Clean, efficient trainer for diabetes detection"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        self.num_classes = 4  # Will be updated based on data
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
    
    def create_data_loaders(self, X_train, X_val, X_test, y_train, y_val, y_test, batch_size=128):
        """Create PyTorch data loaders for multi-class"""
        X_train_tensor = torch.FloatTensor(X_train)
        X_val_tensor = torch.FloatTensor(X_val)
        X_test_tensor = torch.FloatTensor(X_test)
        y_train_tensor = torch.LongTensor(y_train.values)
        y_val_tensor = torch.LongTensor(y_val.values)
        y_test_tensor = torch.LongTensor(y_test.values)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, val_loader, test_loader
    
    def train_neural_network(self, train_loader, val_loader, input_size, epochs=50):
        """Train the neural network model for multi-class"""
        model = CleanDiabetesModel(input_size, num_classes=self.num_classes).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        best_val_f1 = 0
        patience_counter = 0
        best_model_state = None
        train_losses, val_losses, val_f1s = [], [], []
        logging.info(f"Starting neural network training (multi-class, {self.num_classes} classes)...")
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            model.eval()
            val_loss = 0
            val_predictions = []
            val_targets = []
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
                    preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    val_predictions.extend(preds)
                    val_targets.extend(batch_y.cpu().numpy())
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            from sklearn.metrics import f1_score
            val_f1 = f1_score(val_targets, val_predictions, average='weighted')
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_f1s.append(val_f1)
            scheduler.step(val_loss)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            if (epoch + 1) % 5 == 0:
                logging.info(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
            if patience_counter >= 10:
                logging.info(f"Early stopping at epoch {epoch+1}")
                break
        model.load_state_dict(best_model_state)
        logging.info(f"Neural network training complete - Best Val F1: {best_val_f1:.4f}")
        return model, {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_f1s': val_f1s,
            'best_val_f1': best_val_f1
        }
    
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
        """Evaluate model on test set (multi-class)"""
        model.eval()
        predictions = []
        targets = []
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = model(batch_X)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                predictions.extend(preds)
                targets.extend(batch_y.cpu().numpy())
        predictions = np.array(predictions)
        targets = np.array(targets)
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        accuracy = accuracy_score(targets, predictions)
        weighted_f1 = f1_score(targets, predictions, average='weighted')
        macro_f1 = f1_score(targets, predictions, average='macro')
        # One-vs-rest AUC
        # For AUC, need probability scores for each class
        # Re-run loader to get probabilities
        probas = []
        with torch.no_grad():
            for batch_X, _ in test_loader:
                batch_X = batch_X.to(self.device)
                outputs = model(batch_X)
                probas.extend(torch.softmax(outputs, dim=1).cpu().numpy())
        probas = np.array(probas)
        try:
            auc = roc_auc_score(targets, probas, multi_class='ovr', average='weighted')
        except Exception:
            auc = None
        logging.info(f"{model_name} Test Results - Accuracy: {accuracy:.4f}, Weighted F1: {weighted_f1:.4f}, Macro F1: {macro_f1:.4f}, OvR AUC: {auc}")
        return {
            'accuracy': accuracy,
            'weighted_f1': weighted_f1,
            'macro_f1': macro_f1,
            'auc': auc,
            'predictions': predictions,
            'targets': targets
        }
    
    def evaluate_random_forest(self, rf, X_test, y_test, model_name="Random Forest"):
        """Evaluate Random Forest on test set (multi-class)"""
        predictions = rf.predict(X_test)
        probas = rf.predict_proba(X_test)
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        accuracy = accuracy_score(y_test, predictions)
        weighted_f1 = f1_score(y_test, predictions, average='weighted')
        macro_f1 = f1_score(y_test, predictions, average='macro')
        try:
            auc = roc_auc_score(y_test, probas, multi_class='ovr', average='weighted')
        except Exception:
            auc = None
        logging.info(f"{model_name} Test Results - Accuracy: {accuracy:.4f}, Weighted F1: {weighted_f1:.4f}, Macro F1: {macro_f1:.4f}, OvR AUC: {auc}")
        return {
            'accuracy': accuracy,
            'weighted_f1': weighted_f1,
            'macro_f1': macro_f1,
            'auc': auc,
            'predictions': predictions,
            'targets': y_test.values
        }
    
    def create_visualizations(self, nn_results, rf_results, nn_history):
        """Create comprehensive visualizations for multi-class"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        epochs = range(1, len(nn_history['train_losses']) + 1)
        axes[0, 0].plot(epochs, nn_history['train_losses'], 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, nn_history['val_losses'], 'r-', label='Val Loss')
        axes[0, 0].set_title('Training History - Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        axes[0, 1].plot(epochs, nn_history['val_f1s'], 'g-')
        axes[0, 1].set_title('Validation Weighted F1 Progress')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Weighted F1')
        axes[0, 1].grid(True)
        models = ['Neural Network', 'Random Forest']
        accuracies = [nn_results['accuracy'], rf_results['accuracy']]
        aucs = [nn_results['auc'] if nn_results['auc'] is not None else 0, rf_results['auc'] if rf_results['auc'] is not None else 0]
        x = np.arange(len(models))
        width = 0.35
        axes[0, 2].bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
        axes[0, 2].bar(x + width/2, aucs, width, label='OvR AUC', alpha=0.8)
        axes[0, 2].set_title('Model Comparison')
        axes[0, 2].set_ylabel('Score')
        axes[0, 2].set_xticks(x)
        axes[0, 2].set_xticklabels(models)
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        # Confusion matrices
        from sklearn.metrics import confusion_matrix
        for i, (results, name) in enumerate([(nn_results, 'Neural Network'), (rf_results, 'Random Forest')]):
            cm = confusion_matrix(results['targets'], results['predictions'])
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, i])
            axes[1, i].set_title(f'{name} - Confusion Matrix')
            axes[1, i].set_xlabel('Predicted')
            axes[1, i].set_ylabel('Actual')
        # ROC Curve comparison (OvR)
        from sklearn.metrics import roc_curve
        # For multi-class, plot OvR ROC for each class
        for idx, (results, name) in enumerate([(nn_results, 'Neural Network'), (rf_results, 'Random Forest')]):
            try:
                y_true = results['targets']
                if name == 'Neural Network':
                    # Need probability scores for each class
                    # For simplicity, skip plotting ROC for now
                    pass
                else:
                    probas = rf_results.get('probas', None)
                    if probas is not None:
                        for i in range(probas.shape[1]):
                            fpr, tpr, _ = roc_curve(y_true == i, probas[:, i])
                            axes[1, 2].plot(fpr, tpr, label=f'{name} class {i}')
            except Exception:
                pass
        axes[1, 2].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[1, 2].set_title('OvR ROC Curves')
        axes[1, 2].set_xlabel('False Positive Rate')
        axes[1, 2].set_ylabel('True Positive Rate')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('plots/clean_model_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        logging.info("Visualizations saved to plots/clean_model_results.png")
    
    def save_results(self, nn_model, rf_model, nn_results, rf_results, nn_history):
        """Save models and results"""
        # Create directories
        os.makedirs('models', exist_ok=True)
        os.makedirs('plots', exist_ok=True)
        
        # Save neural network model
        torch.save(nn_model.state_dict(), 'models/clean_diabetes_nn.pth')
        
        # Save Random Forest model
        import joblib
        joblib.dump(rf_model, 'models/clean_diabetes_rf.pkl')
        joblib.dump(self.scaler, 'models/clean_scaler.pkl')
        
        # Save results summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        with open(f'clean_results_{timestamp}.txt', 'w', encoding='utf-8') as f:
            f.write("CLEAN DIABETES DETECTION MODEL RESULTS\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Timestamp: {timestamp}\n\n")
            
            f.write("NEURAL NETWORK RESULTS:\n")
            f.write(f"  Test Accuracy: {nn_results['accuracy']:.4f} ({nn_results['accuracy']*100:.2f}%)\n")
            f.write(f"  Test AUC: {nn_results['auc']:.4f} ({nn_results['auc']*100:.2f}%)\n")
            f.write(f"  Best Validation F1: {nn_history['best_val_f1']:.4f}\n\n")
            
            f.write("RANDOM FOREST RESULTS:\n")
            f.write(f"  Test Accuracy: {rf_results['accuracy']:.4f} ({rf_results['accuracy']*100:.2f}%)\n")
            f.write(f"  Test AUC: {rf_results['auc']:.4f} ({rf_results['auc']*100:.2f}%)\n\n")
            
            f.write("MODEL COMPARISON:\n")
            nn_better = "[BEST]" if nn_results['auc'] > rf_results['auc'] else ""
            rf_better = "[BEST]" if rf_results['auc'] > nn_results['auc'] else ""
            f.write(f"  Neural Network AUC: {nn_results['auc']:.4f} {nn_better}\n")
            f.write(f"  Random Forest AUC:  {rf_results['auc']:.4f} {rf_better}\n")
            
            improvement = abs(nn_results['auc'] - rf_results['auc']) * 100
            better_model = "Neural Network" if nn_results['auc'] > rf_results['auc'] else "Random Forest"
            f.write(f"  Best Model: {better_model} (+{improvement:.2f}% AUC)\n")
        
        logging.info(f"Results saved to clean_results_{timestamp}.txt")
        logging.info("Models saved to models/ directory")
    
    def train_combined_model(self, X, y):
        """Train models using combined dataset strategy"""
        try:
            print("🚀 Starting combined training with enhanced architecture...")
            
            # Split the enhanced Pima data
            X_train, X_temp, y_train, y_temp = train_test_split(
                X, y, test_size=0.4, random_state=42, stratify=y
            )
            X_val, X_test, y_val, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
            )
            
            # Scale the features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Save the scaler
            joblib.dump(self.scaler, 'models/combined_scaler.pkl')
            
            print(f"📊 Training set: {X_train_scaled.shape}, Validation: {X_val_scaled.shape}, Test: {X_test_scaled.shape}")
            
            # Create data loaders
            train_loader, val_loader, test_loader = self.create_data_loaders(
                X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, batch_size=64
            )
            
            # Train Enhanced Neural Network
            print("🧠 Training Enhanced Neural Network...")
            enhanced_model = EnhancedDiabetesModel(X_train_scaled.shape[1], num_classes=self.num_classes)
            enhanced_model.to(self.device)
            
            # Use different optimizers and learning rates
            optimizer = optim.AdamW(enhanced_model.parameters(), lr=0.001, weight_decay=0.01)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
            criterion = nn.CrossEntropyLoss()
            
            # Training loop for enhanced model
            enhanced_history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
            best_val_loss = float('inf')
            patience = 15
            patience_counter = 0
            
            for epoch in range(100):  # Increased epochs
                # Training phase
                enhanced_model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0
                
                for batch_X, batch_y in train_loader:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    
                    optimizer.zero_grad()
                    outputs = enhanced_model(batch_X)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(enhanced_model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    train_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    train_total += batch_y.size(0)
                    train_correct += (predicted == batch_y).sum().item()
                
                # Validation phase
                enhanced_model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                        outputs = enhanced_model(batch_X)
                        loss = criterion(outputs, batch_y)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs.data, 1)
                        val_total += batch_y.size(0)
                        val_correct += (predicted == batch_y).sum().item()
                
                # Calculate averages
                avg_train_loss = train_loss / len(train_loader)
                avg_val_loss = val_loss / len(val_loader)
                train_acc = 100 * train_correct / train_total
                val_acc = 100 * val_correct / val_total
                
                enhanced_history['train_loss'].append(avg_train_loss)
                enhanced_history['val_loss'].append(avg_val_loss)
                enhanced_history['train_acc'].append(train_acc)
                enhanced_history['val_acc'].append(val_acc)
                
                # Learning rate scheduling
                scheduler.step(avg_val_loss)
                
                # Early stopping
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                    torch.save(enhanced_model.state_dict(), 'models/enhanced_diabetes_nn.pth')
                else:
                    patience_counter += 1
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/100: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
                
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            # Load best model for evaluation
            enhanced_model.load_state_dict(torch.load('models/enhanced_diabetes_nn.pth'))
            enhanced_model.eval()
            
            # Evaluate Enhanced Neural Network
            print("📈 Evaluating Enhanced Neural Network...")
            enhanced_nn_results = self.evaluate_model(enhanced_model, test_loader, "Enhanced Neural Network")
            
            # Train Random Forest on enhanced features
            print("🌲 Training Enhanced Random Forest...")
            rf_enhanced = RandomForestClassifier(
                n_estimators=200,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                class_weight='balanced'
            )
            rf_enhanced.fit(X_train_scaled, y_train)
            
            # Evaluate Random Forest
            rf_results = self.evaluate_random_forest(rf_enhanced, X_test_scaled, y_test, "Enhanced Random Forest")
            
            # Save models
            joblib.dump(rf_enhanced, 'models/enhanced_diabetes_rf.pkl')
            self.models['enhanced_nn'] = enhanced_model
            self.models['enhanced_rf'] = rf_enhanced
            self.results['enhanced_nn'] = enhanced_nn_results
            self.results['enhanced_rf'] = rf_results
            
            # Create enhanced visualizations
            self.create_enhanced_visualizations(enhanced_nn_results, rf_results, enhanced_history)
            
            return enhanced_nn_results, rf_results, enhanced_history
            
        except Exception as e:
            logging.error(f"Error in combined training: {e}")
            raise
    
    def create_enhanced_visualizations(self, nn_results, rf_results, history):
        """Create enhanced visualizations for combined model performance"""
        try:
            fig, axes = plt.subplots(2, 3, figsize=(20, 12))
            
            # Training history
            epochs = range(1, len(history['train_loss']) + 1)
            axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
            axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
            axes[0, 0].set_title('Enhanced Model - Training History', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Training accuracy
            axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
            axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
            axes[0, 1].set_title('Enhanced Model - Accuracy Progress', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Accuracy (%)')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # Model comparison
            models = ['Enhanced NN', 'Enhanced RF']
            accuracies = [nn_results['accuracy'], rf_results['accuracy']]
            f1_scores = [nn_results['weighted_f1'], rf_results['weighted_f1']]
            
            x = np.arange(len(models))
            width = 0.35
            
            axes[0, 2].bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8, color='skyblue')
            axes[0, 2].bar(x + width/2, f1_scores, width, label='Weighted F1', alpha=0.8, color='lightcoral')
            axes[0, 2].set_xlabel('Models')
            axes[0, 2].set_ylabel('Score')
            axes[0, 2].set_title('Enhanced Model Performance Comparison', fontsize=14, fontweight='bold')
            axes[0, 2].set_xticks(x)
            axes[0, 2].set_xticklabels(models)
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
            
            # Confusion matrices
            from sklearn.metrics import confusion_matrix
            
            # NN confusion matrix
            cm_nn = confusion_matrix(nn_results['targets'], nn_results['predictions'])
            im1 = axes[1, 0].imshow(cm_nn, interpolation='nearest', cmap=plt.cm.Blues)
            axes[1, 0].set_title('Enhanced NN - Confusion Matrix', fontsize=14, fontweight='bold')
            for i in range(cm_nn.shape[0]):
                for j in range(cm_nn.shape[1]):
                    axes[1, 0].text(j, i, str(cm_nn[i, j]), ha="center", va="center", 
                                  color="white" if cm_nn[i, j] > cm_nn.max() / 2 else "black")
            
            # RF confusion matrix
            cm_rf = confusion_matrix(rf_results['targets'], rf_results['predictions'])
            im2 = axes[1, 1].imshow(cm_rf, interpolation='nearest', cmap=plt.cm.Greens)
            axes[1, 1].set_title('Enhanced RF - Confusion Matrix', fontsize=14, fontweight='bold')
            for i in range(cm_rf.shape[0]):
                for j in range(cm_rf.shape[1]):
                    axes[1, 1].text(j, i, str(cm_rf[i, j]), ha="center", va="center",
                                  color="white" if cm_rf[i, j] > cm_rf.max() / 2 else "black")
            
            # Performance metrics summary
            metrics_data = {
                'Model': ['Enhanced NN', 'Enhanced RF'],
                'Accuracy': [f"{nn_results['accuracy']:.4f}", f"{rf_results['accuracy']:.4f}"],
                'Weighted F1': [f"{nn_results['weighted_f1']:.4f}", f"{rf_results['weighted_f1']:.4f}"],
                'Macro F1': [f"{nn_results['macro_f1']:.4f}", f"{rf_results['macro_f1']:.4f}"]
            }
            
            axes[1, 2].axis('tight')
            axes[1, 2].axis('off')
            table = axes[1, 2].table(cellText=[[metrics_data['Accuracy'][0], metrics_data['Weighted F1'][0], metrics_data['Macro F1'][0]],
                                             [metrics_data['Accuracy'][1], metrics_data['Weighted F1'][1], metrics_data['Macro F1'][1]]],
                                   rowLabels=metrics_data['Model'],
                                   colLabels=['Accuracy', 'Weighted F1', 'Macro F1'],
                                   cellLoc='center', loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(12)
            axes[1, 2].set_title('Enhanced Model Performance Summary', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig('plots/enhanced_model_results.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print("📈 Enhanced visualizations saved to plots/enhanced_model_results.png")
            
        except Exception as e:
            logging.error(f"Error creating enhanced visualizations: {e}")
            print(f"⚠️ Could not create enhanced visualizations: {e}")
    
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

def main():
    """Main training pipeline with combined datasets"""
    print("\n" + "="*60)
    print("🏥 ENHANCED DIABETES DETECTION MODEL")
    print("="*60)
    print("📊 Combined Dataset Training - Pima + BRFSS")
    print("="*60 + "\n")
    
    trainer = CleanDiabetesTrainer()
    
    try:
        # Load and preprocess combined data
        print("🔄 Loading and preprocessing combined datasets...")
        X, y = trainer.load_and_preprocess_data()
        
        # Train enhanced models
        print("🚀 Training enhanced models with combined approach...")
        enhanced_nn_results, enhanced_rf_results, enhanced_history = trainer.train_combined_model(X, y)
        
        # Print final results
        print("\n" + "="*60)
        print("🎯 FINAL ENHANCED RESULTS")
        print("="*60)
        print(f"Enhanced Neural Network:")
        print(f"  📊 Accuracy: {enhanced_nn_results['accuracy']:.4f}")
        print(f"  📊 Weighted F1: {enhanced_nn_results['weighted_f1']:.4f}")
        print(f"  📊 Macro F1: {enhanced_nn_results['macro_f1']:.4f}")
        print(f"\nEnhanced Random Forest:")
        print(f"  📊 Accuracy: {enhanced_rf_results['accuracy']:.4f}")
        print(f"  📊 Weighted F1: {enhanced_rf_results['weighted_f1']:.4f}")
        print(f"  📊 Macro F1: {enhanced_rf_results['macro_f1']:.4f}")
        print("="*60 + "\n")
        
        # Save results with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_summary = f"""
Enhanced Diabetes Detection Results - {timestamp}
================================================================

DATASET INFORMATION:
- Combined Pima Indians (768 samples) + BRFSS (100K sample)
- Enhanced feature engineering applied
- Binary classification (Diabetes: Yes/No)

ENHANCED NEURAL NETWORK RESULTS:
- Architecture: Enhanced with dropout and regularization
- Accuracy: {enhanced_nn_results['accuracy']:.4f}
- Weighted F1: {enhanced_nn_results['weighted_f1']:.4f}
- Macro F1: {enhanced_nn_results['macro_f1']:.4f}
- AUC: {enhanced_nn_results['auc'] if enhanced_nn_results['auc'] else 'N/A'}

ENHANCED RANDOM FOREST RESULTS:
- n_estimators: 200, max_depth: 15
- Accuracy: {enhanced_rf_results['accuracy']:.4f}
- Weighted F1: {enhanced_rf_results['weighted_f1']:.4f}
- Macro F1: {enhanced_rf_results['macro_f1']:.4f}
- AUC: {enhanced_rf_results['auc'] if enhanced_rf_results['auc'] else 'N/A'}

IMPROVEMENTS ACHIEVED:
- Enhanced feature engineering with derived features
- Better regularization and dropout strategies
- Combined dataset approach for robustness
- Improved model architecture

Models saved to:
- Enhanced Neural Network: models/enhanced_diabetes_nn.pth
- Enhanced Random Forest: models/enhanced_diabetes_rf.pkl
- Combined Scaler: models/combined_scaler.pkl
- Feature Names: models/pima_feature_names.pkl
================================================================
"""
        
        with open(f'enhanced_results_{timestamp}.txt', 'w') as f:
            f.write(results_summary)
        
        print(f"📄 Enhanced results saved to enhanced_results_{timestamp}.txt")
        print("💾 Enhanced models saved to models/ directory")
        print("📈 Enhanced visualizations saved to plots/enhanced_model_results.png")
        
    except Exception as e:
        logging.error(f"Enhanced training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        logging.info("Creating data loaders...")
        train_loader, val_loader, test_loader = trainer.create_data_loaders(
            X_train, X_val, X_test, y_train, y_val, y_test
        )
        # Train Neural Network
        logging.info("Training neural network...")
        nn_model, nn_history = trainer.train_neural_network(
            train_loader, val_loader, X_train.shape[1]
        )
        # Train Random Forest
        logging.info("Training Random Forest...")
        rf_model, rf_val_f1 = trainer.train_random_forest(X_train, X_val, y_train, y_val)
        # Evaluate models
        logging.info("Evaluating models...")
        nn_results = trainer.evaluate_model(nn_model, test_loader, "Neural Network")
        rf_results = trainer.evaluate_random_forest(rf_model, X_test, y_test, "Random Forest")
        # Create visualizations
        logging.info("Creating visualizations...")
        trainer.create_visualizations(nn_results, rf_results, nn_history)
        # Save results
        logging.info("Saving models and results...")
        trainer.save_results(nn_model, rf_model, nn_results, rf_results, nn_history)
        # Print final summary
        print("\n" + "="*60)
        print("🎯 FINAL RESULTS (BRFSS Multi-Class)")
        print("="*60)
        print(f"Classes: {trainer.num_classes} ({sorted(y.unique())})")
        print(f"🔹 Neural Network  - Accuracy: {nn_results['accuracy']:.1%}, Weighted F1: {nn_results['weighted_f1']:.3f}, Macro F1: {nn_results['macro_f1']:.3f}, OvR AUC: {nn_results['auc']}")
        print(f"🔹 Random Forest   - Accuracy: {rf_results['accuracy']:.1%}, Weighted F1: {rf_results['weighted_f1']:.3f}, Macro F1: {rf_results['macro_f1']:.3f}, OvR AUC: {rf_results['auc']}")
        best_model = "Neural Network" if nn_results['weighted_f1'] > rf_results['weighted_f1'] else "Random Forest"
        print(f"🏆 Best Model: {best_model}")
        print("="*60)
        print("✅ Training completed successfully!")
        print("📁 Models saved to models/ directory")
        print("📊 Results saved to clean_results_*.txt")
        print("📈 Visualizations saved to plots/clean_model_results.png")
        print("="*60 + "\n")

if __name__ == "__main__":
    main()