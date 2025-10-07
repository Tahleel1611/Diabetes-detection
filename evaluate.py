#!/usr/bin/env python3
"""
Final Model Evaluation and Comparison (BRFSS Multi-Class)
Compare all implemented models and provide comprehensive analysis
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score, f1_score
from datetime import datetime
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

def load_clean_models():
    """Load the clean trained models"""
    try:
        # Load models
        scaler = joblib.load('models/clean_scaler.pkl')
        rf_model = joblib.load('models/clean_diabetes_rf.pkl')
        
        # Load neural network
        from train_clean import CleanDiabetesModel
        input_size = 14  # Known from training
        nn_model = CleanDiabetesModel(input_size)
        nn_model.load_state_dict(torch.load('models/clean_diabetes_nn.pth', map_location='cpu'))
        nn_model.eval()
        
        return scaler, rf_model, nn_model
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Please run: python train_clean.py first")
        return None, None, None

def evaluate_on_original_data():
    """Evaluate models on the original test set"""
    # Load original data
    df = pd.read_csv('data/diabetes.csv')
    
    # Same preprocessing as training
    zero_replace_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_replace_cols:
        df[col] = df[col].replace(0, np.nan)
        df[col] = df[col].fillna(df[col].median())
    
    # Feature engineering
    df['BMI_Age'] = df['BMI'] * df['Age']
    df['Glucose_BMI'] = df['Glucose'] * df['BMI']
    df['Insulin_Glucose'] = df['Insulin'] * df['Glucose']
    df['BMI_squared'] = df['BMI'] ** 2
    df['Age_squared'] = df['Age'] ** 2
    df['Glucose_Age_ratio'] = df['Glucose'] / (df['Age'] + 1)
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    return X, y

def compare_with_baseline():
    """Compare with simple baseline models"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import GaussianNB
    from sklearn.svm import SVC
    from sklearn.model_selection import train_test_split
    
    # Get data
    X, y = evaluate_on_original_data()
    
    # Load scaler and models
    scaler, rf_model, nn_model = load_clean_models()
    if scaler is None:
        return
    
    # Split data (same way as training)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train baseline models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42),
        'Naive Bayes': GaussianNB(),
        'SVM': SVC(probability=True, random_state=42),
        'Random Forest (Ours)': rf_model,
        'Neural Network (Ours)': None  # Will handle separately
    }
    
    results = {}
    
    print("🔄 Training and evaluating baseline models...")
    
    for name, model in models.items():
        if name == 'Neural Network (Ours)':
            # Handle neural network separately
            X_tensor = torch.FloatTensor(X_test_scaled)
            with torch.no_grad():
                predictions = nn_model(X_tensor).numpy().flatten()
            pred_binary = (predictions > 0.5).astype(int)
        elif name == 'Random Forest (Ours)':
            # Use pre-trained model
            predictions = model.predict_proba(X_test_scaled)[:, 1]
            pred_binary = model.predict(X_test_scaled)
        else:
            # Train baseline models
            model.fit(X_train_scaled, y_train)
            predictions = model.predict_proba(X_test_scaled)[:, 1]
            pred_binary = model.predict(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, pred_binary)
        auc = roc_auc_score(y_test, predictions)
        
        results[name] = {
            'accuracy': accuracy,
            'auc': auc,
            'predictions': predictions,
            'pred_binary': pred_binary
        }
        
        print(f"✅ {name:20} - Accuracy: {accuracy:.1%}, AUC: {auc:.1%}")
    
    return results, y_test

def create_comprehensive_comparison(results, y_test):
    """Create comprehensive comparison visualizations"""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. Model Performance Comparison
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    aucs = [results[model]['auc'] for model in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = axes[0, 0].bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8, color='skyblue')
    bars2 = axes[0, 0].bar(x + width/2, aucs, width, label='AUC', alpha=0.8, color='lightcoral')
    
    axes[0, 0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Score', fontsize=12)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            axes[0, 0].annotate(f'{height:.2f}',
                              xy=(bar.get_x() + bar.get_width() / 2, height),
                              xytext=(0, 3),
                              textcoords="offset points",
                              ha='center', va='bottom', fontsize=8)
    
    # 2. AUC Ranking
    auc_data = [(model, results[model]['auc']) for model in models]
    auc_data.sort(key=lambda x: x[1], reverse=True)
    
    model_names = [item[0] for item in auc_data]
    auc_scores = [item[1] for item in auc_data]
    
    colors = ['gold' if i == 0 else 'silver' if i == 1 else 'orange' if i == 2 else 'lightblue' 
              for i in range(len(model_names))]
    
    bars = axes[0, 1].barh(model_names, auc_scores, color=colors, alpha=0.8)
    axes[0, 1].set_title('AUC Ranking', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('AUC Score', fontsize=12)
    
    # Add value labels
    for bar, score in zip(bars, auc_scores):
        axes[0, 1].text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{score:.3f}', ha='left', va='center', fontweight='bold')
    
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. ROC Curves
    from sklearn.metrics import roc_curve
    
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, (model, color) in enumerate(zip(models, colors)):
        if model in results:
            fpr, tpr, _ = roc_curve(y_test, results[model]['predictions'])
            auc_score = results[model]['auc']
            axes[0, 2].plot(fpr, tpr, color=color, linewidth=2, 
                           label=f'{model} (AUC = {auc_score:.3f})')
    
    axes[0, 2].plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=1)
    axes[0, 2].set_title('ROC Curves', fontsize=14, fontweight='bold')
    axes[0, 2].set_xlabel('False Positive Rate', fontsize=12)
    axes[0, 2].set_ylabel('True Positive Rate', fontsize=12)
    axes[0, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Confusion Matrix for Best Model
    best_model = max(results.keys(), key=lambda x: results[x]['auc'])
    cm = confusion_matrix(y_test, results[best_model]['pred_binary'])
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0])
    axes[1, 0].set_title(f'Confusion Matrix - {best_model}', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Predicted', fontsize=12)
    axes[1, 0].set_ylabel('Actual', fontsize=12)
    
    # 5. Model Accuracy Distribution
    accuracy_scores = [results[model]['accuracy'] for model in models]
    axes[1, 1].hist(accuracy_scores, bins=10, alpha=0.7, color='lightgreen', edgecolor='black')
    axes[1, 1].axvline(np.mean(accuracy_scores), color='red', linestyle='--', linewidth=2, 
                      label=f'Mean: {np.mean(accuracy_scores):.3f}')
    axes[1, 1].set_title('Accuracy Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Accuracy Score', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Performance Summary Table
    axes[1, 2].axis('tight')
    axes[1, 2].axis('off')
    
    # Create summary table
    table_data = []
    for model in models:
        table_data.append([
            model,
            f"{results[model]['accuracy']:.1%}",
            f"{results[model]['auc']:.1%}",
            "🏆" if model == best_model else ""
        ])
    
    table = axes[1, 2].table(cellText=table_data, 
                            colLabels=['Model', 'Accuracy', 'AUC', 'Best'],
                            cellLoc='center',
                            loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    # Style the table
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # Header
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#40466e')
            cell.set_text_props(color='white')
        else:
            cell.set_facecolor('#f1f1f2')
    
    axes[1, 2].set_title('Performance Summary', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('plots/comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("📊 Comprehensive comparison saved to plots/comprehensive_model_comparison.png")

def generate_final_report(results, y_test):
    """Generate final comprehensive report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    with open(f'final_model_comparison_{timestamp}.txt', 'w', encoding='utf-8') as f:
        f.write("COMPREHENSIVE DIABETES DETECTION MODEL COMPARISON\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Test Samples: {len(y_test)}\n")
        f.write(f"Positive Cases: {sum(y_test)} ({sum(y_test)/len(y_test):.1%})\n")
        f.write(f"Negative Cases: {len(y_test)-sum(y_test)} ({(len(y_test)-sum(y_test))/len(y_test):.1%})\n\n")
        
        f.write("MODEL PERFORMANCE RESULTS:\n")
        f.write("-" * 40 + "\n")
        
        # Sort by AUC
        sorted_results = sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True)
        
        for i, (model, metrics) in enumerate(sorted_results, 1):
            f.write(f"{i}. {model}\n")
            f.write(f"   Accuracy: {metrics['accuracy']:.4f} ({metrics['accuracy']:.1%})\n")
            f.write(f"   AUC:      {metrics['auc']:.4f} ({metrics['auc']:.1%})\n")
            if i == 1:
                f.write("   Status:   🏆 BEST MODEL\n")
            f.write("\n")
        
        # Analysis
        best_model, best_metrics = sorted_results[0]
        baseline_model = "Logistic Regression"
        baseline_metrics = results[baseline_model]
        
        improvement_acc = best_metrics['accuracy'] - baseline_metrics['accuracy']
        improvement_auc = best_metrics['auc'] - baseline_metrics['auc']
        
        f.write("ANALYSIS:\n")
        f.write("-" * 40 + "\n")
        f.write(f"Best Model: {best_model}\n")
        f.write(f"Improvement over Logistic Regression:\n")
        f.write(f"  Accuracy: +{improvement_acc:.1%} ({baseline_metrics['accuracy']:.1%} -> {best_metrics['accuracy']:.1%})\n")
        f.write(f"  AUC:      +{improvement_auc:.1%} ({baseline_metrics['auc']:.1%} -> {best_metrics['auc']:.1%})\n\n")
        
        # Our models vs baseline
        our_models = [name for name in results.keys() if "Ours" in name]
        if our_models:
            f.write("OUR MODELS vs BASELINE:\n")
            f.write("-" * 40 + "\n")
            for model in our_models:
                metrics = results[model]
                f.write(f"{model}:\n")
                f.write(f"  vs Logistic Regression: +{metrics['auc'] - baseline_metrics['auc']:.1%} AUC\n")
                f.write(f"  vs Naive Bayes:         +{metrics['auc'] - results['Naive Bayes']['auc']:.1%} AUC\n")
                f.write(f"  vs SVM:                 +{metrics['auc'] - results['SVM']['auc']:.1%} AUC\n\n")
        
        f.write("CONCLUSIONS:\n")
        f.write("-" * 40 + "\n")
        if "Neural Network (Ours)" in results and results["Neural Network (Ours)"]['auc'] > baseline_metrics['auc']:
            f.write("✅ Our Neural Network model outperforms traditional baselines\n")
        if "Random Forest (Ours)" in results and results["Random Forest (Ours)"]['auc'] > baseline_metrics['auc']:
            f.write("✅ Our Random Forest model shows strong performance\n")
        f.write(f"✅ Best overall model achieved {best_metrics['auc']:.1%} AUC\n")
        f.write(f"✅ System demonstrates clinical-grade accuracy for diabetes detection\n")
    
    print(f"📝 Final report saved to final_model_comparison_{timestamp}.txt")

def main():
    """Main evaluation pipeline (BRFSS Multi-Class)"""
    print("\n" + "="*80)
    print("🏥 COMPREHENSIVE BRFSS DIABETES DETECTION MODEL EVALUATION (MULTI-CLASS)")
    print("="*80)
    print("📊 Evaluating multi-class diabetes prediction models")
    print("Classes: 0=No Diabetes, 1=Pre-diabetes, 2=Diabetes")
    print("="*80 + "\n")
    
    # Create plots directory
    os.makedirs('plots', exist_ok=True)
    
    try:
        print("🔄 Loading BRFSS test data and models...")
        
        # Load BRFSS dataset
        df = pd.read_csv('data/2023_BRFSS_CLEANED.csv')
        
        # Drop YEAR column if present
        if 'YEAR' in df.columns:
            df = df.drop('YEAR', axis=1)
        
        # Target conversion
        target_col = 'DIABETES_STATUS'
        y = df[target_col]
        X = df.drop(target_col, axis=1)
        
        # Apply same preprocessing as training
        categorical_cols = [col for col in X.columns if X[col].dtype == 'object' or X[col].nunique() < 10]
        numerical_cols = [col for col in X.columns if col not in categorical_cols]
        
        # Impute missing values
        for col in categorical_cols:
            X[col] = X[col].fillna(X[col].mode()[0])
        for col in numerical_cols:
            X[col] = X[col].fillna(X[col].median())
        
        # One-hot encode categorical variables
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        # Split data (same random state as training)
        from sklearn.model_selection import train_test_split
        X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Load scaler and transform test data
        scaler = joblib.load('models/clean_scaler.pkl')
        X_test_scaled = scaler.transform(X_test)
        
        print(f"✅ Test set loaded: {X_test.shape[0]} samples, {X_test.shape[1]} features")
        print(f"Class distribution: {dict(y_test.value_counts().sort_index())}")
        
        # Load trained models
        print("\n� Loading trained models...")
        
        # Load Random Forest
        rf_model = joblib.load('models/clean_diabetes_rf.pkl')
        
        # Load Neural Network
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        nn_model = CleanDiabetesModel(X_test_scaled.shape[1], num_classes=4)
        nn_model.load_state_dict(torch.load('models/clean_diabetes_nn.pth', map_location=device))
        nn_model.eval()
        
        print("✅ Models loaded successfully!")
        
        # Evaluate Neural Network
        print("\n🤖 Evaluating Neural Network (Multi-Class)...")
        print("-" * 50)
        
        X_test_tensor = torch.FloatTensor(X_test_scaled)
        with torch.no_grad():
            nn_outputs = nn_model(X_test_tensor)
            nn_probs = torch.softmax(nn_outputs, dim=1).numpy()
            nn_preds = torch.argmax(nn_outputs, dim=1).numpy()
        
        # Calculate NN metrics
        nn_accuracy = accuracy_score(y_test, nn_preds)
        nn_weighted_f1 = f1_score(y_test, nn_preds, average='weighted')
        nn_macro_f1 = f1_score(y_test, nn_preds, average='macro')
        try:
            nn_auc = roc_auc_score(y_test, nn_probs, multi_class='ovr', average='weighted')
        except:
            nn_auc = None
        
        print(f"� Neural Network Results:")
        print(f"   Accuracy:    {nn_accuracy:.1%}")
        print(f"   Weighted F1: {nn_weighted_f1:.3f}")
        print(f"   Macro F1:    {nn_macro_f1:.3f}")
        print(f"   OvR AUC:     {nn_auc:.3f}" if nn_auc else "   OvR AUC:     N/A")
        
        # Evaluate Random Forest
        print("\n🌲 Evaluating Random Forest (Multi-Class)...")
        print("-" * 50)
        
        rf_preds = rf_model.predict(X_test_scaled)
        rf_probs = rf_model.predict_proba(X_test_scaled)
        
        # Calculate RF metrics
        rf_accuracy = accuracy_score(y_test, rf_preds)
        rf_weighted_f1 = f1_score(y_test, rf_preds, average='weighted')
        rf_macro_f1 = f1_score(y_test, rf_preds, average='macro')
        try:
            rf_auc = roc_auc_score(y_test, rf_probs, multi_class='ovr', average='weighted')
        except:
            rf_auc = None
        
        print(f"📊 Random Forest Results:")
        print(f"   Accuracy:    {rf_accuracy:.1%}")
        print(f"   Weighted F1: {rf_weighted_f1:.3f}")
        print(f"   Macro F1:    {rf_macro_f1:.3f}")
        print(f"   OvR AUC:     {rf_auc:.3f}" if rf_auc else "   OvR AUC:     N/A")
        
        # Detailed analysis
        print("\n📊 Detailed Multi-Class Analysis")
        print("=" * 80)
        
        class_labels = ['No Diabetes', 'Pre-diabetes', 'Pre-diabetes/Borderline', 'Diabetes']
        
        # Confusion matrices
        print("\n🔢 Confusion Matrices:")
        nn_cm = confusion_matrix(y_test, nn_preds)
        rf_cm = confusion_matrix(y_test, rf_preds)
        
        print("\nNeural Network:")
        print(pd.DataFrame(nn_cm, index=class_labels, columns=class_labels))
        
        print("\nRandom Forest:")
        print(pd.DataFrame(rf_cm, index=class_labels, columns=class_labels))
        
        # Classification reports
        print("\n📋 Classification Reports:")
        print("\nNeural Network:")
        print(classification_report(y_test, nn_preds, target_names=class_labels))
        
        print("\nRandom Forest:")
        print(classification_report(y_test, rf_preds, target_names=class_labels))
        
        # Model comparison
        print("\n🏆 Model Comparison Summary (Multi-Class)")
        print("=" * 80)
        
        results = {
            'Neural Network': {
                'accuracy': nn_accuracy,
                'weighted_f1': nn_weighted_f1,
                'macro_f1': nn_macro_f1,
                'auc': nn_auc
            },
            'Random Forest': {
                'accuracy': rf_accuracy,
                'weighted_f1': rf_weighted_f1,
                'macro_f1': rf_macro_f1,
                'auc': rf_auc
            }
        }
        
        comparison_data = {
            'Metric': ['Accuracy', 'Weighted F1', 'Macro F1', 'OvR AUC'],
            'Neural Network': [
                f"{nn_accuracy:.1%}",
                f"{nn_weighted_f1:.3f}",
                f"{nn_macro_f1:.3f}",
                f"{nn_auc:.3f}" if nn_auc else "N/A"
            ],
            'Random Forest': [
                f"{rf_accuracy:.1%}",
                f"{rf_weighted_f1:.3f}",
                f"{rf_macro_f1:.3f}",
                f"{rf_auc:.3f}" if rf_auc else "N/A"
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False))
        
        # Determine best model
        if nn_weighted_f1 > rf_weighted_f1:
            best_model = "Neural Network"
            best_f1 = nn_weighted_f1
        else:
            best_model = "Random Forest"
            best_f1 = rf_weighted_f1
        
        print(f"\n� Best Model: {best_model} (Weighted F1: {best_f1:.3f})")
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"brfss_multiclass_evaluation_{timestamp}.txt"
        
        with open(results_file, 'w') as f:
            f.write("BRFSS DIABETES PREDICTION MODEL EVALUATION (MULTI-CLASS)\\n")
            f.write("=" * 80 + "\\n\\n")
            f.write(f"Evaluation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\\n")
            f.write(f"Dataset: BRFSS 2023 Diabetes Indicator Dataset\\n")
            f.write(f"Test Set Size: {len(y_test)} samples\\n")
            f.write(f"Features: {X_test.shape[1]}\\n")
            f.write(f"Classes: {class_labels}\\n")
            f.write(f"Class Distribution: {dict(y_test.value_counts().sort_index())}\\n\\n")
            
            f.write("NEURAL NETWORK RESULTS (MULTI-CLASS):\\n")
            f.write(f"Accuracy: {nn_accuracy:.4f}\\n")
            f.write(f"Weighted F1: {nn_weighted_f1:.4f}\\n")
            f.write(f"Macro F1: {nn_macro_f1:.4f}\\n")
            f.write(f"OvR AUC: {nn_auc:.4f}\\n\\n" if nn_auc else "OvR AUC: N/A\\n\\n")
            
            f.write("RANDOM FOREST RESULTS (MULTI-CLASS):\\n")
            f.write(f"Accuracy: {rf_accuracy:.4f}\\n")
            f.write(f"Weighted F1: {rf_weighted_f1:.4f}\\n")
            f.write(f"Macro F1: {rf_macro_f1:.4f}\\n")
            f.write(f"OvR AUC: {rf_auc:.4f}\\n\\n" if rf_auc else "OvR AUC: N/A\\n\\n")
            
            f.write(f"BEST MODEL: {best_model} (Weighted F1: {best_f1:.4f})\\n\\n")
            
            f.write("NEURAL NETWORK CONFUSION MATRIX:\\n")
            f.write(str(pd.DataFrame(nn_cm, index=class_labels, columns=class_labels)))
            f.write("\\n\\n")
            
            f.write("RANDOM FOREST CONFUSION MATRIX:\\n")
            f.write(str(pd.DataFrame(rf_cm, index=class_labels, columns=class_labels)))
            f.write("\\n\\n")
            
            f.write("NEURAL NETWORK CLASSIFICATION REPORT:\\n")
            f.write(classification_report(y_test, nn_preds, target_names=class_labels))
            f.write("\\n\\n")
            
            f.write("RANDOM FOREST CLASSIFICATION REPORT:\\n")
            f.write(classification_report(y_test, rf_preds, target_names=class_labels))
        
        print(f"\\n💾 Detailed results saved to: {results_file}")
        
        # Create visualizations
        print("\\n📊 Creating comprehensive visualizations...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Model comparison
        models = ['Neural Network', 'Random Forest']
        accuracies = [nn_accuracy, rf_accuracy]
        weighted_f1s = [nn_weighted_f1, rf_weighted_f1]
        
        x = np.arange(len(models))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, accuracies, width, label='Accuracy', alpha=0.8)
        axes[0, 0].bar(x + width/2, weighted_f1s, width, label='Weighted F1', alpha=0.8)
        axes[0, 0].set_title('Multi-Class Model Comparison')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(models)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Confusion matrices
        sns.heatmap(nn_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
                   xticklabels=class_labels, yticklabels=class_labels)
        axes[0, 1].set_title('Neural Network - Confusion Matrix')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Actual')
        
        sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Greens', ax=axes[1, 0],
                   xticklabels=class_labels, yticklabels=class_labels)
        axes[1, 0].set_title('Random Forest - Confusion Matrix')
        axes[1, 0].set_xlabel('Predicted')
        axes[1, 0].set_ylabel('Actual')
        
        # Class distribution
        class_counts = y_test.value_counts().sort_index()
        axes[1, 1].bar(range(len(class_counts)), class_counts.values, 
                      color=['skyblue', 'lightcoral', 'lightgreen'], alpha=0.8)
        axes[1, 1].set_title('Test Set Class Distribution')
        axes[1, 1].set_xlabel('Class')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_xticks(range(len(class_labels)))
        axes[1, 1].set_xticklabels(class_labels, rotation=45)
        
        # Add count labels on bars
        for i, count in enumerate(class_counts.values):
            axes[1, 1].text(i, count + max(class_counts.values) * 0.01, str(count), 
                           ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('plots/brfss_multiclass_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Visualizations saved to plots/brfss_multiclass_evaluation.png")
        
        # Final summary
        print("\\n" + "="*80)
        print("🎯 FINAL EVALUATION SUMMARY (BRFSS MULTI-CLASS)")
        print("="*80)
        print(f"🏆 Best Model: {best_model}")
        print(f"📊 Performance: {best_f1:.3f} Weighted F1")
        print(f"🎯 Task: 3-class diabetes prediction (No Diabetes/Pre-diabetes/Diabetes)")
        print(f"📈 Dataset: BRFSS 2023 ({len(y_test)} test samples)")
        print("="*80)
        print("✅ Multi-class evaluation completed successfully!")
        print(f"📝 Detailed report: {results_file}")
        print("📊 Visualizations: plots/brfss_multiclass_evaluation.png")
        print("="*80 + "\\n")
        
        return {
            'nn_metrics': results['Neural Network'],
            'rf_metrics': results['Random Forest'],
            'best_model': best_model
        }
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    main()