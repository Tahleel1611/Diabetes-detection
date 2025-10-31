import os
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import optuna
from sklearn.metrics import confusion_matrix, roc_curve

def create_final_visualizations(nn_results, rf_results, rf_model, feature_names, study, history):
    """Create a dashboard of final visualizations."""
    print(" Generating final visualization dashboard...")
    os.makedirs('output/plots', exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Training History (Loss and Accuracy)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('Neural Network Training History', fontsize=16, fontweight='bold')

    # Loss plot
    ax1.plot(history['train_loss'], label='Training Loss', color='blue')
    ax1.plot(history['val_loss'], label='Validation Loss', color='orange')
    ax1.set_title('Loss Over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    # Accuracy plot
    ax2.plot(history['train_acc'], label='Training Accuracy', color='blue')
    ax2.plot(history['val_acc'], label='Validation Accuracy', color='orange')
    ax2.set_title('Accuracy Over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('output/plots/nn_training_history.png', dpi=300)
    plt.close()
    print(" Saved training history plot.")

    # 2. Confusion Matrices
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

    # 3. ROC Curves
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

    # 4. RF Feature Importance
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

    # 5. Optuna Visualizations
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

    print(" All visualizations saved to output/plots/")
