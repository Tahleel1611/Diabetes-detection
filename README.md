# Diabetes Detection with PyTorch and Scikit-learn 🩺🤖

A machine learning project for diabetes detection that trains and compares a PyTorch neural network and a Random Forest model on a combined dataset.

## 🚀 Project Overview

This project provides a complete pipeline for training, evaluating, and saving diabetes detection models. It leverages a combination of the Pima Indians Diabetes Dataset and the BRFSS dataset to create more robust models. The main script trains both a custom-built neural network and a Random Forest classifier, evaluates their performance, and saves the results, models, and visualizations.

### Key Features

- **🧠 Dual-Model Training**: Trains both an `EnhancedDiabetesModel` (PyTorch) and a `RandomForestClassifier` (Scikit-learn) for comparison.
- **💾 Combined Datasets**: Uses both the Pima and a 100,000-sample subset of the BRFSS dataset for training.
- **🔧 Feature Engineering**: Automatically creates new features from the Pima dataset to improve model performance, including:
  - `BMI_Age`, `Glucose_BMI`, `Insulin_Glucose`
  - Squared features (`BMI_squared`, `Age_squared`)
  - Ratios (`BMI_Age_ratio`, `Glucose_Age_ratio`)
- **📊 Evaluation & Visualization**: Generates a comprehensive plot (`enhanced_model_results.png`) comparing model performance, training history, and confusion matrices.
- **💾 Model Persistence**: Saves the trained models (`.pth`, `.pkl`) and the data scaler (`.pkl`) to the `models/` directory for later use.
- **📜 Results Logging**: Creates a detailed text file with performance metrics after each run.

## 📊 Performance

The models currently achieve around **73-76% accuracy** on the test set. The performance can be seen in the output logs and the generated plots.

| Model Type              | Accuracy | Weighted F1 |
| ----------------------- | -------- | ----------- |
| Enhanced Neural Network | ~75%     | ~0.74       |
| Enhanced Random Forest  | ~73%     | ~0.73       |

## 🚀 Quick Start

### 1. Install Dependencies

Ensure you have all the required packages installed.

```bash
pip install -r requirements.txt
```

### 2. Run Training

Execute the main training script. This will preprocess the data, train both models, evaluate them, and save all artifacts (models, plots, results).

```bash
python train.py
```

### 3. Make a Simple Prediction

Use the `predict_simple.py` script to get a prediction for a single patient using the last trained models.

```bash
python predict_simple.py
```

### 4. Evaluate Models

To regenerate evaluation plots like the confusion matrix and ROC curve from the saved models, run the evaluation script.

```bash
python evaluate.py
```

### 5. Interpret Model Predictions

This project includes scripts to explain _why_ a model makes a certain prediction.

```bash
# Generate SHAP and LIME summary plots for the trained models
python interpretability.py

# Get a detailed explanation for a specific patient's risk profile
python explain_patient.py
```

## 🔬 Project Structure

- `train.py`: The main script for training and evaluating the models.
- `predict_simple.py`: A script for making predictions on a single data point.
- `evaluate.py`: Script to evaluate the trained models and generate plots.
- `interpretability.py`: Generates LIME and SHAP explanations for the models.
- `explain_patient.py`: Provides a detailed breakdown of a single patient's prediction.
- `data/`: Contains the datasets (`diabetes.csv`, `2023_BRFSS_CLEANED.csv`).
- `models/`: Stores the saved models and the data scaler.
- `plots/`: Saves the output visualizations.
- `explanations/`: Stores the output from the interpretability scripts.

## 🐛 Troubleshooting

### File Not Found Error

If you see a `FileNotFoundError` (e.g., for `models/pima_feature_names.pkl` or `plots/enhanced_model_results.png`), it's likely because the `models/` or `plots/` directories do not exist.

**Solution**: Manually create the directories in your project root.

```bash
mkdir models
mkdir plots
```

The training script should now be able to run successfully.
