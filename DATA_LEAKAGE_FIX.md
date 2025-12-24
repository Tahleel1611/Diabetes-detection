# Data Leakage Fix Documentation

## Problem Identified

The original `train.py` code had a **critical data leakage issue** where feature engineering was performed on the entire dataset before splitting into train/test sets.

### Issue Location (Original Code)

```python
# ❌ WRONG: Feature engineering BEFORE train/test split
X_pima_enhanced = X_pima.copy()

if 'BMI' in X_pima.columns and 'Age' in X_pima.columns:
    X_pima_enhanced['BMI_Age'] = X_pima['BMI'] * X_pima['Age']
    X_pima_enhanced['BMI_squared'] = X_pima['BMI'] ** 2
    X_pima_enhanced['Age_squared'] = X_pima['Age'] ** 2
    X_pima_enhanced['BMI_Age_ratio'] = X_pima['BMI'] / (X_pima['Age'] + 1)

if 'Glucose' in X_pima.columns:
    X_pima_enhanced['Glucose_BMI'] = X_pima['Glucose'] * X_pima['BMI']
    X_pima_enhanced['Glucose_squared'] = X_pima['Glucose'] ** 2
    X_pima_enhanced['Glucose_Age_ratio'] = X_pima['Glucose'] / (X_pima['Age'] + 1)

# Then split happens AFTER feature creation
X_train, X_test, y_train, y_test = train_test_split(X_pima_enhanced, y_pima, ...)
```

## Why This Is a Problem

### 1. **Information Leakage**
When you create engineered features (like ratios, products, squares) using the entire dataset:
- The test set statistics influence the training set features
- The model indirectly "sees" test data during training
- This leads to **overly optimistic performance metrics**

### 2. **Invalid Model Evaluation**
- Test accuracy/F1 scores will be **artificially inflated**
- Model performance in production will be **worse than reported**
- Cannot trust cross-validation results

### 3. **Scaling Issues**
The `StandardScaler` was also affected:
```python
# ❌ WRONG: Scaler fitted on combined data after feature engineering
X_train_scaled = self.scaler.fit_transform(X_train)  
```
The scaler learns from test set statistics through the engineered features.

## The Correct Approach

### Step 1: Split Data FIRST
```python
# ✅ CORRECT: Split raw data before any transformation
X_train, X_test, y_train, y_test = train_test_split(
    X_pima,  # Original features only
    y_pima, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_pima
)
```

### Step 2: Create Features on Training Set Only
```python
# ✅ CORRECT: Feature engineering on training set
X_train_enhanced = X_train.copy()

if 'BMI' in X_train.columns and 'Age' in X_train.columns:
    X_train_enhanced['BMI_Age'] = X_train['BMI'] * X_train['Age']
    X_train_enhanced['BMI_squared'] = X_train['BMI'] ** 2
    # ... etc
```

### Step 3: Apply Same Transformations to Test Set
```python
# ✅ CORRECT: Apply identical transformations to test set
X_test_enhanced = X_test.copy()

if 'BMI' in X_test.columns and 'Age' in X_test.columns:
    X_test_enhanced['BMI_Age'] = X_test['BMI'] * X_test['Age']
    X_test_enhanced['BMI_squared'] = X_test['BMI'] ** 2
    # ... etc
```

### Step 4: Fit Scaler on Training Set Only
```python
# ✅ CORRECT: Scaler learns from training set only
X_train_scaled = self.scaler.fit_transform(X_train_enhanced)
X_test_scaled = self.scaler.transform(X_test_enhanced)  # transform only!
```

## Implementation Checklist

To fix data leakage in `train.py`, ensure:

- [ ] Train/test split happens BEFORE any feature engineering
- [ ] Feature engineering code is applied separately to train and test sets
- [ ] Scaler is `fit` on training data only
- [ ] Scaler only `transforms` test data (no fitting)
- [ ] No statistics from test set influence training
- [ ] Same applies to K-Fold cross-validation (each fold treated independently)

## Code Structure for Fix

Create a helper function to avoid code duplication:

```python
def engineer_features(X):
    """Apply feature engineering transformations to dataset."""
    X_enhanced = X.copy()
    
    if 'BMI' in X.columns and 'Age' in X.columns:
        X_enhanced['BMI_Age'] = X['BMI'] * X['Age']
        X_enhanced['Glucose_BMI'] = X['Glucose'] * X['BMI']
        X_enhanced['BMI_squared'] = X['BMI'] ** 2
        X_enhanced['Age_squared'] = X['Age'] ** 2
        X_enhanced['BMI_Age_ratio'] = X['BMI'] / (X['Age'] + 1)
        
    if 'Glucose' in X.columns:
        X_enhanced['Glucose_squared'] = X['Glucose'] ** 2
        X_enhanced['Glucose_Age_ratio'] = X['Glucose'] / (X['Age'] + 1)
    
    return X_enhanced

# Usage:
X_train_enhanced = engineer_features(X_train)
X_test_enhanced = engineer_features(X_test)
```

## Expected Impact After Fix

### Performance Changes
- **Test accuracy may decrease by 2-5%** (this is expected and correct)
- **Cross-validation scores will be more realistic**
- **Model generalization will actually improve** in production

### Trust in Results
- ✅ Performance metrics are now trustworthy
- ✅ Model evaluation reflects real-world performance
- ✅ Can confidently deploy the model

## Verification Steps

After implementing the fix:

1. **Check feature names**: Ensure train and test sets have identical feature names
2. **Verify scaler**: Confirm scaler is only fitted once on training data
3. **Compare results**: Document before/after performance metrics
4. **Review code flow**: Ensure no test data touches training pipeline

## Related Files

Files that need to be checked for similar issues:
- `train.py` - Main training script (PRIMARY FIX NEEDED)
- `predict_simple.py` - Ensure same feature engineering is applied
- `interpretability.py` - Feature engineering for SHAP analysis
- `evaluate.py` - If it exists, check data loading

## Best Practices Going Forward

1. **Always split first, transform second**
2. **Use sklearn Pipeline** to encapsulate transformations
3. **Create reusable feature engineering functions**
4. **Document transformation order clearly**
5. **Add unit tests to catch data leakage**

## References

- [Scikit-learn: Data Leakage](https://scikit-learn.org/stable/common_pitfalls.html#data-leakage)
- [Kaggle: Data Leakage in Machine Learning](https://www.kaggle.com/code/alexisbcook/data-leakage)
- [Machine Learning Mastery: Data Leakage](https://machinelearningmastery.com/data-leakage-machine-learning/)

---

**Note**: This fix is a **critical priority**. Data leakage makes all reported metrics unreliable and can lead to poor model performance in production.
