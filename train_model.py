"""
Quick training script - trains model directly on existing data
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
# import shap  # Skipping due to dependency issue
import joblib
import json
from pathlib import Path

print("=" * 60)
print("WORKFORCEIQ - QUICK MODEL TRAINING")
print("=" * 60)

# Load the synthetic training data
print("\n1. Loading training data...")
df = pd.read_csv('archive/layoffs/train.csv')
print(f"✅ Loaded {len(df)} records")
print(f"Columns: {df.columns.tolist()}")

# Prepare features
print("\n2. Preparing features...")
feature_columns = ['funding_amount', 'employee_count', 'growth_rate', 'valuation']
X = df[feature_columns].copy()
y = df['layoff_happened'].copy()

# Handle missing values
X = X.fillna(X.median())

print(f"Features shape: {X.shape}")
print(f"Target distribution:\n{y.value_counts()}")

# Split data
print("\n3. Splitting data (80-20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Train XGBoost
print("\n4. Training XGBoost model...")
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'
)

model.fit(X_train, y_train)
print("✅ Model trained!")

# Evaluate
print("\n5. Evaluating model...")
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No Layoff', 'Layoff']))

roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"\nROC-AUC Score: {roc_auc:.4f}")

# Create simple explainer (feature importance based)
print("\n6. Creating explainer...")
# We'll use XGBoost's built-in feature importance instead of SHAP
explainer = None  # Will use feature_importances_ from model
print("✅ Explainer ready (using built-in feature importance)!")

# Save models
print("\n7. Saving models...")
Path('backend/models').mkdir(parents=True, exist_ok=True)

joblib.dump(model, 'backend/models/layoff_predictor.joblib')
# Save feature importances instead of SHAP
feature_importance = dict(zip(X.columns, model.feature_importances_))
joblib.dump(feature_importance, 'backend/models/shap_explainer.joblib')  # Keep filename for compatibility
joblib.dump(list(X.columns), 'backend/models/feature_list.joblib')

# Save model info
model_info = {
    'features': list(X.columns),
    'n_features': len(X.columns),
    'roc_auc': float(roc_auc),
    'training_samples': len(X_train),
    'test_samples': len(X_test)
}

with open('backend/models/model_info.json', 'w') as f:
    json.dump(model_info, f, indent=2)

print("✅ Models saved to backend/models/")
print("\nFiles created:")
print("  - layoff_predictor.joblib")
print("  - shap_explainer.joblib")
print("  - feature_list.joblib")
print("  - model_info.json")

print("\n" + "=" * 60)
print("✅ TRAINING COMPLETE!")
print("=" * 60)
print("\nNext steps:")
print("1. Start API: cd backend ; uvicorn api.main:app --reload")
print("2. Start Dashboard: streamlit run frontend/app.py")
