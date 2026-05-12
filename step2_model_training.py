"""
================================================================================
SMART WAREHOUSE DEMAND PREDICTION OF LPG CYLINDERS
Step 2: Model Training
================================================================================
WHAT THIS FILE DOES:
- Loads the preprocessed data from Step 1
- Trains MULTIPLE ML models for both tasks:
    TASK A — Regression  : Predict exact demand (cylinders)
    TASK B — Classification: Predict stockout risk (0 or 1)
- Compares all models and picks the BEST one for each task
- Saves the best models to disk
================================================================================
MODELS WE TRY:
  Regression:
    1. Linear Regression     — simple baseline, draws a straight line
    2. Decision Tree         — learns if/else rules from data
    3. Random Forest         — 100 decision trees working together
    4. Gradient Boosting     — builds trees sequentially, fixing errors

  Classification:
    1. Logistic Regression   — predicts probability of stockout
    2. Decision Tree         — same idea but for yes/no output
    3. Random Forest         — ensemble of trees for classification
    4. Gradient Boosting     — sequential boosting for classification
================================================================================
"""

import pandas as pd
import numpy as np
import joblib, os, warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model    import LinearRegression, LogisticRegression
from sklearn.tree            import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble        import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble        import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.metrics         import (mean_absolute_error, mean_squared_error, r2_score,
                                     accuracy_score, precision_score, recall_score,
                                     f1_score, classification_report)

OUTPUT_PATH = "outputs/"
MODEL_PATH  = "models/"

print("=" * 65)
print("  STEP 2: MODEL TRAINING")
print("=" * 65)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD PREPROCESSED DATA (from Step 1)
# ─────────────────────────────────────────────────────────────────────────────
print("\n[1/5] Loading preprocessed data...")

X_train = pd.read_csv(OUTPUT_PATH + 'X_train.csv')
X_test  = pd.read_csv(OUTPUT_PATH + 'X_test.csv')
y_train_demand = pd.read_csv(OUTPUT_PATH + 'y_train_demand.csv').squeeze()
y_test_demand  = pd.read_csv(OUTPUT_PATH + 'y_test_demand.csv').squeeze()
y_train_stock  = pd.read_csv(OUTPUT_PATH + 'y_train_stock.csv').squeeze().astype(int)
y_test_stock   = pd.read_csv(OUTPUT_PATH + 'y_test_stock.csv').squeeze().astype(int)

print(f"   X_train: {X_train.shape} | X_test: {X_test.shape}")

# ─────────────────────────────────────────────────────────────────────────────
# TASK A: REGRESSION — Predict Actual Demand (cylinders)
# Metric used: R² (closer to 1.0 = better), MAE (lower = better)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  TASK A: REGRESSION — Predict Demand (cylinders)")
print("─" * 65)

regression_models = {
    "Linear Regression"    : LinearRegression(),
    "Decision Tree"        : DecisionTreeRegressor(max_depth=8, random_state=42),
    "Random Forest"        : RandomForestRegressor(n_estimators=100, max_depth=12,
                                                   random_state=42, n_jobs=-1),
    "Gradient Boosting"    : GradientBoostingRegressor(n_estimators=150, max_depth=5,
                                                       learning_rate=0.1, random_state=42),
}

reg_results = {}
print(f"\n   {'Model':<25} {'R² Score':>10} {'MAE':>10} {'RMSE':>10}")
print(f"   {'─'*25} {'─'*10} {'─'*10} {'─'*10}")

for name, model in regression_models.items():
    model.fit(X_train, y_train_demand)
    preds = model.predict(X_test)
    r2   = r2_score(y_test_demand, preds)
    mae  = mean_absolute_error(y_test_demand, preds)
    rmse = np.sqrt(mean_squared_error(y_test_demand, preds))
    reg_results[name] = {'model': model, 'r2': r2, 'mae': mae, 'rmse': rmse, 'preds': preds}
    marker = " ◄ BEST" if name == max(reg_results, key=lambda k: reg_results[k]['r2']) else ""
    print(f"   {name:<25} {r2:>10.4f} {mae:>10.4f} {rmse:>10.4f}{marker}")

# Pick best regression model (highest R²)
best_reg_name  = max(reg_results, key=lambda k: reg_results[k]['r2'])
best_reg_model = reg_results[best_reg_name]['model']
best_reg_preds = reg_results[best_reg_name]['preds']

print(f"\n   ✅ Best Regression Model: {best_reg_name}")
print(f"      R² = {reg_results[best_reg_name]['r2']:.4f}  "
      f"MAE = {reg_results[best_reg_name]['mae']:.4f}  "
      f"RMSE = {reg_results[best_reg_name]['rmse']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# TASK B: CLASSIFICATION — Predict Stockout Risk (0 or 1)
# Metric used: F1-Score (balances precision and recall)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 65)
print("  TASK B: CLASSIFICATION — Predict Stockout Risk (0 or 1)")
print("─" * 65)

classification_models = {
    "Logistic Regression"  : LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree"        : DecisionTreeClassifier(max_depth=8, random_state=42),
    "Random Forest"        : RandomForestClassifier(n_estimators=100, max_depth=12,
                                                    random_state=42, n_jobs=-1),
    "Gradient Boosting"    : GradientBoostingClassifier(n_estimators=150, max_depth=5,
                                                        learning_rate=0.1, random_state=42),
}

clf_results = {}
print(f"\n   {'Model':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
print(f"   {'─'*25} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

for name, model in classification_models.items():
    model.fit(X_train, y_train_stock)
    preds = model.predict(X_test)
    acc  = accuracy_score(y_test_stock, preds)
    prec = precision_score(y_test_stock, preds, zero_division=0)
    rec  = recall_score(y_test_stock, preds, zero_division=0)
    f1   = f1_score(y_test_stock, preds, zero_division=0)
    clf_results[name] = {'model': model, 'acc': acc, 'prec': prec,
                         'rec': rec, 'f1': f1, 'preds': preds}
    print(f"   {name:<25} {acc:>10.4f} {prec:>10.4f} {rec:>10.4f} {f1:>10.4f}")

best_clf_name  = max(clf_results, key=lambda k: clf_results[k]['f1'])
best_clf_model = clf_results[best_clf_name]['model']
best_clf_preds = clf_results[best_clf_name]['preds']

print(f"\n   ✅ Best Classification Model: {best_clf_name}")
print(f"      Accuracy={clf_results[best_clf_name]['acc']:.4f}  "
      f"F1={clf_results[best_clf_name]['f1']:.4f}")

# Detailed classification report
print("\n   Detailed Classification Report (Best Model):")
report = classification_report(y_test_stock, best_clf_preds,
                               target_names=['No Stockout','Stockout'])
for line in report.split('\n'):
    print('   ' + line)

# ─────────────────────────────────────────────────────────────────────────────
# SAVE MODELS & RESULTS
# ─────────────────────────────────────────────────────────────────────────────
print("\n[5/5] Saving models and results...")

joblib.dump(best_reg_model, MODEL_PATH + 'best_demand_model.pkl')
joblib.dump(best_clf_model, MODEL_PATH + 'best_stockout_model.pkl')

# Save all model comparison results
reg_df = pd.DataFrame([
    {'Model': k, 'R2_Score': v['r2'], 'MAE': v['mae'], 'RMSE': v['rmse']}
    for k, v in reg_results.items()
]).sort_values('R2_Score', ascending=False)

clf_df = pd.DataFrame([
    {'Model': k, 'Accuracy': v['acc'], 'Precision': v['prec'],
     'Recall': v['rec'], 'F1_Score': v['f1']}
    for k, v in clf_results.items()
]).sort_values('F1_Score', ascending=False)

reg_df.to_csv(OUTPUT_PATH + 'regression_model_comparison.csv', index=False)
clf_df.to_csv(OUTPUT_PATH + 'classification_model_comparison.csv', index=False)

# Save predictions
pred_df = pd.DataFrame({
    'Actual_Demand'       : y_test_demand.values,
    'Predicted_Demand'    : best_reg_preds,
    'Actual_Stockout'     : y_test_stock.values,
    'Predicted_Stockout'  : best_clf_preds,
})
pred_df.to_csv(OUTPUT_PATH + 'test_predictions.csv', index=False)

# Save model info
model_info = {
    'best_regression_model'    : best_reg_name,
    'regression_r2'            : reg_results[best_reg_name]['r2'],
    'regression_mae'           : reg_results[best_reg_name]['mae'],
    'regression_rmse'          : reg_results[best_reg_name]['rmse'],
    'best_classification_model': best_clf_name,
    'classification_accuracy'  : clf_results[best_clf_name]['acc'],
    'classification_f1'        : clf_results[best_clf_name]['f1'],
}
joblib.dump(model_info, MODEL_PATH + 'model_info.pkl')

print(f"   Models saved to: {MODEL_PATH}")

print("\n" + "=" * 65)
print("  TRAINING COMPLETE!")
print(f"  Best Demand Model   : {best_reg_name}")
print(f"  Best Stockout Model : {best_clf_name}")
print("=" * 65)
print("  Next: Run step3_visualization.py")
print("=" * 65)
