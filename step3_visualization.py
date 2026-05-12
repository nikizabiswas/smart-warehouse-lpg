"""
================================================================================
SMART WAREHOUSE DEMAND PREDICTION OF LPG CYLINDERS
Step 3: Visualization & Charts
================================================================================
WHAT THIS FILE DOES:
- Creates 8 professional charts to visualize model performance and data insights:
  1. Model Comparison (Regression R2 scores)
  2. Model Comparison (Classification F1 scores)
  3. Actual vs Predicted Demand (scatter plot)
  4. Prediction Error Distribution
  5. Monthly Demand Trend by Season
  6. Feature Importance (which inputs matter most)
  7. Stockout Risk by Warehouse Zone
  8. Confusion Matrix (Classification accuracy breakdown)

UPDATED FOR v2 DATASET:
- Reads from Warehouse_Demand_Realistic_v2.xlsx
- Sheet name: 'Train_80pct' with header=1
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
import joblib, os, warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import confusion_matrix

OUTPUT_PATH = "outputs/"
MODEL_PATH  = "models/"
CHART_PATH  = "charts/"
os.makedirs(CHART_PATH, exist_ok=True)

plt.rcParams.update({
    'font.family'      : 'DejaVu Sans',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
    'axes.grid'        : True,
    'grid.alpha'       : 0.3,
    'figure.dpi'       : 150,
})
COLORS = ['#1A5276','#1E8449','#B7950B','#C0392B','#7D3C98','#117A65']

print("=" * 65)
print("  STEP 3: VISUALIZATION & CHARTS  (v2 Dataset)")
print("=" * 65)

# -- Load Data -----------------------------------------------------------------
print("\n[1/9] Loading data and models...")
X_train       = pd.read_csv(OUTPUT_PATH + 'X_train.csv')
X_test        = pd.read_csv(OUTPUT_PATH + 'X_test.csv')
y_test_demand = pd.read_csv(OUTPUT_PATH + 'y_test_demand.csv').squeeze()
y_test_stock  = pd.read_csv(OUTPUT_PATH + 'y_test_stock.csv').squeeze().astype(int)
pred_df       = pd.read_csv(OUTPUT_PATH + 'test_predictions.csv')
reg_df        = pd.read_csv(OUTPUT_PATH + 'regression_model_comparison.csv')
clf_df        = pd.read_csv(OUTPUT_PATH + 'classification_model_comparison.csv')
model_info    = joblib.load(MODEL_PATH  + 'model_info.pkl')
best_reg      = joblib.load(MODEL_PATH  + 'best_demand_model.pkl')
best_clf      = joblib.load(MODEL_PATH  + 'best_stockout_model.pkl')
feature_cols  = joblib.load(MODEL_PATH  + 'feature_cols.pkl')

# Load raw training data for trend charts (v2 sheet/header format)
raw_train = pd.read_excel(
    "data/Warehouse_Demand_Realistic_v2.xlsx",
    sheet_name="Train_80pct",
    header=1
)
for c in ['Month_Number','Year','Actual_Demand (cylinders)','Stockout_Occurred','Zone_Opening_Stock']:
    raw_train[c] = pd.to_numeric(raw_train[c], errors='coerce')

print("   Data loaded successfully.\n")

# =============================================================================
# CHART 1: Regression Model Comparison (R2 scores)
# =============================================================================
print("[2/9] Chart 1 - Regression model comparison...")
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(reg_df['Model'], reg_df['R2_Score'], color=COLORS[:len(reg_df)], height=0.5)
ax.set_xlim(0, 1.05)
ax.set_xlabel('R2 Score (higher = better)', fontsize=11)
ax.set_title('Regression Model Comparison\nR2 Score on Test Set', fontsize=13, fontweight='bold', pad=12)
for bar, val in zip(bars, reg_df['R2_Score']):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=10, fontweight='bold')
ax.axvline(0.9, color='red', linestyle='--', alpha=0.5, label='Target: 0.90')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(CHART_PATH + 'chart1_regression_comparison.png', bbox_inches='tight')
plt.close()
print("   Saved: chart1_regression_comparison.png")

# =============================================================================
# CHART 2: Classification Model Comparison (F1 scores)
# =============================================================================
print("[3/9] Chart 2 - Classification model comparison...")
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.barh(clf_df['Model'], clf_df['F1_Score'], color=COLORS[:len(clf_df)], height=0.5)
ax.set_xlim(0, 1.05)
ax.set_xlabel('F1 Score (higher = better)', fontsize=11)
ax.set_title('Classification Model Comparison\nF1 Score on Test Set', fontsize=13, fontweight='bold', pad=12)
for bar, val in zip(bars, clf_df['F1_Score']):
    ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(CHART_PATH + 'chart2_classification_comparison.png', bbox_inches='tight')
plt.close()
print("   Saved: chart2_classification_comparison.png")

# =============================================================================
# CHART 3: Actual vs Predicted Demand
# =============================================================================
print("[4/9] Chart 3 - Actual vs predicted demand...")
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(pred_df['Actual_Demand'], pred_df['Predicted_Demand'],
           alpha=0.4, color='#1A5276', s=20, label='Predictions')
lims = [pred_df['Actual_Demand'].min()-0.5, pred_df['Actual_Demand'].max()+0.5]
ax.plot(lims, lims, 'r--', linewidth=1.5, label='Perfect Prediction')
ax.set_xlabel('Actual Demand (cylinders)', fontsize=11)
ax.set_ylabel('Predicted Demand (cylinders)', fontsize=11)
ax.set_title(f'Actual vs Predicted Demand\nBest Model: {model_info["best_regression_model"]} '
             f'| R2={model_info["regression_r2"]:.4f}', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(CHART_PATH + 'chart3_actual_vs_predicted.png', bbox_inches='tight')
plt.close()
print("   Saved: chart3_actual_vs_predicted.png")

# =============================================================================
# CHART 4: Prediction Error Distribution
# =============================================================================
print("[5/9] Chart 4 - Prediction error distribution...")
errors = pred_df['Actual_Demand'] - pred_df['Predicted_Demand']
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(errors, bins=40, color='#1A5276', alpha=0.75, edgecolor='white')
ax.axvline(0,             color='red',    linestyle='--', linewidth=1.5, label='Zero Error')
ax.axvline(errors.mean(), color='orange', linestyle='--', linewidth=1.5,
           label=f'Mean Error: {errors.mean():.3f}')
ax.set_xlabel('Prediction Error (Actual - Predicted)', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Distribution of Prediction Errors\n(Centered around 0 = good model)', fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(CHART_PATH + 'chart4_error_distribution.png', bbox_inches='tight')
plt.close()
print("   Saved: chart4_error_distribution.png")

# =============================================================================
# CHART 5: Monthly Demand Trend
# =============================================================================
print("[6/9] Chart 5 - Monthly demand trend...")
monthly = raw_train.groupby('Month_Number')['Actual_Demand (cylinders)'].mean().reset_index()
month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
season_colors = {1:'#5DADE2',2:'#5DADE2',3:'#F39C12',4:'#F39C12',5:'#F39C12',
                 6:'#27AE60',7:'#27AE60',8:'#27AE60',9:'#27AE60',
                 10:'#E67E22',11:'#E67E22',12:'#5DADE2'}
fig, ax = plt.subplots(figsize=(11, 5))
bar_colors = [season_colors[m] for m in monthly['Month_Number']]
bars = ax.bar(month_names, monthly['Actual_Demand (cylinders)'], color=bar_colors, edgecolor='white', width=0.6)
ax.set_xlabel('Month', fontsize=11)
ax.set_ylabel('Avg Demand (cylinders)', fontsize=11)
ax.set_title('Average Monthly LPG Demand by Month\n(Training Data — v2 Dataset)', fontsize=13, fontweight='bold')
legend_els = [mpatches.Patch(color='#5DADE2', label='Winter (Dec-Feb)'),
              mpatches.Patch(color='#F39C12', label='Summer (Mar-May)'),
              mpatches.Patch(color='#27AE60', label='Monsoon (Jun-Sep)'),
              mpatches.Patch(color='#E67E22', label='Autumn (Oct-Nov)')]
ax.legend(handles=legend_els, fontsize=9)
for bar in bars:
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
            f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig(CHART_PATH + 'chart5_monthly_demand_trend.png', bbox_inches='tight')
plt.close()
print("   Saved: chart5_monthly_demand_trend.png")

# =============================================================================
# CHART 6: Feature Importance
# =============================================================================
print("[7/9] Chart 6 - Feature importance...")
if hasattr(best_reg, 'feature_importances_'):
    importances = best_reg.feature_importances_
    feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=True).tail(12)
    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(feat_imp.index, feat_imp.values, color='#1A5276', height=0.6)
    ax.set_xlabel('Feature Importance Score', fontsize=11)
    ax.set_title(f'Top Feature Importances\n{model_info["best_regression_model"]} - Demand Prediction',
                 fontsize=12, fontweight='bold')
    for bar, val in zip(bars, feat_imp.values):
        ax.text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2,
                f'{val:.4f}', va='center', fontsize=9)
    plt.tight_layout()
    plt.savefig(CHART_PATH + 'chart6_feature_importance.png', bbox_inches='tight')
    plt.close()
    print("   Saved: chart6_feature_importance.png")

# =============================================================================
# CHART 7: Stockout Rate by Zone (using v2 zone data from Zone_Summary sheet)
# =============================================================================
print("[8/9] Chart 7 - Stockout rate by zone...")
raw_train['Stockout_Occurred'] = pd.to_numeric(raw_train['Stockout_Occurred'], errors='coerce')
zone_stock = raw_train.groupby('Warehouse_Zone')['Stockout_Occurred'].mean().sort_values(ascending=False) * 100
fig, ax = plt.subplots(figsize=(11, 5))
bar_colors2 = ['#C0392B' if v > zone_stock.mean() else '#1E8449' for v in zone_stock.values]
bars = ax.bar(zone_stock.index, zone_stock.values, color=bar_colors2, edgecolor='white', width=0.6)
ax.axhline(zone_stock.mean(), color='orange', linestyle='--', linewidth=1.5,
           label=f'Avg: {zone_stock.mean():.1f}%')
ax.set_ylabel('Stockout Rate (%)', fontsize=11)
ax.set_title('Stockout Rate by Warehouse Zone\n(Red = above average risk)', fontsize=12, fontweight='bold')
ax.set_xticklabels(zone_stock.index, rotation=30, ha='right')
ax.legend(fontsize=10)
for bar, val in zip(bars, zone_stock.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
            f'{val:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.savefig(CHART_PATH + 'chart7_stockout_by_zone.png', bbox_inches='tight')
plt.close()
print("   Saved: chart7_stockout_by_zone.png")

# =============================================================================
# CHART 8: Confusion Matrix
# =============================================================================
print("[9/9] Chart 8 - Confusion matrix...")
cm = confusion_matrix(y_test_stock, pred_df['Predicted_Stockout'])
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['No Stockout','Stockout'],
            yticklabels=['No Stockout','Stockout'],
            annot_kws={'size':14, 'weight':'bold'})
ax.set_xlabel('Predicted', fontsize=11)
ax.set_ylabel('Actual', fontsize=11)
ax.set_title(f'Confusion Matrix - Stockout Prediction\n{model_info["best_classification_model"]}',
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(CHART_PATH + 'chart8_confusion_matrix.png', bbox_inches='tight')
plt.close()
print("   Saved: chart8_confusion_matrix.png")

print("\n" + "=" * 65)
print("  ALL CHARTS SAVED!")
print(f"  Location: {CHART_PATH}")
print("=" * 65)
print("  Next: Run step4_predict_new.py")
print("=" * 65)
