# 🏭 Smart Warehouse — LPG Demand Prediction
### Mitra Bharatgas Agency | Murshidabad, West Bengal

A machine learning system that predicts monthly LPG cylinder demand and stockout
risk across 10 warehouse zones in Murshidabad district.
Built with Scikit-learn, Streamlit, and Plotly.

---

## 📁 Project Structure

```
smart_warehouse_v2/
│
├── app/
│   ├── app.py                      ← Streamlit app (5 pages, dark theme)
│   ├── requirements.txt            ← pip dependencies for the app
│   └── README.md                   ← App-level quick-start notes
│
├── .streamlit/
│   └── config.toml                 ← Dark theme config (fixes white sidebar)
│
├── step1_data_preprocessing.py     ← Load, clean, encode, scale dataset
├── step2_model_training.py         ← Train 4 regression + 4 classification models
├── step3_visualization.py          ← Generate 8 chart PNGs
├── step4_predict_new.py            ← Batch-predict future 2025 scenarios
├── step5_dashboard.py              ← Master dashboard image + Excel report
│
├── data/
│   └── Warehouse_Demand_Realistic_v2.xlsx   ← 5,000-record dataset (v2)
│
├── models/                         ← Trained model artifacts (pre-built)
│   ├── best_demand_model.pkl       ← Random Forest regressor
│   ├── best_stockout_model.pkl     ← Gradient Boosting classifier
│   ├── scaler.pkl                  ← StandardScaler (fit on train set)
│   ├── label_encoders.pkl          ← Zone + Subsidy LabelEncoders
│   ├── feature_cols.pkl            ← Ordered list of 18 feature columns
│   └── model_info.pkl              ← Best model names + all metrics
│
├── outputs/                        ← Pre-computed results (app reads these)
│   ├── test_predictions.csv        ← Actual vs predicted on 1000 test rows
│   ├── regression_model_comparison.csv
│   ├── classification_model_comparison.csv
│   ├── future_predictions.csv      ← 2025 scenario forecasts from step4
│   ├── X_train.csv / X_test.csv   ← Scaled feature matrices
│   ├── y_train/test_demand.csv     ← Demand targets
│   ├── y_train/test_stock.csv      ← Stockout targets
│   └── smart_warehouse_v2_report.xlsx  ← Full Excel report (step5)
│
└── charts/                         ← Generated chart PNGs (step3 + step5)
    ├── chart1_regression_comparison.png
    ├── chart2_classification_comparison.png
    ├── chart3_actual_vs_predicted.png
    ├── chart4_error_distribution.png
    ├── chart5_monthly_demand_trend.png
    ├── chart6_feature_importance.png
    ├── chart7_stockout_by_zone.png
    ├── chart8_confusion_matrix.png
    └── dashboard_master.png
```

---

## 🚀 Run the Streamlit App

### Step 1 — Install dependencies
```bash
cd smart_warehouse_v2
pip install -r app/requirements.txt
```

### Step 2 — Launch
```bash
streamlit run app/app.py
```
Opens at **http://localhost:8501**

> The `models/` and `outputs/` folders are pre-built — no retraining needed.

---

## 🔁 Re-run the Full Pipeline (optional)

Only needed if you change the dataset or want to retrain from scratch:

```bash
python step1_data_preprocessing.py
python step2_model_training.py
python step3_visualization.py
python step4_predict_new.py
python step5_dashboard.py
```

---

## 📊 App Pages

| Page | What it shows |
|------|--------------|
| 🏠 **Dashboard** | 5 KPI cards · stockout-by-zone bar chart · seasonal demand trend · zone stats table · 2025 forecast table |
| 🔮 **Predict Demand** | Input form → live demand prediction + stockout probability gauge |
| 📊 **Model Performance** | R², MAE, F1 charts for all 4 models · confusion matrix · actual vs predicted scatter |
| 🗺️ **Zone Analysis** | Per-zone monthly demand · cross-zone metric comparison · risk bubble matrix |
| 📈 **Data Explorer** | Filter by zone/year/month → income distribution · LPG price trend · demand by family size |

---

## 🤖 Models

| Task | Best Model | Metric |
|------|-----------|--------|
| Demand Forecasting | Random Forest Regressor | R² = 0.2413 |
| Stockout Prediction | Gradient Boosting Classifier | Accuracy = 91.2%, F1 = 0.20 |

> The low R² on demand is expected — `Actual_Demand` is nearly binary (1 or 2 cylinders
> per family per month), making it a near-classification problem. The stockout model
> (91.2% accuracy) is the primary operational signal.

---

## 📦 Dataset — v2

| Property | Value |
|----------|-------|
| File | `Warehouse_Demand_Realistic_v2.xlsx` |
| Total records | 5,000 |
| Date range | January 2022 – December 2024 |
| Train / Test split | 80% / 20% (pre-split in Excel) |
| Sheet names | `Train_80pct`, `Test_20pct`, `Zone_Summary` |
| Zones | Bali, Berhampore, Domkal, Farakka, Jangipur, Kandi, Lalbagh, Raghunathganj, Samserganj, Suti |
| Features used | 18 columns (leakage columns excluded) |

---

## 🛠️ Tech Stack

| Layer | Libraries |
|-------|----------|
| Machine Learning | scikit-learn, joblib |
| App / UI | Streamlit, Plotly |
| Data | Pandas, NumPy, OpenPyXL |
| Pipeline charts | Matplotlib, Seaborn |

---

## 📄 License
MIT — free to use and modify.
