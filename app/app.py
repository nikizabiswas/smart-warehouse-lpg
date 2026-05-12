"""
================================================================================
SMART WAREHOUSE DEMAND PREDICTION — STREAMLIT APP
Mitra Bharatgas Agency | Murshidabad, West Bengal
================================================================================
Run with:  streamlit run app/app.py
From the smart_warehouse_v2/ directory.
================================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Warehouse — LPG Demand Prediction",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
ZONES       = ['Bali','Berhampore','Domkal','Farakka','Jangipur',
               'Kandi','Lalbagh','Raghunathganj','Samserganj','Suti']
ZONE_ENC    = {z: i for i, z in enumerate(ZONES)}
SUBSIDY_ENC = {'Non-Subsidized': 0, 'PMUY': 1}
MONTH_NAMES = ['Jan','Feb','Mar','Apr','May','Jun',
               'Jul','Aug','Sep','Oct','Nov','Dec']
SEASON_MAP  = {1:'Winter',2:'Winter',3:'Summer',4:'Summer',5:'Summer',
               6:'Monsoon',7:'Monsoon',8:'Monsoon',9:'Monsoon',
               10:'Autumn',11:'Autumn',12:'Winter'}
FESTIVAL_MONTHS = {1,8,9,10,11}   # Makar Sankranti, Eid, Durga Puja, Diwali
WINTER_MONTHS   = {11,12,1,2}

COLORS = ['#1A5276','#1E8449','#B7950B','#C0392B','#7D3C98',
          '#117A65','#784212','#1F618D','#7B241C','#1E8449']

ZONE_SUMMARY = {
    'Bali':          {'records':504,'avg_demand':1.03,'total_demand':518,'stockouts':44,'avg_lead':4.11,'avg_closing':751.74,'units_short':87,'avg_family':4.36,'avg_income':23218.78},
    'Berhampore':    {'records':504,'avg_demand':1.02,'total_demand':512,'stockouts':59,'avg_lead':6.64,'avg_closing':610.70,'units_short':121,'avg_family':4.46,'avg_income':23648.97},
    'Domkal':        {'records':504,'avg_demand':1.03,'total_demand':519,'stockouts':42,'avg_lead':4.17,'avg_closing':918.08,'units_short':84,'avg_family':4.52,'avg_income':23040.33},
    'Farakka':       {'records':504,'avg_demand':1.03,'total_demand':520,'stockouts':56,'avg_lead':6.72,'avg_closing':625.06,'units_short':109,'avg_family':4.48,'avg_income':23477.44},
    'Jangipur':      {'records':464,'avg_demand':1.02,'total_demand':475,'stockouts':45,'avg_lead':5.43,'avg_closing':700.08,'units_short':81,'avg_family':4.41,'avg_income':23850.18},
    'Kandi':         {'records':504,'avg_demand':1.01,'total_demand':511,'stockouts':33,'avg_lead':3.25,'avg_closing':849.39,'units_short':68,'avg_family':4.49,'avg_income':23038.88},
    'Lalbagh':       {'records':504,'avg_demand':1.03,'total_demand':517,'stockouts':38,'avg_lead':4.33,'avg_closing':848.25,'units_short':70,'avg_family':4.55,'avg_income':24031.93},
    'Raghunathganj': {'records':504,'avg_demand':1.02,'total_demand':514,'stockouts':58,'avg_lead':5.58,'avg_closing':810.02,'units_short':118,'avg_family':4.36,'avg_income':23234.22},
    'Samserganj':    {'records':504,'avg_demand':1.01,'total_demand':511,'stockouts':41,'avg_lead':3.50,'avg_closing':980.46,'units_short':76,'avg_family':4.27,'avg_income':22520.26},
    'Suti':          {'records':504,'avg_demand':1.03,'total_demand':520,'stockouts':45,'avg_lead':5.47,'avg_closing':732.42,'units_short':84,'avg_family':4.64,'avg_income':24166.97},
}

BASE_DIR  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, 'models')
OUT_DIR   = os.path.join(BASE_DIR, 'outputs')
CHART_DIR = os.path.join(BASE_DIR, 'charts')
DATA_DIR  = os.path.join(BASE_DIR, 'data')

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODELS  (cached so they only load once)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    reg      = joblib.load(os.path.join(MODEL_DIR, 'best_demand_model.pkl'))
    clf      = joblib.load(os.path.join(MODEL_DIR, 'best_stockout_model.pkl'))
    scaler   = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    encoders = joblib.load(os.path.join(MODEL_DIR, 'label_encoders.pkl'))
    feat_cols= joblib.load(os.path.join(MODEL_DIR, 'feature_cols.pkl'))
    info     = joblib.load(os.path.join(MODEL_DIR, 'model_info.pkl'))
    return reg, clf, scaler, encoders, feat_cols, info

@st.cache_data
def load_outputs():
    pred_df = pd.read_csv(os.path.join(OUT_DIR, 'test_predictions.csv'))
    reg_df  = pd.read_csv(os.path.join(OUT_DIR, 'regression_model_comparison.csv'))
    clf_df  = pd.read_csv(os.path.join(OUT_DIR, 'classification_model_comparison.csv'))
    fut_df  = pd.read_csv(os.path.join(OUT_DIR, 'future_predictions.csv'))
    return pred_df, reg_df, clf_df, fut_df

@st.cache_data
def load_raw_data():
    train = pd.read_excel(
        os.path.join(DATA_DIR, 'Warehouse_Demand_Realistic_v2.xlsx'),
        sheet_name='Train_80pct', header=1)
    for c in ['Month_Number','Year','Actual_Demand (cylinders)',
              'Stockout_Occurred','Zone_Opening_Stock','Monthly_Income (₹)',
              'LPG_Price_per_Cylinder (₹)','No_of_Family_Members']:
        train[c] = pd.to_numeric(train[c], errors='coerce')
    return train

reg_model, clf_model, scaler, encoders, feat_cols, model_info = load_models()

# Dark theme defaults for all Plotly charts
PLOTLY_DARK = dict(
    plot_bgcolor='#1A1D24',
    paper_bgcolor='#1A1D24',
    font_color='#FAFAFA',
    xaxis=dict(gridcolor='#2C2F36', linecolor='#3D4048', tickcolor='#9BAAB8'),
    yaxis=dict(gridcolor='#2C2F36', linecolor='#3D4048', tickcolor='#9BAAB8'),
    legend=dict(bgcolor='#1A1D24', bordercolor='#3D4048'),
)
pred_df, reg_df, clf_df, fut_df = load_outputs()
raw_train = load_raw_data()

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Force dark everywhere ── */
    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stApp"], .main, .block-container {
        background-color: #0E1117 !important;
        color: #FAFAFA !important;
    }

    /* Sidebar — fully dark */
    [data-testid="stSidebar"] {
        background-color: #1A1D24 !important;
        border-right: 1px solid #2C2F36;
    }
    [data-testid="stSidebar"] * {
        color: #FAFAFA !important;
    }
    [data-testid="stSidebar"] .stRadio label {
        color: #FAFAFA !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: #2C2F36 !important;
    }

    /* Main header banner */
    .main-header {
        background: linear-gradient(135deg, #1F3864 0%, #2874A6 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .main-header h1 { color: #FFFFFF; font-size: 1.9rem; margin: 0; font-weight: 700; }
    .main-header p  { color: #AED6F1; font-size: 0.95rem; margin: 0.3rem 0 0 0; }

    /* KPI cards — dark style */
    .kpi-card {
        background: #1A1D24;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        border-left: 5px solid #2874A6;
        box-shadow: 0 2px 12px rgba(0,0,0,0.4);
        margin-bottom: 0.5rem;
    }
    .kpi-card .kpi-value { font-size: 1.7rem; font-weight: 800; color: #5DADE2; }
    .kpi-card .kpi-label { font-size: 0.78rem; color: #9BAAB8; font-weight: 500;
                           text-transform: uppercase; letter-spacing: 0.05em; }

    /* Risk badges */
    .risk-high   { background:#C0392B; color:#fff; padding:0.3rem 0.8rem;
                   border-radius:20px; font-weight:700; font-size:0.85rem; }
    .risk-medium { background:#B7950B; color:#fff; padding:0.3rem 0.8rem;
                   border-radius:20px; font-weight:700; font-size:0.85rem; }
    .risk-low    { background:#1E8449; color:#fff; padding:0.3rem 0.8rem;
                   border-radius:20px; font-weight:700; font-size:0.85rem; }

    /* Prediction result box */
    .pred-box {
        background: #1A1D24;
        border: 2px solid #2874A6;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    .pred-box h2 { color: #AED6F1; font-size: 1.1rem; margin: 0 0 0.8rem 0; }

    /* Section titles */
    .section-title {
        font-size: 1.15rem;
        font-weight: 700;
        color: #AED6F1;
        border-bottom: 2px solid #2874A6;
        padding-bottom: 0.4rem;
        margin: 1rem 0 0.8rem 0;
    }

    /* Streamlit widgets — dark inputs */
    .stSelectbox > div, .stNumberInput > div, .stSlider {
        background-color: #1A1D24 !important;
    }
    .stTextInput input, .stNumberInput input {
        background-color: #2C2F36 !important;
        color: #FAFAFA !important;
        border-color: #3D4048 !important;
    }

    /* DataFrame table */
    [data-testid="stDataFrame"] {
        background-color: #1A1D24 !important;
    }

    /* Tab buttons */
    .stTabs [data-baseweb="tab"] {
        background-color: #1A1D24 !important;
        color: #9BAAB8 !important;
        border-radius: 6px 6px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2874A6 !important;
        color: #FFFFFF !important;
    }

    /* Form submit button */
    .stFormSubmitButton button {
        background: linear-gradient(135deg, #1F3864, #2874A6) !important;
        color: white !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 8px !important;
    }
    .stFormSubmitButton button:hover {
        background: linear-gradient(135deg, #2874A6, #1A5276) !important;
    }

    /* Dividers */
    hr { border-color: #2C2F36 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏭 Smart Warehouse")
    st.markdown("**Mitra Bharatgas Agency**")
    st.markdown("Murshidabad, West Bengal")
    st.divider()
    page = st.radio(
        "Navigate to",
        ["🏠 Dashboard",
         "🔮 Predict Demand",
         "📊 Model Performance",
         "🗺️ Zone Analysis",
         "📈 Data Explorer"],
        label_visibility="collapsed"
    )
    st.divider()
    st.caption(f"**Best Demand Model:** {model_info['best_regression_model']}")
    st.caption(f"**R² Score:** {model_info['regression_r2']:.4f}")
    st.caption(f"**Best Stockout Model:** {model_info['best_classification_model']}")
    st.caption(f"**Stockout Accuracy:** {model_info['classification_accuracy']*100:.1f}%")
    st.divider()
    st.caption("Dataset v2 · 5,000 records · Jan 2022–Dec 2024 · 10 Zones")

# ─────────────────────────────────────────────────────────────────────────────
# HELPER: RUN PREDICTION
# ─────────────────────────────────────────────────────────────────────────────
def run_prediction(zone, year, month, income, subsidy, lpg_price,
                   family_members, adults, children,
                   opening_stock, ordered, lead_time, delivered,
                   damaged, closing_stock):
    is_winter   = 1 if month in WINTER_MONTHS else 0
    is_festival = 1 if month in FESTIVAL_MONTHS else 0
    days        = [31,28,31,30,31,30,31,31,30,31,30,31][month-1]

    row = {
        'Warehouse_Zone'            : ZONE_ENC[zone],
        'Year'                      : year,
        'Month_Number'              : month,
        'Is_Winter'                 : is_winter,
        'Is_Festival_Month'         : is_festival,
        'Days_in_Month'             : days,
        'No_of_Family_Members'      : family_members,
        'No_of_Adults'              : adults,
        'No_of_Children'            : children,
        'Monthly_Income (₹)'        : income,
        'Subsidy_Type'              : SUBSIDY_ENC[subsidy],
        'LPG_Price_per_Cylinder (₹)': lpg_price,
        'Zone_Opening_Stock'        : opening_stock,
        'Zone_Cylinders_Ordered'    : ordered,
        'Lead_Time_Days'            : lead_time,
        'Zone_Cylinders_Delivered'  : delivered,
        'Damaged_Cylinders'         : damaged,
        'Zone_Closing_Stock'        : closing_stock,
    }

    input_df     = pd.DataFrame([{c: row[c] for c in feat_cols}])
    input_scaled = scaler.transform(input_df)

    demand_pred   = max(1, round(float(reg_model.predict(input_scaled)[0]), 2))
    stockout_pred = int(clf_model.predict(input_scaled)[0])
    stockout_prob = float(clf_model.predict_proba(input_scaled)[0][1]) * 100

    safety_stock      = max(1, round(demand_pred * 0.15))
    reorder_point     = max(1, round(demand_pred * 0.25))
    recommended_order = round(demand_pred * 1.10 + safety_stock)

    risk = "HIGH"   if stockout_prob > 60 else \
           "MEDIUM" if stockout_prob > 30 else "LOW"

    return {
        'demand'           : demand_pred,
        'stockout_prob'    : stockout_prob,
        'stockout_pred'    : stockout_pred,
        'risk'             : risk,
        'safety_stock'     : safety_stock,
        'reorder_point'    : reorder_point,
        'recommended_order': recommended_order,
        'is_winter'        : is_winter,
        'is_festival'      : is_festival,
        'season'           : SEASON_MAP[month],
    }

# =============================================================================
# PAGE 1: DASHBOARD
# =============================================================================
if page == "🏠 Dashboard":
    st.markdown("""
    <div class="main-header">
        <h1>🏭 Smart Warehouse — LPG Demand Prediction</h1>
        <p>Mitra Bharatgas Agency &nbsp;|&nbsp; Murshidabad, West Bengal &nbsp;|&nbsp; Dataset v2 &nbsp;|&nbsp; ML-Based Demand & Stockout Forecasting</p>
    </div>
    """, unsafe_allow_html=True)

    # --- Top KPI Row ---------------------------------------------------------
    total_stockouts = sum(z['stockouts'] for z in ZONE_SUMMARY.values())
    total_demand    = sum(z['total_demand'] for z in ZONE_SUMMARY.values())
    stockout_rate   = round(total_stockouts / sum(z['records'] for z in ZONE_SUMMARY.values()) * 100, 1)
    avg_income      = round(np.mean([z['avg_income'] for z in ZONE_SUMMARY.values()]))

    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">5,000</div>
            <div class="kpi-label">Total Records</div></div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">10</div>
            <div class="kpi-label">Warehouse Zones</div></div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{total_demand:,}</div>
            <div class="kpi-label">Total Cylinders Demanded</div></div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">{stockout_rate}%</div>
            <div class="kpi-label">Overall Stockout Rate</div></div>""", unsafe_allow_html=True)
    with k5:
        st.markdown(f"""<div class="kpi-card">
            <div class="kpi-value">₹{avg_income:,}</div>
            <div class="kpi-label">Avg Monthly Income</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Charts Row 1: Stockout by Zone + Monthly Trend ----------------------
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">📍 Stockout Rate by Zone</div>', unsafe_allow_html=True)
        zone_df = pd.DataFrame([
            {'Zone': z, 'Stockout Rate (%)': round(d['stockouts']/d['records']*100,1),
             'Records': d['records']}
            for z, d in ZONE_SUMMARY.items()
        ]).sort_values('Stockout Rate (%)', ascending=False)
        avg_so = zone_df['Stockout Rate (%)'].mean()
        zone_df['Color'] = zone_df['Stockout Rate (%)'].apply(
            lambda x: '#C0392B' if x > avg_so else '#1E8449')
        fig = go.Figure(go.Bar(
            x=zone_df['Zone'], y=zone_df['Stockout Rate (%)'],
            marker_color=zone_df['Color'],
            text=zone_df['Stockout Rate (%)'].apply(lambda x: f'{x:.1f}%'),
            textposition='outside',
        ))
        fig.add_hline(y=avg_so, line_dash='dash', line_color='orange',
                      annotation_text=f'Avg {avg_so:.1f}%')
        fig.update_layout(height=320, margin=dict(t=20,b=20,l=20,r=20),
                          plot_bgcolor='#1A1D24', paper_bgcolor='#1A1D24',
                          xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-title">📅 Monthly Demand Trend</div>', unsafe_allow_html=True)
        monthly = (raw_train.groupby('Month_Number')['Actual_Demand (cylinders)']
                   .mean().reset_index())
        monthly['Month'] = monthly['Month_Number'].apply(lambda x: MONTH_NAMES[x-1])
        monthly['Season'] = monthly['Month_Number'].apply(lambda x: SEASON_MAP[x])
        season_color = {'Winter':'#5DADE2','Summer':'#F39C12',
                        'Monsoon':'#27AE60','Autumn':'#E67E22'}
        monthly['Color'] = monthly['Season'].map(season_color)
        fig2 = go.Figure(go.Bar(
            x=monthly['Month'], y=monthly['Actual_Demand (cylinders)'],
            marker_color=monthly['Color'],
            text=monthly['Actual_Demand (cylinders)'].apply(lambda x: f'{x:.2f}'),
            textposition='outside',
        ))
        for season, color in season_color.items():
            fig2.add_trace(go.Scatter(x=[None], y=[None], mode='markers',
                marker=dict(color=color, size=10, symbol='square'),
                name=season, showlegend=True))
        fig2.update_layout(height=320, margin=dict(t=20,b=20,l=20,r=20),
                           plot_bgcolor='#1A1D24', paper_bgcolor='#1A1D24',
                           legend=dict(orientation='h', y=1.12))
        st.plotly_chart(fig2, use_container_width=True)

    # --- Charts Row 2: Zone Stats Table + Future Predictions ------------------
    col3, col4 = st.columns([1.1, 0.9])

    with col3:
        st.markdown('<div class="section-title">🗂️ Zone Summary Statistics</div>', unsafe_allow_html=True)
        zone_tbl = pd.DataFrame([
            {
                'Zone': z,
                'Stockouts': d['stockouts'],
                'SO Rate %': f"{d['stockouts']/d['records']*100:.1f}%",
                'Avg Lead Time': f"{d['avg_lead']:.1f}d",
                'Avg Closing Stock': f"{d['avg_closing']:.0f}",
                'Avg Income ₹': f"{d['avg_income']:,.0f}",
            }
            for z, d in ZONE_SUMMARY.items()
        ])
        st.dataframe(zone_tbl, use_container_width=True, hide_index=True,
                     height=280)

    with col4:
        st.markdown('<div class="section-title">🔮 2025 Scenario Forecasts</div>', unsafe_allow_html=True)
        risk_color_map = {'LOW RISK':'#1E8449','MEDIUM RISK':'#B7950B','HIGH RISK':'#C0392B'}
        fut_short = fut_df.copy()
        fut_short['Zone'] = fut_short['Scenario'].apply(lambda s: s.split('|')[0].strip())
        fut_short['Month/Event'] = fut_short['Scenario'].apply(
            lambda s: s.split('|')[1].strip() if '|' in s else s)
        fut_short['Risk'] = fut_short['Risk_Level']
        display_cols = ['Zone','Month/Event','Predicted_Demand','Stockout_Probability_%','Risk']
        st.dataframe(
            fut_short[display_cols].rename(columns={
                'Predicted_Demand':'Demand',
                'Stockout_Probability_%':'SO Prob %'
            }),
            use_container_width=True, hide_index=True, height=280
        )

    # --- Model Performance Summary -------------------------------------------
    st.markdown('<div class="section-title">🤖 Model Performance Summary</div>', unsafe_allow_html=True)
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        st.metric("Best Demand Model", model_info['best_regression_model'])
    with mc2:
        st.metric("R² Score", f"{model_info['regression_r2']:.4f}",
                  help="Closer to 1.0 is better")
    with mc3:
        st.metric("Best Stockout Model", model_info['best_classification_model'])
    with mc4:
        st.metric("Stockout Accuracy", f"{model_info['classification_accuracy']*100:.2f}%")

# =============================================================================
# PAGE 2: PREDICT DEMAND
# =============================================================================
elif page == "🔮 Predict Demand":
    st.markdown("""
    <div class="main-header">
        <h1>🔮 Predict LPG Demand</h1>
        <p>Enter zone and household details to get demand forecast and stockout risk assessment</p>
    </div>
    """, unsafe_allow_html=True)

    with st.form("prediction_form"):
        st.markdown('<div class="section-title">📍 Zone & Time</div>', unsafe_allow_html=True)
        fc1, fc2, fc3 = st.columns(3)
        with fc1:
            zone = st.selectbox("Warehouse Zone", ZONES,
                                help="Select the warehouse distribution zone")
        with fc2:
            year = st.selectbox("Year", [2024, 2025, 2026], index=1)
        with fc3:
            month = st.selectbox("Month", range(1,13),
                                 format_func=lambda x: MONTH_NAMES[x-1])

        st.markdown('<div class="section-title">👨‍👩‍👧 Family Profile</div>', unsafe_allow_html=True)
        ff1, ff2, ff3, ff4 = st.columns(4)
        zs = ZONE_SUMMARY[zone]
        with ff1:
            family_members = st.number_input("Total Family Members", 1, 15,
                                             int(zs['avg_family']),
                                             help="Total people in household")
        with ff2:
            adults   = st.number_input("Adults", 1, 10, 2)
        with ff3:
            children = st.number_input("Children", 0, 10,
                                       max(0, int(zs['avg_family'])-2))
        with ff4:
            subsidy  = st.selectbox("Subsidy Type", ['PMUY','Non-Subsidized'])

        income = st.slider("Monthly Income (₹)", 5000, 80000,
                           int(zs['avg_income']), step=500,
                           help="Household monthly income")

        st.markdown('<div class="section-title">🏪 Market & Warehouse Operations</div>', unsafe_allow_html=True)
        fm1, fm2 = st.columns(2)
        with fm1:
            lpg_price = st.number_input("LPG Price per Cylinder (₹)",
                                        min_value=500, max_value=1500,
                                        value=916, step=1)
        with fm2:
            lead_time = st.number_input("Lead Time (Days)",
                                        min_value=1, max_value=30,
                                        value=int(round(zs['avg_lead'])))

        fo1, fo2, fo3, fo4, fo5 = st.columns(5)
        avg_cs = int(zs['avg_closing'])
        with fo1:
            opening_stock = st.number_input("Opening Stock", 0, 5000,
                                            avg_cs, step=10)
        with fo2:
            ordered = st.number_input("Cylinders Ordered", 0, 2000, 450, step=10)
        with fo3:
            delivered = st.number_input("Cylinders Delivered", 0, 2000, 420, step=10)
        with fo4:
            closing_stock = st.number_input("Closing Stock", 0, 5000, avg_cs, step=10)
        with fo5:
            damaged = st.number_input("Damaged Cylinders", 0, 50, 2)

        submitted = st.form_submit_button("🔮 Predict Now", use_container_width=True,
                                          type="primary")

    if submitted:
        result = run_prediction(
            zone, year, month, income, subsidy, lpg_price,
            family_members, adults, children,
            opening_stock, ordered, lead_time, delivered,
            damaged, closing_stock
        )

        st.markdown("---")
        st.markdown("### 📋 Prediction Results")

        risk_html = {
            'HIGH':   '<span class="risk-high">🔴 HIGH RISK</span>',
            'MEDIUM': '<span class="risk-medium">🟡 MEDIUM RISK</span>',
            'LOW':    '<span class="risk-low">🟢 LOW RISK</span>',
        }[result['risk']]

        info_tags = []
        if result['is_winter']:
            info_tags.append("❄️ Winter Month")
        if result['is_festival']:
            info_tags.append("🎉 Festival Month")
        info_tags.append(f"🌤️ {result['season']}")

        tags_html = " &nbsp; ".join(info_tags)

        st.markdown(f"""
        <div class="pred-box">
            <h2>Zone: {zone} &nbsp;|&nbsp; {MONTH_NAMES[month-1]} {year} &nbsp;|&nbsp; {tags_html}</h2>
            <hr style="border-color:#AED6F1; margin:0.5rem 0;">
        </div>
        """, unsafe_allow_html=True)

        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1:
            st.metric("📦 Predicted Demand",
                      f"{result['demand']:.2f} cyl",
                      help="Cylinders needed for this family/month")
        with rc2:
            st.metric("⚠️ Stockout Probability",
                      f"{result['stockout_prob']:.1f}%")
        with rc3:
            st.metric("🛒 Recommended Order",
                      f"{result['recommended_order']} cyl",
                      help="Demand × 1.10 + safety stock")
        with rc4:
            st.metric("🔒 Safety Stock Level",
                      f"{result['safety_stock']} cyl")

        ra, rb = st.columns(2)
        with ra:
            st.markdown(f"**Stockout Risk Level:** {risk_html}", unsafe_allow_html=True)
        with rb:
            st.metric("📍 Reorder Point", f"{result['reorder_point']} cyl",
                      help="Trigger reorder when stock falls to this level")

        # Gauge chart for stockout probability
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=result['stockout_prob'],
            number={'suffix': '%', 'font': {'size': 30}},
            title={'text': "Stockout Risk", 'font': {'size': 16}},
            gauge={
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': '#1F3864'},
                'steps': [
                    {'range': [0,  30], 'color': '#D5F5E3'},
                    {'range': [30, 60], 'color': '#FDEBD0'},
                    {'range': [60, 100],'color': '#FADBD8'},
                ],
                'threshold': {
                    'line': {'color': 'red', 'width': 3},
                    'thickness': 0.75,
                    'value': result['stockout_prob']
                }
            }
        ))
        fig_gauge.update_layout(height=260, margin=dict(t=30,b=10,l=20,r=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

# =============================================================================
# PAGE 3: MODEL PERFORMANCE
# =============================================================================
elif page == "📊 Model Performance":
    st.markdown("""
    <div class="main-header">
        <h1>📊 Model Performance</h1>
        <p>Comparison of all trained ML models for demand forecasting and stockout prediction</p>
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📉 Regression Models", "🎯 Classification Models", "🔍 Prediction Analysis"])

    with tab1:
        st.markdown('<div class="section-title">Regression Models — Predict Demand (cylinders)</div>', unsafe_allow_html=True)
        st.caption("R² closer to 1.0 = better fit. MAE and RMSE lower = less error.")

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            fig_r2 = px.bar(
                reg_df.sort_values('R2_Score'),
                x='R2_Score', y='Model', orientation='h',
                color='R2_Score', color_continuous_scale='Blues',
                text=reg_df.sort_values('R2_Score')['R2_Score'].apply(lambda x: f'{x:.4f}'),
                title='R² Score Comparison'
            )
            fig_r2.add_vline(x=0.9, line_dash='dash', line_color='red',
                             annotation_text='Target 0.90')
            fig_r2.update_layout(height=320, showlegend=False,
                                 plot_bgcolor='#1A1D24', coloraxis_showscale=False,
                                 margin=dict(t=40,b=20,l=10,r=20))
            st.plotly_chart(fig_r2, use_container_width=True)

        with col_r2:
            fig_mae = px.bar(
                reg_df.sort_values('MAE'),
                x='MAE', y='Model', orientation='h',
                color='MAE', color_continuous_scale='Reds_r',
                text=reg_df.sort_values('MAE')['MAE'].apply(lambda x: f'{x:.4f}'),
                title='MAE Comparison (lower = better)'
            )
            fig_mae.update_layout(height=320, showlegend=False,
                                  plot_bgcolor='#1A1D24', coloraxis_showscale=False,
                                  margin=dict(t=40,b=20,l=10,r=20))
            st.plotly_chart(fig_mae, use_container_width=True)

        st.dataframe(reg_df.style.highlight_max(subset=['R2_Score'], color='#D5F5E3')
                              .highlight_min(subset=['MAE','RMSE'], color='#D5F5E3')
                              .format({'R2_Score':'{:.4f}','MAE':'{:.4f}','RMSE':'{:.4f}'}),
                     use_container_width=True, hide_index=True)

    with tab2:
        st.markdown('<div class="section-title">Classification Models — Predict Stockout Risk (0 or 1)</div>', unsafe_allow_html=True)
        st.caption("F1 Score balances precision and recall. Accuracy alone can be misleading on imbalanced data (only 9.2% stockouts).")

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            metrics_long = clf_df.melt(id_vars='Model',
                value_vars=['Accuracy','Precision','Recall','F1_Score'],
                var_name='Metric', value_name='Score')
            fig_clf = px.bar(
                metrics_long, x='Model', y='Score', color='Metric',
                barmode='group', title='All Classification Metrics',
                color_discrete_sequence=COLORS
            )
            fig_clf.update_layout(height=350, plot_bgcolor='#1A1D24',
                                  margin=dict(t=40,b=20,l=10,r=20),
                                  xaxis_tickangle=-20)
            st.plotly_chart(fig_clf, use_container_width=True)

        with col_c2:
            fig_f1 = px.bar(
                clf_df.sort_values('F1_Score'),
                x='F1_Score', y='Model', orientation='h',
                color='F1_Score', color_continuous_scale='Greens',
                text=clf_df.sort_values('F1_Score')['F1_Score'].apply(lambda x: f'{x:.4f}'),
                title='F1 Score Comparison'
            )
            fig_f1.update_layout(height=350, showlegend=False, coloraxis_showscale=False,
                                 plot_bgcolor='#1A1D24', margin=dict(t=40,b=20,l=10,r=20))
            st.plotly_chart(fig_f1, use_container_width=True)

        st.dataframe(clf_df.style.highlight_max(subset=['Accuracy','Precision','Recall','F1_Score'], color='#D5F5E3')
                              .format({'Accuracy':'{:.4f}','Precision':'{:.4f}','Recall':'{:.4f}','F1_Score':'{:.4f}'}),
                     use_container_width=True, hide_index=True)

    with tab3:
        st.markdown('<div class="section-title">Test Set Prediction Analysis</div>', unsafe_allow_html=True)
        col_p1, col_p2 = st.columns(2)

        with col_p1:
            fig_scatter = px.scatter(
                pred_df, x='Actual_Demand', y='Predicted_Demand',
                opacity=0.5, title='Actual vs Predicted Demand',
                color_discrete_sequence=['#1A5276']
            )
            mn = pred_df['Actual_Demand'].min() - 0.1
            mx = pred_df['Actual_Demand'].max() + 0.1
            fig_scatter.add_shape(type='line', x0=mn, y0=mn, x1=mx, y1=mx,
                                  line=dict(color='red', dash='dash', width=2))
            fig_scatter.add_annotation(text='Perfect Prediction', x=mx, y=mx,
                                       showarrow=False, font=dict(color='red', size=10))
            fig_scatter.update_layout(height=320, plot_bgcolor='#1A1D24',
                                      margin=dict(t=40,b=20,l=20,r=20))
            st.plotly_chart(fig_scatter, use_container_width=True)

        with col_p2:
            errors = pred_df['Actual_Demand'] - pred_df['Predicted_Demand']
            fig_err = px.histogram(errors, nbins=40,
                                   title='Prediction Error Distribution',
                                   color_discrete_sequence=['#1A5276'])
            fig_err.add_vline(x=0, line_dash='dash', line_color='red',
                              annotation_text='Zero Error')
            fig_err.add_vline(x=errors.mean(), line_dash='dot', line_color='orange',
                              annotation_text=f'Mean={errors.mean():.3f}')
            fig_err.update_layout(height=320, plot_bgcolor='#1A1D24',
                                  showlegend=False,
                                  margin=dict(t=40,b=20,l=20,r=20))
            st.plotly_chart(fig_err, use_container_width=True)

        st.markdown("**Confusion Matrix — Stockout Prediction**")
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(pred_df['Actual_Stockout'], pred_df['Predicted_Stockout'])
        cm_df = pd.DataFrame(cm,
                             index=['Actual: No Stockout','Actual: Stockout'],
                             columns=['Pred: No Stockout','Pred: Stockout'])
        fig_cm = px.imshow(cm_df, text_auto=True, color_continuous_scale='Blues',
                           title='Confusion Matrix')
        fig_cm.update_layout(height=300, margin=dict(t=40,b=20,l=20,r=20))
        st.plotly_chart(fig_cm, use_container_width=True)

# =============================================================================
# PAGE 4: ZONE ANALYSIS
# =============================================================================
elif page == "🗺️ Zone Analysis":
    st.markdown("""
    <div class="main-header">
        <h1>🗺️ Zone-Level Analysis</h1>
        <p>Deep dive into warehouse zone performance, stockout risk, and demand patterns</p>
    </div>
    """, unsafe_allow_html=True)

    selected_zone = st.selectbox("Select a Zone for Deep Dive", ZONES)

    zd = ZONE_SUMMARY[selected_zone]
    z1, z2, z3, z4 = st.columns(4)
    with z1:
        st.metric("Total Records", f"{zd['records']:,}")
    with z2:
        st.metric("Stockout Records", f"{zd['stockouts']}",
                  delta=f"{zd['stockouts']/zd['records']*100:.1f}% rate",
                  delta_color="inverse")
    with z3:
        st.metric("Avg Lead Time", f"{zd['avg_lead']:.1f} days")
    with z4:
        st.metric("Avg Closing Stock", f"{zd['avg_closing']:.0f} cyl")

    st.divider()

    col_z1, col_z2 = st.columns(2)

    with col_z1:
        st.markdown('<div class="section-title">Monthly Demand Trend — ' + selected_zone + '</div>', unsafe_allow_html=True)
        zone_monthly = (raw_train[raw_train['Warehouse_Zone'] == selected_zone]
                        .groupby('Month_Number')['Actual_Demand (cylinders)']
                        .mean().reset_index())
        zone_monthly['Month'] = zone_monthly['Month_Number'].apply(lambda x: MONTH_NAMES[x-1])
        zone_monthly['Season'] = zone_monthly['Month_Number'].map(SEASON_MAP)
        sc = {'Winter':'#5DADE2','Summer':'#F39C12','Monsoon':'#27AE60','Autumn':'#E67E22'}
        zone_monthly['Color'] = zone_monthly['Season'].map(sc)
        fig_zm = go.Figure(go.Bar(
            x=zone_monthly['Month'], y=zone_monthly['Actual_Demand (cylinders)'],
            marker_color=zone_monthly['Color'],
            text=zone_monthly['Actual_Demand (cylinders)'].apply(lambda x: f'{x:.2f}'),
            textposition='outside',
        ))
        fig_zm.update_layout(height=320, plot_bgcolor='#1A1D24',
                             margin=dict(t=20,b=20,l=20,r=20))
        st.plotly_chart(fig_zm, use_container_width=True)

    with col_z2:
        st.markdown('<div class="section-title">All Zones — Comparative Metrics</div>', unsafe_allow_html=True)
        compare_df = pd.DataFrame([
            {'Zone': z,
             'Stockout Rate %': round(d['stockouts']/d['records']*100, 1),
             'Avg Lead Time':   d['avg_lead'],
             'Avg Closing Stock': d['avg_closing'],
             'Avg Income ₹':    d['avg_income']}
            for z, d in ZONE_SUMMARY.items()
        ])
        metric_choice = st.selectbox("Compare by", 
                                     ['Stockout Rate %','Avg Lead Time',
                                      'Avg Closing Stock','Avg Income ₹'])
        highlight = [selected_zone == z for z in compare_df['Zone']]
        colors_z = ['#C0392B' if h else '#1A5276' for h in highlight]
        fig_comp = go.Figure(go.Bar(
            x=compare_df['Zone'],
            y=compare_df[metric_choice],
            marker_color=colors_z,
            text=compare_df[metric_choice].apply(lambda x: f'{x:.1f}'),
            textposition='outside',
        ))
        fig_comp.update_layout(height=280, plot_bgcolor='#1A1D24',
                               margin=dict(t=20,b=20,l=20,r=20),
                               xaxis_tickangle=-30)
        st.plotly_chart(fig_comp, use_container_width=True)

    # Stockout trend across all zones bubble chart
    st.markdown('<div class="section-title">📊 Zone Risk Matrix — Stockout Rate vs Lead Time</div>', unsafe_allow_html=True)
    risk_matrix = pd.DataFrame([
        {'Zone': z,
         'Stockout Rate %': round(d['stockouts']/d['records']*100, 1),
         'Avg Lead Time':   d['avg_lead'],
         'Total Demand':    d['total_demand'],
         'Avg Income ₹':    d['avg_income']}
        for z, d in ZONE_SUMMARY.items()
    ])
    fig_bubble = px.scatter(
        risk_matrix,
        x='Avg Lead Time', y='Stockout Rate %',
        size='Total Demand', color='Zone',
        hover_data=['Avg Income ₹'],
        text='Zone',
        title='Zone Risk Matrix (bubble size = total demand, higher-right = more risk)',
        color_discrete_sequence=COLORS
    )
    fig_bubble.update_traces(textposition='top center')
    fig_bubble.update_layout(height=380, plot_bgcolor='#1A1D24',
                             margin=dict(t=50,b=20,l=20,r=20))
    st.plotly_chart(fig_bubble, use_container_width=True)

# =============================================================================
# PAGE 5: DATA EXPLORER
# =============================================================================
elif page == "📈 Data Explorer":
    st.markdown("""
    <div class="main-header">
        <h1>📈 Data Explorer</h1>
        <p>Explore and filter the raw training dataset interactively</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">🔎 Filters</div>', unsafe_allow_html=True)
    fe1, fe2, fe3 = st.columns(3)
    with fe1:
        sel_zones = st.multiselect("Filter by Zone", ZONES, default=ZONES[:3])
    with fe2:
        sel_years = st.multiselect("Filter by Year",
                                   sorted(raw_train['Year'].dropna().astype(int).unique()),
                                   default=[2022])
    with fe3:
        sel_months = st.multiselect("Filter by Month",
                                    list(range(1,13)),
                                    format_func=lambda x: MONTH_NAMES[x-1],
                                    default=list(range(1,4)))

    filtered = raw_train.copy()
    if sel_zones:
        filtered = filtered[filtered['Warehouse_Zone'].isin(sel_zones)]
    if sel_years:
        filtered = filtered[filtered['Year'].isin(sel_years)]
    if sel_months:
        filtered = filtered[filtered['Month_Number'].isin(sel_months)]

    st.markdown(f"**Showing {len(filtered):,} records**")

    fe_c1, fe_c2 = st.columns(2)

    with fe_c1:
        st.markdown('<div class="section-title">Income Distribution</div>', unsafe_allow_html=True)
        fig_inc = px.histogram(filtered, x='Monthly_Income (₹)',
                               color='Warehouse_Zone',
                               nbins=30, opacity=0.75,
                               color_discrete_sequence=COLORS)
        fig_inc.update_layout(height=300, plot_bgcolor='#1A1D24',
                              margin=dict(t=20,b=20,l=20,r=20))
        st.plotly_chart(fig_inc, use_container_width=True)

    with fe_c2:
        st.markdown('<div class="section-title">LPG Price Trend</div>', unsafe_allow_html=True)
        price_trend = (filtered.groupby(['Year','Month_Number'])
                       ['LPG_Price_per_Cylinder (₹)'].mean().reset_index())
        price_trend['Period'] = (price_trend['Year'].astype(str) + '-' +
                                 price_trend['Month_Number'].astype(str).str.zfill(2))
        fig_price = px.line(price_trend, x='Period', y='LPG_Price_per_Cylinder (₹)',
                            title='Avg LPG Price Over Time',
                            color_discrete_sequence=['#C0392B'])
        fig_price.update_layout(height=300, plot_bgcolor='#1A1D24',
                                margin=dict(t=20,b=20,l=20,r=20),
                                xaxis_tickangle=-45)
        st.plotly_chart(fig_price, use_container_width=True)

    st.markdown('<div class="section-title">Demand vs Family Size</div>', unsafe_allow_html=True)
    fig_fam = px.box(filtered, x='No_of_Family_Members',
                     y='Actual_Demand (cylinders)',
                     color='Warehouse_Zone',
                     color_discrete_sequence=COLORS,
                     title='Demand Distribution by Family Size')
    fig_fam.update_layout(height=350, plot_bgcolor='#1A1D24',
                          margin=dict(t=40,b=20,l=20,r=20))
    st.plotly_chart(fig_fam, use_container_width=True)

    with st.expander("📄 View Raw Data Table"):
        display_cols = ['Warehouse_Zone','Year','Month_Number','Actual_Demand (cylinders)',
                        'Stockout_Occurred','Monthly_Income (₹)','LPG_Price_per_Cylinder (₹)',
                        'No_of_Family_Members','Subsidy_Type','Zone_Opening_Stock',
                        'Zone_Closing_Stock','Lead_Time_Days']
        available = [c for c in display_cols if c in filtered.columns]
        st.dataframe(filtered[available].head(500), use_container_width=True, hide_index=True)
        st.caption("Showing up to 500 rows.")
