import streamlit as st
import pandas as pd
import numpy as np
import joblib, os, warnings
warnings.filterwarnings("ignore")

from preprocessing import encode

st.set_page_config(
    page_title="Smart Cricket Pod — Analytics Dashboard",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Brand colours ─────────────────────────────────────────────────────────────
PRIMARY   = "#1D9E75"
SECONDARY = "#7F77DD"
ACCENT    = "#EF9F27"
DANGER    = "#D85A30"
DARK      = "#1a1a2e"

st.markdown(f"""
<style>
    [data-testid="stSidebar"] {{ background-color: #0f0f1a; }}
    [data-testid="stSidebar"] * {{ color: #e0e0e0 !important; }}
    .metric-card {{
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid {PRIMARY}44;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        text-align: center;
        color: white;
    }}
    .metric-card .val {{ font-size: 2rem; font-weight: 700; color: {PRIMARY}; }}
    .metric-card .lbl {{ font-size: 0.78rem; color: #aaa; margin-top: 4px; }}
    .section-header {{
        font-size: 1.1rem; font-weight: 600;
        border-left: 4px solid {PRIMARY};
        padding-left: 0.7rem; margin: 1.2rem 0 0.8rem;
        color: inherit;
    }}
    .insight-box {{
        background: #1a1a2e; border-left: 3px solid {ACCENT};
        padding: 0.8rem 1rem; border-radius: 0 8px 8px 0;
        font-size: 0.85rem; color: #ddd; margin: 0.5rem 0;
    }}
    .stDataFrame {{ border-radius: 10px; }}
    div[data-testid="metric-container"] {{
        background: #1a1a2e; border: 1px solid #333;
        border-radius: 10px; padding: 0.5rem 1rem;
    }}
</style>
""", unsafe_allow_html=True)

# ── Load data & models ────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("cricket_pod_survey_data.csv")
    return df

@st.cache_resource
def load_models():
    model_dir = "models"
    out = {}
    for fname in os.listdir(model_dir):
        key = fname.replace(".pkl","")
        out[key] = joblib.load(os.path.join(model_dir, fname))
    return out

@st.cache_data
def get_encoded(df):
    df_enc = encode(df)
    # attach cluster labels
    models = load_models()
    if "kmeans" in models and "scaler_clust" in models and "cluster_features" in models:
        from preprocessing import get_cluster_features
        X_c = get_cluster_features(df_enc)
        X_cs = models["scaler_clust"].transform(X_c)
        df_enc["cluster"] = models["kmeans"].predict(X_cs)
        pmap = models.get("persona_map", {})
        df_enc["persona"] = df_enc["cluster"].map(pmap)
    return df_enc

# ── Session state ────────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.session_state.df     = load_data()
if "models" not in st.session_state:
    st.session_state.models = load_models()
if "df_enc" not in st.session_state:
    st.session_state.df_enc = get_encoded(st.session_state.df)

df     = st.session_state.df
models = st.session_state.models
df_enc = st.session_state.df_enc

# ── Sidebar navigation ────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏏 Smart Cricket Pod")
    st.markdown("**Data-Driven Analytics Platform**")
    st.markdown("---")
    page = st.radio("Navigate to", [
        "🏠  Home — Executive Summary",
        "📊  Descriptive Analysis",
        "🔍  Diagnostic Analysis",
        "🎯  Classification",
        "👥  Clustering — Personas",
        "🔗  Association Rule Mining",
        "📈  Regression — Spend Forecast",
        "🚀  New Customer Predictor",
    ])
    st.markdown("---")
    st.caption(f"Dataset: {len(df):,} respondents · {df.shape[1]} features")
    st.caption("v1.0 — Smart Cricket Pod")

# ── Route to pages ────────────────────────────────────────────────────────────
if   "Home"         in page:
    import page_home;          page_home.show(df, df_enc, models)
elif "Descriptive"  in page:
    import page_descriptive;   page_descriptive.show(df, df_enc, models)
elif "Diagnostic"   in page:
    import page_diagnostic;    page_diagnostic.show(df, df_enc, models)
elif "Classification" in page:
    import page_classification; page_classification.show(df, df_enc, models)
elif "Clustering"   in page:
    import page_clustering;    page_clustering.show(df, df_enc, models)
elif "Association"  in page:
    import page_association;   page_association.show(df, df_enc, models)
elif "Regression"   in page:
    import page_regression;    page_regression.show(df, df_enc, models)
elif "Predictor"    in page:
    import page_predictor;     page_predictor.show(df, df_enc, models)
