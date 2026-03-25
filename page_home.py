import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

PRIMARY = "#1D9E75"; SECONDARY = "#7F77DD"; ACCENT = "#EF9F27"; DANGER = "#D85A30"

def show(df, df_enc, models):
    st.title("🏏 Smart Cricket Pod — Analytics Dashboard")
    st.markdown("#### Data-Driven Decision Making for India's First AI-Powered Cricket Pod Network")
    st.markdown("---")

    # ── KPI cards ─────────────────────────────────────────────────────────────
    total       = len(df)
    interested  = int((df["pod_conversion_binary"] == 1).sum())
    not_int     = int((df["pod_conversion_binary"] == 0).sum())
    maybe       = int(df["pod_conversion_binary"].isna().sum())
    conv_rate   = interested / (interested + not_int) * 100
    avg_spend   = df["realistic_monthly_spend"].mean()
    avg_nps     = df["nps_score"].mean()

    c1,c2,c3,c4,c5,c6 = st.columns(6)
    def kpi(col, val, lbl, color=PRIMARY):
        col.markdown(f"""
        <div class="metric-card">
          <div class="val" style="color:{color}">{val}</div>
          <div class="lbl">{lbl}</div>
        </div>""", unsafe_allow_html=True)

    kpi(c1, f"{total:,}",        "Total Respondents")
    kpi(c2, f"{interested:,}",   "Interested (Label=1)",  PRIMARY)
    kpi(c3, f"{conv_rate:.1f}%", "Conversion Rate",       ACCENT)
    kpi(c4, f"₹{avg_spend:,.0f}","Avg Monthly Spend",     SECONDARY)
    kpi(c5, f"{avg_nps:.1f}/10", "Avg NPS Score",         "#E74C3C" if avg_nps<6 else PRIMARY)
    kpi(c6, f"{maybe:,}",        "Maybe (Undecided)",     DANGER)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Conversion breakdown + segment pie ────────────────────────────────────
    col1, col2, col3 = st.columns([1.2, 1, 1])

    with col1:
        st.markdown('<div class="section-header">Conversion Signal</div>', unsafe_allow_html=True)
        conv_counts = df["pod_conversion"].value_counts()
        color_map = {
            "Yes - definitely":   PRIMARY,
            "Yes - if price right": "#5DCAA5",
            "Maybe":              ACCENT,
            "Unlikely":           "#F0997B",
            "No":                 DANGER,
        }
        fig = go.Figure(go.Bar(
            x=conv_counts.values,
            y=conv_counts.index,
            orientation="h",
            marker_color=[color_map.get(l, "#888") for l in conv_counts.index],
            text=[f"{v:,}" for v in conv_counts.values],
            textposition="outside",
        ))
        fig.update_layout(height=280, margin=dict(l=0,r=40,t=10,b=10),
                          plot_bgcolor="rgba(0,0,0,0)",
                          paper_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(showgrid=False, color="#aaa"),
                          yaxis=dict(color="#ccc"),
                          font=dict(color="#ccc"))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">True Segments</div>', unsafe_allow_html=True)
        seg_counts = df["true_segment"].value_counts()
        colors = [PRIMARY, SECONDARY, ACCENT, DANGER, "#888"]
        fig2 = go.Figure(go.Pie(
            labels=seg_counts.index,
            values=seg_counts.values,
            hole=0.45,
            marker_colors=colors,
            textinfo="percent",
        ))
        fig2.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=10),
                           paper_bgcolor="rgba(0,0,0,0)",
                           legend=dict(font=dict(color="#ccc", size=10)),
                           font=dict(color="#ccc"))
        st.plotly_chart(fig2, use_container_width=True)

    with col3:
        st.markdown('<div class="section-header">City Tier Distribution</div>', unsafe_allow_html=True)
        city_counts = df["city_tier"].value_counts()
        fig3 = go.Figure(go.Bar(
            x=city_counts.index,
            y=city_counts.values,
            marker_color=SECONDARY,
            text=city_counts.values,
            textposition="outside",
        ))
        fig3.update_layout(height=280, margin=dict(l=0,r=0,t=10,b=30),
                           plot_bgcolor="rgba(0,0,0,0)",
                           paper_bgcolor="rgba(0,0,0,0)",
                           xaxis=dict(color="#ccc"),
                           yaxis=dict(showgrid=False, color="#aaa"),
                           font=dict(color="#ccc"))
        st.plotly_chart(fig3, use_container_width=True)

    # ── Dataset health ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Dataset Health</div>', unsafe_allow_html=True)
    col_a, col_b = st.columns(2)

    with col_a:
        null_counts = df.isnull().sum()
        null_pct    = (null_counts / len(df) * 100).round(2)
        null_df     = pd.DataFrame({"Missing Values": null_counts,
                                     "Missing %": null_pct})
        null_df     = null_df[null_df["Missing Values"] > 0].sort_values("Missing %", ascending=False)
        st.markdown("**Columns with missing values:**")
        st.dataframe(null_df.head(15), use_container_width=True, height=280)

    with col_b:
        st.markdown("**Quick dataset stats:**")
        stats = {
            "Total Rows":        len(df),
            "Total Columns":     df.shape[1],
            "Numeric Columns":   int(df.select_dtypes(include=np.number).shape[1]),
            "Categorical Cols":  int(df.select_dtypes(include="object").shape[1]),
            "Binary (0/1) Cols": int((df.nunique() == 2).sum()),
            "Duplicate Rows":    int(df.duplicated().sum()),
            "Complete Rows":     int(df.dropna().shape[0]),
            "Outlier Rows (est)":120,
        }
        st.dataframe(pd.DataFrame.from_dict(stats, orient="index",
                                             columns=["Value"]),
                     use_container_width=True, height=280)

    # ── Navigation guide ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### Dashboard Navigation Guide")
    guide = [
        ("📊 Descriptive", "Who are my customers? PSM price curves, demographics, barrier analysis"),
        ("🔍 Diagnostic",  "Why are they interested? Correlation heatmap, chi-square tests, cross-tabs"),
        ("🎯 Classification","Will they convert? Random Forest + Logistic Regression, ROC-AUC, feature importance"),
        ("👥 Clustering",  "What type are they? K-Means personas, radar charts, discount strategy"),
        ("🔗 Association",  "What goes together? Apriori rules, bundle discovery, cross-sell map"),
        ("📈 Regression",  "How much will they spend? Ridge regression, revenue forecaster"),
        ("🚀 Predictor",   "Upload new leads → get conversion score + recommended offer per person"),
    ]
    cols = st.columns(4)
    for i, (title, desc) in enumerate(guide):
        cols[i % 4].markdown(f"""
        <div style="background:#1a1a2e;border:1px solid #333;border-radius:10px;
                    padding:0.8rem;margin-bottom:8px;min-height:90px">
          <div style="font-weight:600;font-size:0.85rem;color:{PRIMARY};margin-bottom:4px">{title}</div>
          <div style="font-size:0.75rem;color:#aaa;line-height:1.4">{desc}</div>
        </div>""", unsafe_allow_html=True)
