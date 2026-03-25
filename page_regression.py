import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PRIMARY = "#1D9E75"; SECONDARY = "#7F77DD"; ACCENT = "#EF9F27"; DANGER = "#D85A30"

COEF_LABELS = {
    "income_num":"Income","role_num":"Cricket Role","pod_interest":"Pod Interest",
    "practice_num":"Practice Days","data_importance":"Data Importance",
    "tech_num":"Tech Adoption","past_exp_count":"Past Experiences",
    "addon_count":"Add-on Interest","feat_count":"Feature Interest",
    "spend_num":"Current Rec Spend","nps_score":"NPS Score",
    "city_num":"City Tier","age_num":"Age","digital_num":"Digital Spend",
    "mem_num":"Membership WTP","frust_count":"Frustration Count",
    "dist_num":"Distance Tolerance","barrier_count":"Barrier Count",
}

def show(df, df_enc, models):
    st.title("📈 Regression — How Much Will They Spend?")
    st.markdown("Predicting **monthly spend** per customer using Ridge and Linear Regression.")
    st.markdown("---")

    results = models.get("all_results", {})
    if not results or "regression" not in results:
        st.error("Regression results not found.")
        return

    reg = results["regression"]
    ridge = reg["ridge"]
    lr    = reg["lr"]

    # ── Model metrics ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Model Performance</div>', unsafe_allow_html=True)
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Ridge R²",    f"{ridge['r2']:.4f}")
    c2.metric("Ridge RMSE",  f"₹{ridge['rmse']:.0f}")
    c3.metric("Ridge MAE",   f"₹{ridge['mae']:.0f}")
    c4.metric("LinReg R²",   f"{lr['r2']:.4f}")
    c5.metric("LinReg RMSE", f"₹{lr['rmse']:.0f}")
    c6.metric("LinReg MAE",  f"₹{lr['mae']:.0f}")

    st.markdown("---")

    # ── Actual vs Predicted scatter ───────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Actual vs Predicted Spend</div>',
                    unsafe_allow_html=True)
        y_test = ridge["y_test"]
        y_pred = ridge["y_pred"]
        fig_ap = go.Figure()
        fig_ap.add_trace(go.Scatter(
            x=y_test, y=y_pred, mode="markers",
            marker=dict(color=PRIMARY, size=4, opacity=0.5),
            name="Predictions",
        ))
        max_val = max(max(y_test), max(y_pred))
        fig_ap.add_trace(go.Scatter(
            x=[0, max_val], y=[0, max_val], mode="lines",
            line=dict(color=DANGER, dash="dash", width=1.5),
            name="Perfect fit",
        ))
        fig_ap.update_layout(
            height=360, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Actual ₹", color="#ccc", showgrid=True, gridcolor="#333"),
            yaxis=dict(title="Predicted ₹", color="#aaa", showgrid=True, gridcolor="#333"),
            legend=dict(font=dict(color="#ccc")),
            font=dict(color="#ccc"), margin=dict(t=10, b=40, l=0, r=0),
        )
        st.plotly_chart(fig_ap, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Residual Plot</div>',
                    unsafe_allow_html=True)
        residuals = np.array(y_pred) - np.array(y_test)
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(
            x=y_pred, y=residuals, mode="markers",
            marker=dict(color=SECONDARY, size=4, opacity=0.5),
            name="Residuals",
        ))
        fig_res.add_hline(y=0, line_color=DANGER, line_dash="dash", line_width=1.5)
        fig_res.update_layout(
            height=360, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Predicted ₹", color="#ccc", showgrid=True, gridcolor="#333"),
            yaxis=dict(title="Residual (Pred − Actual)", color="#aaa",
                       showgrid=True, gridcolor="#333"),
            font=dict(color="#ccc"), margin=dict(t=10, b=40, l=0, r=0),
        )
        st.plotly_chart(fig_res, use_container_width=True)

    st.markdown("---")

    # ── Coefficient importance ────────────────────────────────────────────────
    st.markdown('<div class="section-header">Feature Coefficients — Ridge Regression (|abs| importance)</div>',
                unsafe_allow_html=True)

    coef_imp = reg.get("coef_imp", {})
    if coef_imp:
        ci_series = pd.Series(coef_imp).sort_values(ascending=True)
        ci_labels = [COEF_LABELS.get(k, k) for k in ci_series.index]
        fig_coef = go.Figure(go.Bar(
            x=ci_series.values, y=ci_labels, orientation="h",
            marker_color=[PRIMARY if v >= ci_series.quantile(0.7) else ACCENT if v >= ci_series.quantile(0.4) else "#555"
                          for v in ci_series.values],
            text=[f"{v:.2f}" for v in ci_series.values], textposition="outside",
        ))
        fig_coef.update_layout(
            height=420, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, color="#aaa", title="|Coefficient|"),
            yaxis=dict(color="#ccc"),
            font=dict(color="#ccc"), margin=dict(t=10, b=20, l=0, r=80),
        )
        st.plotly_chart(fig_coef, use_container_width=True)

    st.markdown("---")

    # ── Spend by cluster ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Predicted Spend Distribution by Cluster</div>',
                unsafe_allow_html=True)

    if "cluster" in df_enc.columns and "realistic_monthly_spend" in df_enc.columns:
        persona_map = models.get("persona_map", {})
        spend_by_cluster = df_enc.groupby("cluster")["realistic_monthly_spend"].agg(
            ["mean","median","std"]).reset_index()
        spend_by_cluster["persona"] = spend_by_cluster["cluster"].map(persona_map)
        spend_by_cluster = spend_by_cluster.sort_values("mean", ascending=False)

        fig_clust_spend = go.Figure()
        colors = [PRIMARY, SECONDARY, ACCENT, DANGER, "#5DCAA5"]
        for i, row in spend_by_cluster.iterrows():
            c = colors[int(row["cluster"]) % len(colors)]
            fig_clust_spend.add_trace(go.Bar(
                name=str(row["persona"]),
                x=[str(row["persona"])],
                y=[row["mean"]],
                error_y=dict(type="data", array=[row["std"]], visible=True, color="#666"),
                marker_color=c,
                text=f"₹{row['mean']:.0f}",
                textposition="outside",
            ))
        fig_clust_spend.update_layout(
            height=340, barmode="group",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(color="#ccc"),
            yaxis=dict(title="Avg Monthly Spend (₹)", showgrid=False, color="#aaa"),
            legend=dict(font=dict(color="#ccc")),
            font=dict(color="#ccc"), margin=dict(t=10, b=60, l=0, r=0),
        )
        st.plotly_chart(fig_clust_spend, use_container_width=True)

    st.markdown("---")

    # ── Revenue forecaster ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Interactive Revenue Forecaster</div>',
                unsafe_allow_html=True)
    st.markdown("Estimate monthly revenue based on pod rollout assumptions.")

    fc1, fc2, fc3, fc4 = st.columns(4)
    n_pods       = fc1.slider("Number of pods",        1, 50, 3)
    sessions_day = fc2.slider("Sessions/pod/day",      5, 30, 15)
    avg_price    = fc3.slider("Avg price per session ₹", 100, 500, 220)
    occupancy    = fc4.slider("Occupancy rate %",      20, 100, 60)

    days_month = 26
    monthly_sessions = n_pods * sessions_day * days_month * (occupancy / 100)
    monthly_revenue  = monthly_sessions * avg_price
    annual_revenue   = monthly_revenue * 12

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Monthly Sessions",  f"{int(monthly_sessions):,}")
    m2.metric("Monthly Revenue",   f"₹{monthly_revenue:,.0f}")
    m3.metric("Annual Revenue",    f"₹{annual_revenue:,.0f}")
    m4.metric("Revenue per Pod/mo", f"₹{monthly_revenue/max(n_pods,1):,.0f}")

    # 12-month projection
    months  = list(range(1, 13))
    rev_proj = [monthly_revenue * (1 + 0.02) ** m for m in range(12)]
    fig_proj = go.Figure(go.Scatter(
        x=[f"M{m}" for m in months], y=rev_proj, mode="lines+markers",
        line=dict(color=PRIMARY, width=2.5),
        marker=dict(color=ACCENT, size=7),
        fill="tozeroy", fillcolor=f"{PRIMARY}22",
        text=[f"₹{v:,.0f}" for v in rev_proj], textposition="top center",
    ))
    fig_proj.update_layout(
        title="12-month revenue projection (2% MoM growth assumed)",
        height=300, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(color="#ccc"),
        yaxis=dict(title="Revenue ₹", color="#aaa", showgrid=False),
        font=dict(color="#ccc"), margin=dict(t=40, b=20, l=0, r=0),
    )
    st.plotly_chart(fig_proj, use_container_width=True)
