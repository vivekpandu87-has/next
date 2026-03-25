import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

PRIMARY = "#1D9E75"; SECONDARY = "#7F77DD"; ACCENT = "#EF9F27"; DANGER = "#D85A30"

def show(df, df_enc, models):
    st.title("🎯 Classification — Will This Customer Convert?")
    st.markdown("Predicting pod conversion using **Random Forest** and **Logistic Regression**.")
    st.markdown("---")

    results = models.get("all_results", {})
    if not results or "classification" not in results:
        st.error("Model results not found. Please ensure models/ folder is present.")
        return

    clf = results["classification"]
    rf  = clf["rf"]
    lr  = clf["lr"]

    # ── Model comparison KPIs ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Model Performance Comparison</div>',
                unsafe_allow_html=True)

    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    rf_vals = [rf["acc"], rf["prec"], rf["rec"], rf["f1"], rf["auc"]]
    lr_vals = [lr["acc"], lr["prec"], lr["rec"], lr["f1"], lr["auc"]]

    cols = st.columns(5)
    for i, (m, rv, lv) in enumerate(zip(metrics, rf_vals, lr_vals)):
        delta = rv - lv
        cols[i].metric(
            label=m,
            value=f"{rv:.3f}",
            delta=f"RF vs LR: {delta:+.3f}",
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Comparison bar chart
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(name="Random Forest", x=metrics, y=rf_vals,
                               marker_color=PRIMARY,
                               text=[f"{v:.3f}" for v in rf_vals], textposition="outside"))
    fig_comp.add_trace(go.Bar(name="Logistic Regression", x=metrics, y=lr_vals,
                               marker_color=SECONDARY,
                               text=[f"{v:.3f}" for v in lr_vals], textposition="outside"))
    fig_comp.update_layout(
        barmode="group", height=320,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(color="#ccc"), yaxis=dict(range=[0, 1.15], showgrid=False, color="#aaa"),
        legend=dict(font=dict(color="#ccc")),
        font=dict(color="#ccc"), margin=dict(t=10, b=10, l=0, r=0),
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("---")

    # ── ROC Curve ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">ROC Curve — Both Models</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=rf["fpr"], y=rf["tpr"], mode="lines",
            name=f"Random Forest (AUC={rf['auc']:.3f})",
            line=dict(color=PRIMARY, width=2.5),
        ))
        fig_roc.add_trace(go.Scatter(
            x=lr["fpr"], y=lr["tpr"], mode="lines",
            name=f"Logistic Regression (AUC={lr['auc']:.3f})",
            line=dict(color=SECONDARY, width=2, dash="dash"),
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Random baseline",
            line=dict(color="#555", width=1, dash="dot"),
        ))
        fig_roc.update_layout(
            height=380,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="False Positive Rate", color="#ccc",
                       showgrid=True, gridcolor="#333"),
            yaxis=dict(title="True Positive Rate", color="#aaa",
                       showgrid=True, gridcolor="#333"),
            legend=dict(font=dict(color="#ccc", size=11)),
            font=dict(color="#ccc"), margin=dict(t=10, b=40, l=0, r=0),
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # ── Confusion Matrix ──────────────────────────────────────────────────────
    with col2:
        st.markdown("**Confusion Matrix — Random Forest**")
        cm = np.array(rf["cm"])
        labels_cm = ["Not Interested (0)", "Interested (1)"]
        fig_cm = ff.create_annotated_heatmap(
            z=cm, x=labels_cm, y=labels_cm,
            colorscale=[[0, "#1a1a2e"], [1, PRIMARY]],
            annotation_text=cm.astype(str),
            showscale=True,
        )
        fig_cm.update_layout(
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#ccc"),
            margin=dict(t=30, b=60, l=0, r=0),
            xaxis=dict(title="Predicted", color="#ccc"),
            yaxis=dict(title="Actual", color="#ccc"),
        )
        st.plotly_chart(fig_cm, use_container_width=True)

        # Metrics table
        metrics_tbl = pd.DataFrame({
            "Metric": ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"],
            "Random Forest": [f"{rf[k]:.4f}" for k in ["acc","prec","rec","f1","auc"]],
            "Logistic Reg.": [f"{lr[k]:.4f}" for k in ["acc","prec","rec","f1","auc"]],
        })
        st.dataframe(metrics_tbl, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Feature Importance ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Feature Importance — Random Forest (Top 20)</div>',
                unsafe_allow_html=True)

    feat_imp = clf.get("feat_imp", {})
    if feat_imp:
        feat_names_map = {
            "role_num":"Cricket Role","pod_interest":"Pod Interest",
            "data_importance":"Data Importance","income_num":"Income",
            "past_exp_count":"Past Exp Count","tech_num":"Tech Adoption",
            "nps_score":"NPS Score","feat_count":"Feature Interest Count",
            "addon_count":"Add-on Interest","barrier_count":"Barrier Count",
            "age_num":"Age","city_num":"City Tier","spend_num":"Rec Spend",
            "practice_num":"Practice Days","frust_count":"Frustration Count",
            "bar_aidistrust":"Barrier: AI Distrust","bar_notserious":"Barrier: Not Serious",
            "past_boxcricket":"Past: Box Cricket","past_vr":"Past: VR Gaming",
            "feat_ai":"Feature: AI Analysis","mem_num":"Membership WTP",
            "dist_num":"Distance Tolerance","digital_num":"Digital Spend",
            "gender_num":"Gender","edu_num":"Education",
        }
        fi_series = pd.Series(feat_imp).sort_values(ascending=True)
        fi_labels = [feat_names_map.get(k, k) for k in fi_series.index]
        colors = [PRIMARY if v >= fi_series.quantile(0.75) else
                  ACCENT  if v >= fi_series.quantile(0.50) else "#555"
                  for v in fi_series.values]

        fig_fi = go.Figure(go.Bar(
            x=fi_series.values, y=fi_labels, orientation="h",
            marker_color=colors,
            text=[f"{v:.4f}" for v in fi_series.values], textposition="outside",
        ))
        fig_fi.update_layout(
            height=520, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, color="#aaa", title="Importance Score"),
            yaxis=dict(color="#ccc"),
            font=dict(color="#ccc"), margin=dict(t=10, b=20, l=0, r=80),
        )
        st.plotly_chart(fig_fi, use_container_width=True)

        top3 = list(pd.Series(feat_imp).sort_values(ascending=False).head(3).index)
        top3_names = [feat_names_map.get(k, k) for k in top3]
        st.markdown(f"""
        <div class="insight-box">
        📌 <strong>Top 3 predictors of conversion:</strong>
        {top3_names[0]}, {top3_names[1]}, {top3_names[2]}.<br>
        These variables should form the core of your lead-scoring model and sales targeting criteria.
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── High probability leads ────────────────────────────────────────────────
    st.markdown('<div class="section-header">Top High-Probability Leads (from Survey Data)</div>',
                unsafe_allow_html=True)

    try:
        from preprocessing import get_classification_features, encode
        rf_model   = models["rf_classifier"]
        scaler_clf = models["scaler_clf"]
        clf_feats  = models["clf_features"]

        valid_mask = df_enc["pod_conversion_binary"].notna()
        df_score   = df_enc[valid_mask].copy()
        X_score    = df_score[clf_feats].fillna(df_score[clf_feats].median())
        X_score_s  = scaler_clf.transform(X_score)
        probs      = rf_model.predict_proba(X_score_s)[:, 1]
        df_score["conversion_probability"] = probs
        df_score["lead_grade"] = pd.cut(probs, bins=[0, 0.45, 0.65, 0.80, 1.0],
                                         labels=["Cold", "Warm", "Hot", "Very Hot"])

        display_cols = ["respondent_id","true_segment","city_tier","income_bracket",
                        "cricket_role","conversion_probability","lead_grade"]
        avail_d = [c for c in display_cols if c in df_score.columns]
        top_leads = df_score.sort_values("conversion_probability", ascending=False)[avail_d].head(30)
        top_leads["conversion_probability"] = top_leads["conversion_probability"].round(3)
        st.dataframe(top_leads, use_container_width=True, hide_index=True)

        grade_counts = df_score["lead_grade"].value_counts()
        c1, c2, c3, c4 = st.columns(4)
        for col, grade, color in zip([c1, c2, c3, c4],
                                      ["Very Hot", "Hot", "Warm", "Cold"],
                                      [DANGER, ACCENT, PRIMARY, "#888"]):
            count = int(grade_counts.get(grade, 0))
            col.markdown(f"""
            <div class="metric-card">
              <div class="val" style="color:{color}">{count}</div>
              <div class="lbl">{grade} Leads</div>
            </div>""", unsafe_allow_html=True)

    except Exception as e:
        st.warning(f"Could not score leads: {e}")
