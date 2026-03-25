import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import io

PRIMARY = "#1D9E75"; SECONDARY = "#7F77DD"; ACCENT = "#EF9F27"; DANGER = "#D85A30"

def _action(prob, persona):
    if prob >= 0.75:
        return f"🔥 HOT LEAD — Offer free trial session immediately"
    elif prob >= 0.55:
        return f"⭐ WARM — Send ₹100 first-session discount voucher"
    elif prob >= 0.40:
        return f"🔄 NURTURE — Share AI demo video + feature explainer"
    else:
        return f"📧 COLD — Add to monthly newsletter only"

def _channel(persona):
    ch = {
        "Rising Star":           "WhatsApp / school coach network",
        "Elite Competitor":      "Cricket academy / coach referral",
        "Corporate Cricket Fan": "LinkedIn / employer HR tie-up",
        "Recreational Player":   "Instagram / Google Maps SEO",
        "Sceptic / Disengaged":  "YouTube retargeting / free trial push",
    }
    return ch.get(str(persona), "Social media / digital ads")

def score_new_data(new_df, models):
    from preprocessing import encode, CLASSIFICATION_FEATURES, REGRESSION_FEATURES, CLUSTERING_FEATURES

    # Fill missing columns with 0 / mode placeholders
    defaults = {
        "age_group":"19-25","gender":"Male","city_tier":"Metro",
        "occupation":"Salaried private","income_bracket":"40K-75K",
        "education":"Bachelors","cricket_role":"Regular",
        "practice_days":"1-2","data_importance":3,"fantasy_cricket":"Occasional",
        "pod_interest":3,"monthly_rec_spend":"501-1000",
        "psm_too_cheap":"100-149","psm_reasonable":"200-249",
        "psm_expensive":"400-499","psm_too_expensive":"600-799",
        "membership_wtp":"500-999","digital_spend":"201-500",
        "food_delivery_freq":"1-2/week","influence_source":"Self",
        "tech_adoption":"Early majority","distance_tolerance":"Up to 5km",
        "preferred_timeslot":"Evening","nps_score":7,
    }
    for col, val in defaults.items():
        if col not in new_df.columns:
            new_df[col] = val

    # Binary multi-select columns default to 0
    from preprocessing import MULTI_SELECT_COLS
    for col in MULTI_SELECT_COLS:
        if col not in new_df.columns:
            new_df[col] = 0

    df_enc = encode(new_df)

    # Clustering
    clust_feats = models.get("cluster_features", CLUSTERING_FEATURES)
    avail_c = [f for f in clust_feats if f in df_enc.columns]
    X_c = df_enc[avail_c].fillna(df_enc[avail_c].median())
    X_cs = models["scaler_clust"].transform(X_c)
    clusters = models["kmeans"].predict(X_cs)
    persona_map = models.get("persona_map", {})
    personas = [persona_map.get(c, f"Cluster {c}") for c in clusters]

    # Classification
    clf_feats = models.get("clf_features", CLASSIFICATION_FEATURES)
    avail_f = [f for f in clf_feats if f in df_enc.columns]
    X_clf = df_enc[avail_f].fillna(df_enc[avail_f].median())
    X_clf_s = models["scaler_clf"].transform(X_clf)
    probs = models["rf_classifier"].predict_proba(X_clf_s)[:, 1]

    # Regression
    reg_feats = models.get("reg_features", REGRESSION_FEATURES)
    avail_r = [f for f in reg_feats if f in df_enc.columns]
    X_reg = df_enc[avail_r].fillna(df_enc[avail_r].median())
    X_reg_s = models["scaler_reg"].transform(X_reg)
    spend_pred = models["ridge_regressor"].predict(X_reg_s)
    spend_pred = np.clip(spend_pred, 0, 5000)

    out = new_df.copy()
    out["conversion_probability"] = np.round(probs, 3)
    out["lead_grade"] = pd.cut(probs, bins=[0, 0.40, 0.55, 0.75, 1.01],
                                labels=["Cold", "Warm", "Hot", "Very Hot"])
    out["predicted_cluster"]   = clusters
    out["persona"]             = personas
    out["predicted_spend_pm"]  = np.round(spend_pred, 0)
    out["recommended_action"]  = [_action(p, per) for p, per in zip(probs, personas)]
    out["recommended_channel"] = [_channel(per) for per in personas]

    return out

def show(df, df_enc, models):
    st.title("🚀 New Customer Predictor")
    st.markdown("Upload new survey responses → instantly score each lead with **conversion probability**, "
                "**persona**, **predicted spend**, and **recommended action**.")
    st.markdown("---")

    # ── Sample download ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Step 1 — Download Sample Template</div>',
                unsafe_allow_html=True)
    st.markdown("Download a sample CSV showing the expected column format:")

    sample_cols = ["age_group","gender","city_tier","occupation","income_bracket",
                   "education","cricket_role","practice_days","data_importance",
                   "fantasy_cricket","pod_interest","monthly_rec_spend",
                   "psm_too_cheap","psm_reasonable","psm_expensive","psm_too_expensive",
                   "membership_wtp","digital_spend","food_delivery_freq",
                   "influence_source","tech_adoption","distance_tolerance",
                   "preferred_timeslot","nps_score"]
    sample_rows = [
        ["19-25","Male","Metro","Salaried private","40K-75K","Bachelors",
         "Regular","3-4",4,"Active",4,"1001-2500",
         "100-149","200-249","400-499","600-799",
         "1000-1999","201-500","3-4/week",
         "Friends/Teammates","Early majority","Up to 5km","Evening",8],
        ["15-18","Male","Tier 1","School student","Below 20K","Up to 10th",
         "Competitive","5-6",5,"Occasional",5,"501-1000",
         "50-99","150-199","300-399","500-599",
         "500-999","1-200","1-2/week",
         "Parents/Family","Early adopter","Up to 3km","Early morning",9],
        ["26-35","Female","Metro","Professional","75K-150K","Masters+",
         "Fan only","0",2,"Not interested",2,"1001-2500",
         "150-199","250-299","500-599","800+",
         "Would not subscribe","501-1000","1-2/week",
         "Social media","Late majority","Any distance","Evening",6],
    ]
    sample_df = pd.DataFrame(sample_rows, columns=sample_cols)
    csv_buffer = io.StringIO()
    sample_df.to_csv(csv_buffer, index=False)
    st.download_button(
        label="⬇️ Download sample template CSV",
        data=csv_buffer.getvalue(),
        file_name="new_customers_template.csv",
        mime="text/csv",
    )

    st.markdown("---")

    # ── Upload ────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Step 2 — Upload New Customer Data</div>',
                unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "Upload CSV with new respondents (must include at least age_group, income_bracket, cricket_role)",
        type=["csv"],
    )

    if uploaded is not None:
        try:
            new_df = pd.read_csv(uploaded)
            st.success(f"✅ Loaded {len(new_df):,} rows × {new_df.shape[1]} columns")
            st.dataframe(new_df.head(5), use_container_width=True)

            with st.spinner("Scoring all leads..."):
                scored = score_new_data(new_df.copy(), models)

            st.markdown("---")
            st.markdown('<div class="section-header">Scoring Results</div>',
                        unsafe_allow_html=True)

            # Summary KPIs
            grade_counts = scored["lead_grade"].value_counts()
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Scored",    len(scored))
            c2.metric("🔥 Very Hot",     int(grade_counts.get("Very Hot", 0)))
            c3.metric("⭐ Hot",          int(grade_counts.get("Hot", 0)))
            c4.metric("🔄 Warm",         int(grade_counts.get("Warm", 0)))
            c5.metric("📧 Cold",         int(grade_counts.get("Cold", 0)))

            # Lead grade donut
            col_a, col_b = st.columns(2)
            with col_a:
                fig_g = go.Figure(go.Pie(
                    labels=grade_counts.index, values=grade_counts.values, hole=0.45,
                    marker_colors=[DANGER, ACCENT, PRIMARY, "#555"],
                ))
                fig_g.update_layout(
                    title="Lead Grade Distribution", height=300,
                    paper_bgcolor="rgba(0,0,0,0)",
                    legend=dict(font=dict(color="#ccc")),
                    font=dict(color="#ccc"), margin=dict(t=35, b=10, l=0, r=0),
                )
                st.plotly_chart(fig_g, use_container_width=True)

            with col_b:
                persona_counts = scored["persona"].value_counts()
                fig_p = go.Figure(go.Bar(
                    x=persona_counts.values, y=persona_counts.index, orientation="h",
                    marker_color=SECONDARY,
                    text=persona_counts.values, textposition="outside",
                ))
                fig_p.update_layout(
                    title="Personas Detected", height=300,
                    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(showgrid=False, color="#aaa"),
                    yaxis=dict(color="#ccc"),
                    font=dict(color="#ccc"), margin=dict(t=35, b=10, l=0, r=60),
                )
                st.plotly_chart(fig_p, use_container_width=True)

            # Full scored table
            st.markdown("**Full scored results:**")
            display_scored = scored[[c for c in scored.columns
                                     if c in new_df.columns or
                                     c in ["conversion_probability","lead_grade",
                                           "persona","predicted_spend_pm",
                                           "recommended_action","recommended_channel"]]]
            st.dataframe(display_scored, use_container_width=True, hide_index=True)

            # Download
            out_buffer = io.StringIO()
            display_scored.to_csv(out_buffer, index=False)
            st.download_button(
                label="⬇️ Download scored leads CSV",
                data=out_buffer.getvalue(),
                file_name="scored_leads.csv",
                mime="text/csv",
            )

            # Avg predicted spend by lead grade
            st.markdown('<div class="section-header">Predicted Monthly Spend by Lead Grade</div>',
                        unsafe_allow_html=True)
            spend_by_grade = scored.groupby("lead_grade")["predicted_spend_pm"].mean().reset_index()
            fig_sg = go.Figure(go.Bar(
                x=spend_by_grade["lead_grade"].astype(str),
                y=spend_by_grade["predicted_spend_pm"],
                marker_color=[DANGER, ACCENT, PRIMARY, "#555"],
                text=[f"₹{v:.0f}" for v in spend_by_grade["predicted_spend_pm"]],
                textposition="outside",
            ))
            fig_sg.update_layout(
                height=300, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(color="#ccc"), yaxis=dict(showgrid=False, color="#aaa", title="Avg ₹/month"),
                font=dict(color="#ccc"), margin=dict(t=10, b=20, l=0, r=0),
            )
            st.plotly_chart(fig_sg, use_container_width=True)

        except Exception as e:
            st.error(f"Error processing file: {e}")
            st.exception(e)

    else:
        st.info("👆 Upload a CSV file above to start scoring new customers.")
        st.markdown("""
        <div class="insight-box">
        📌 <strong>How this works:</strong><br>
        1. Your CSV is encoded using the same pipeline as training data<br>
        2. Missing columns are filled with training-set defaults automatically<br>
        3. Random Forest scores conversion probability (0.0 – 1.0)<br>
        4. K-Means assigns each person to a persona cluster<br>
        5. Ridge Regression predicts their monthly spend<br>
        6. Each row gets a recommended marketing action and channel
        </div>""", unsafe_allow_html=True)
