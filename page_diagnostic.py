import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

PRIMARY = "#1D9E75"; SECONDARY = "#7F77DD"; ACCENT = "#EF9F27"; DANGER = "#D85A30"

def show(df, df_enc, models):
    st.title("🔍 Diagnostic Analysis")
    st.markdown("Uncovering *why* customers are or aren't interested — correlations, cross-tabs and statistical tests.")
    st.markdown("---")

    # ── Correlation heatmap ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">Correlation Heatmap — Numeric Features</div>',
                unsafe_allow_html=True)

    num_cols = ["age_num","income_num","city_num","role_num","practice_num",
                "data_importance","pod_interest","spend_num","tech_num",
                "dist_num","nps_score","digital_num","addon_count","feat_count",
                "past_exp_count","barrier_count","frust_count","pod_conversion_binary"]
    avail = [c for c in num_cols if c in df_enc.columns]
    corr_df = df_enc[avail].apply(pd.to_numeric, errors="coerce").corr().round(2)

    labels_map = {
        "age_num":"Age","income_num":"Income","city_num":"City Tier",
        "role_num":"Cricket Role","practice_num":"Practice Days",
        "data_importance":"Data Importance","pod_interest":"Pod Interest",
        "spend_num":"Rec Spend","tech_num":"Tech Adoption",
        "dist_num":"Distance Toler.","nps_score":"NPS Score",
        "digital_num":"Digital Spend","addon_count":"Addon Count",
        "feat_count":"Feature Count","past_exp_count":"Past Exp",
        "barrier_count":"Barrier Count","frust_count":"Frustration Ct",
        "pod_conversion_binary":"Conversion",
    }
    tick_labels = [labels_map.get(c, c) for c in corr_df.columns]

    fig_hm = go.Figure(go.Heatmap(
        z=corr_df.values,
        x=tick_labels, y=tick_labels,
        colorscale=[[0, DANGER],[0.5,"#1a1a2e"],[1, PRIMARY]],
        zmid=0, zmin=-1, zmax=1,
        text=corr_df.values.round(2),
        texttemplate="%{text}",
        textfont=dict(size=8),
        colorbar=dict(tickfont=dict(color="#ccc")),
    ))
    fig_hm.update_layout(
        height=500, margin=dict(t=10, b=10, l=0, r=0),
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickfont=dict(color="#ccc", size=9), tickangle=-45),
        yaxis=dict(tickfont=dict(color="#ccc", size=9)),
        font=dict(color="#ccc"),
    )
    st.plotly_chart(fig_hm, use_container_width=True)

    # Top correlations with conversion
    if "pod_conversion_binary" in corr_df.columns:
        conv_corr = corr_df["pod_conversion_binary"].drop("pod_conversion_binary").abs().sort_values(ascending=False)
        st.markdown("**Top features correlated with conversion (pod_conversion_binary):**")
        top5 = conv_corr.head(5)
        cols = st.columns(5)
        for i, (feat, val) in enumerate(top5.items()):
            cols[i].metric(labels_map.get(feat, feat), f"{val:.3f}")

    st.markdown("---")

    # ── Income × WTP scatter ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">Income vs Willingness to Pay (Reasonable Price)</div>',
                unsafe_allow_html=True)

    PSM_R_MID = {"100-149":125,"150-199":175,"200-249":225,"250-299":275,"300-349":325}
    INC_ORD   = {"Below 20K":1,"20K-40K":2,"40K-75K":3,"75K-150K":4,"Above 150K":5}

    scatter_df = df.copy()
    scatter_df["psm_r_val"]  = scatter_df["psm_reasonable"].map(PSM_R_MID)
    scatter_df["income_val"] = scatter_df["income_bracket"].map(INC_ORD)
    scatter_df = scatter_df.dropna(subset=["psm_r_val","income_val"])

    inc_labels = {1:"<20K",2:"20-40K",3:"40-75K",4:"75-150K",5:">150K"}
    scatter_df["income_label"] = scatter_df["income_val"].map(inc_labels)

    col1, col2 = st.columns(2)
    with col1:
        avg_wtp = scatter_df.groupby("income_label")["psm_r_val"].mean().reset_index()
        order_inc = ["<20K","20-40K","40-75K","75-150K",">150K"]
        avg_wtp["order"] = avg_wtp["income_label"].map({v:i for i,v in enumerate(order_inc)})
        avg_wtp = avg_wtp.sort_values("order")
        fig_sc = go.Figure(go.Bar(
            x=avg_wtp["income_label"], y=avg_wtp["psm_r_val"],
            marker_color=PRIMARY, text=[f"₹{v:.0f}" for v in avg_wtp["psm_r_val"]],
            textposition="outside",
        ))
        fig_sc.update_layout(
            title="Avg reasonable price by income", height=300,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(color="#ccc"), yaxis=dict(showgrid=False, color="#aaa", title="₹"),
            font=dict(color="#ccc"), margin=dict(t=35, b=10, l=0, r=0),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    with col2:
        # City tier × conversion rate
        conv_by_city = df_enc.groupby("city_tier")["pod_conversion_binary"].mean().dropna() * 100
        conv_by_city = conv_by_city.sort_values(ascending=False)
        fig_city = go.Figure(go.Bar(
            x=conv_by_city.index, y=conv_by_city.values,
            marker_color=[PRIMARY if v == conv_by_city.max() else SECONDARY for v in conv_by_city.values],
            text=[f"{v:.1f}%" for v in conv_by_city.values], textposition="outside",
        ))
        fig_city.update_layout(
            title="Conversion rate by city tier", height=300,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(color="#ccc"), yaxis=dict(showgrid=False, color="#aaa", title="%"),
            font=dict(color="#ccc"), margin=dict(t=35, b=10, l=0, r=0),
        )
        st.plotly_chart(fig_city, use_container_width=True)

    # ── Chi-square: past behaviour vs conversion ──────────────────────────────
    st.markdown('<div class="section-header">Statistical Tests</div>', unsafe_allow_html=True)
    col3, col4 = st.columns(2)

    with col3:
        st.markdown("**Chi-Square: Past Tech-Leisure Experience vs Conversion**")
        if "past_boxcricket" in df_enc.columns and "pod_conversion_binary" in df_enc.columns:
            test_df = df_enc[["past_boxcricket","pod_conversion_binary"]].dropna()
            test_df["past_boxcricket"] = test_df["past_boxcricket"].fillna(0).astype(int)
            ct = pd.crosstab(test_df["past_boxcricket"], test_df["pod_conversion_binary"].astype(int))
            chi2, p, dof, _ = stats.chi2_contingency(ct)
            st.dataframe(ct, use_container_width=True)
            result_color = PRIMARY if p < 0.05 else DANGER
            st.markdown(f"""
            <div class="insight-box">
            χ² = <strong>{chi2:.2f}</strong> | p-value = <strong>{p:.4f}</strong> | dof = {dof}<br>
            {'✅ Statistically significant (p < 0.05) — past tech-leisure experience IS a strong predictor of conversion.' if p < 0.05 else '❌ Not significant at 0.05 level.'}
            </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown("**ANOVA: Tech Adoption Style vs Pod Interest Score**")
        if "tech_adoption" in df_enc.columns:
            groups = [df_enc[df_enc["tech_adoption"] == g]["pod_interest"].dropna().values
                      for g in df_enc["tech_adoption"].unique() if len(df_enc[df_enc["tech_adoption"] == g]) > 5]
            if len(groups) >= 2:
                f_stat, p_val = stats.f_oneway(*groups)
                avg_by_tech = df_enc.groupby("tech_adoption")["pod_interest"].mean().sort_values(ascending=False)
                fig_an = go.Figure(go.Bar(
                    x=avg_by_tech.index, y=avg_by_tech.values,
                    marker_color=SECONDARY, text=[f"{v:.2f}" for v in avg_by_tech.values],
                    textposition="outside",
                ))
                fig_an.update_layout(
                    height=220, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(color="#ccc", tickangle=-20),
                    yaxis=dict(showgrid=False, color="#aaa", title="Avg pod interest"),
                    font=dict(color="#ccc", size=10), margin=dict(t=10, b=60, l=0, r=0),
                )
                st.plotly_chart(fig_an, use_container_width=True)
                st.markdown(f"""
                <div class="insight-box">
                F = <strong>{f_stat:.2f}</strong> | p = <strong>{p_val:.4f}</strong><br>
                {'✅ Significant — tech adoption style significantly affects pod interest.' if p_val < 0.05 else '❌ Not significant.'}
                </div>""", unsafe_allow_html=True)

    # ── Frustration → switchability heatmap ──────────────────────────────────
    st.markdown('<div class="section-header">Competitor Frustration → Conversion Heatmap</div>',
                unsafe_allow_html=True)

    frust_cols = ["frust_nodata","frust_coachattention","frust_timing","frust_crowded",
                  "frust_distance","frust_cost","frust_equipment","frust_notracking"]
    frust_labels = ["No data/feedback","Coach inattentive","Bad timing","Too crowded",
                    "Too far","High cost","Old equipment","No progress tracking"]

    avail_f = [c for c in frust_cols if c in df_enc.columns]
    labels_avail = [frust_labels[frust_cols.index(c)] for c in avail_f]

    if avail_f and "pod_conversion_binary" in df_enc.columns:
        rows = []
        for col, lbl in zip(avail_f, labels_avail):
            sub = df_enc[[col, "pod_conversion_binary"]].dropna()
            sub[col] = sub[col].fillna(0)
            g0 = sub[sub[col] == 0]["pod_conversion_binary"].mean()
            g1 = sub[sub[col] == 1]["pod_conversion_binary"].mean()
            rows.append({"Frustration": lbl, "Has frustration": g1, "No frustration": g0,
                         "Lift": g1 - g0})
        frust_df = pd.DataFrame(rows).sort_values("Lift", ascending=False)

        fig_frust = go.Figure()
        fig_frust.add_trace(go.Bar(name="Has frustration", x=frust_df["Frustration"],
                                    y=(frust_df["Has frustration"]*100).round(1),
                                    marker_color=PRIMARY, text=[f"{v:.1f}%" for v in frust_df["Has frustration"]*100],
                                    textposition="outside"))
        fig_frust.add_trace(go.Bar(name="No frustration", x=frust_df["Frustration"],
                                    y=(frust_df["No frustration"]*100).round(1),
                                    marker_color="#444"))
        fig_frust.update_layout(
            barmode="group", height=340,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(color="#ccc", tickangle=-25),
            yaxis=dict(showgrid=False, color="#aaa", title="Conversion rate %"),
            legend=dict(font=dict(color="#ccc")),
            font=dict(color="#ccc"), margin=dict(t=10, b=80, l=0, r=0),
        )
        st.plotly_chart(fig_frust, use_container_width=True)
        st.markdown("""
        <div class="insight-box">
        📌 <strong>Founder action:</strong> Respondents frustrated by "No data/feedback" and
        "No progress tracking" show the highest conversion lift — these are your warmest leads
        and your core marketing message.
        </div>""", unsafe_allow_html=True)
