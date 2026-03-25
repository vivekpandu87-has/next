import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA

PRIMARY = "#1D9E75"; SECONDARY = "#7F77DD"; ACCENT = "#EF9F27"; DANGER = "#D85A30"
CLUSTER_COLORS = [PRIMARY, SECONDARY, ACCENT, DANGER, "#5DCAA5", "#F0997B", "#85B7EB"]

DISCOUNT_MAP = {
    "Rising Star":        ("Student / U-18 discount + free first session", "WhatsApp groups, school coaches"),
    "Elite Competitor":   ("Buy-5-get-1 + AI coaching bundle at 15% off", "Coach network, BCCI academies"),
    "Corporate Cricket Fan": ("Corporate group package + employer tie-up", "LinkedIn, HR managers"),
    "Recreational Player":("Weekend off-peak discount + referral offer",  "Instagram, Google Maps"),
    "Sceptic / Disengaged":("Free trial session — no commitment needed",  "YouTube content, re-targeting ads"),
}

def show(df, df_enc, models):
    st.title("👥 Clustering — Customer Personas")
    st.markdown("K-Means segmentation to discover natural customer groups for personalised offers.")
    st.markdown("---")

    results = models.get("all_results", {})
    if not results or "clustering" not in results:
        st.error("Clustering results not found.")
        return

    clust = results["clustering"]

    # ── Elbow + Silhouette ───────────────────────────────────────────────────
    st.markdown('<div class="section-header">Choosing Optimal K — Elbow & Silhouette</div>',
                unsafe_allow_html=True)

    k_vals = list(range(2, 9))
    col1, col2 = st.columns(2)

    with col1:
        fig_elbow = go.Figure(go.Scatter(
            x=k_vals, y=clust["inertias"], mode="lines+markers",
            line=dict(color=PRIMARY, width=2.5),
            marker=dict(color=ACCENT, size=8),
        ))
        fig_elbow.add_vline(x=clust["best_k"], line_color=DANGER, line_dash="dash",
                             annotation_text=f"Chosen k={clust['best_k']}",
                             annotation_font_color=DANGER)
        fig_elbow.update_layout(
            title="Elbow Curve (Inertia)", height=280,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Number of clusters (k)", color="#ccc"),
            yaxis=dict(title="Inertia", color="#aaa", showgrid=False),
            font=dict(color="#ccc"), margin=dict(t=35, b=40, l=0, r=0),
        )
        st.plotly_chart(fig_elbow, use_container_width=True)

    with col2:
        fig_sil = go.Figure(go.Scatter(
            x=k_vals, y=clust["silhouettes"], mode="lines+markers",
            line=dict(color=SECONDARY, width=2.5),
            marker=dict(color=ACCENT, size=8),
        ))
        best_sil = max(clust["silhouettes"])
        best_k_sil = k_vals[clust["silhouettes"].index(best_sil)]
        fig_sil.add_vline(x=best_k_sil, line_color=PRIMARY, line_dash="dash",
                           annotation_text=f"Best silhouette k={best_k_sil}",
                           annotation_font_color=PRIMARY)
        fig_sil.update_layout(
            title="Silhouette Score by k", height=280,
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Number of clusters (k)", color="#ccc"),
            yaxis=dict(title="Silhouette Score", color="#aaa", showgrid=False),
            font=dict(color="#ccc"), margin=dict(t=35, b=40, l=0, r=0),
        )
        st.plotly_chart(fig_sil, use_container_width=True)

    st.markdown("---")

    # ── PCA 2D scatter ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Cluster Visualisation — PCA 2D Projection</div>',
                unsafe_allow_html=True)

    persona_map = models.get("persona_map", {})
    if "cluster" in df_enc.columns:
        from preprocessing import get_cluster_features
        X_c = get_cluster_features(df_enc)
        X_cs = models["scaler_clust"].transform(X_c)
        pca  = PCA(n_components=2, random_state=42)
        pcs  = pca.fit_transform(X_cs)
        pca_df = pd.DataFrame({"PC1": pcs[:,0], "PC2": pcs[:,1],
                                "Cluster": df_enc["cluster"].astype(str),
                                "Persona": df_enc["cluster"].map(persona_map)})
        fig_pca = go.Figure()
        for c_id in sorted(pca_df["Cluster"].unique()):
            sub = pca_df[pca_df["Cluster"] == c_id]
            persona = persona_map.get(int(c_id), f"Cluster {c_id}")
            fig_pca.add_trace(go.Scatter(
                x=sub["PC1"], y=sub["PC2"], mode="markers",
                name=f"C{c_id}: {persona}",
                marker=dict(color=CLUSTER_COLORS[int(c_id) % len(CLUSTER_COLORS)],
                            size=5, opacity=0.65),
            ))
        fig_pca.update_layout(
            height=420, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title=f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)",
                       color="#ccc", showgrid=True, gridcolor="#333"),
            yaxis=dict(title=f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)",
                       color="#aaa", showgrid=True, gridcolor="#333"),
            legend=dict(font=dict(color="#ccc", size=10)),
            font=dict(color="#ccc"), margin=dict(t=10, b=40, l=0, r=0),
        )
        st.plotly_chart(fig_pca, use_container_width=True)

    st.markdown("---")

    # ── Persona cards ─────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Customer Personas & Recommended Strategies</div>',
                unsafe_allow_html=True)

    profiles = models.get("cluster_profiles", [])
    for i, prof in enumerate(profiles):
        c_id    = prof["cluster"]
        persona = persona_map.get(c_id, f"Cluster {c_id}")
        disc, channel = DISCOUNT_MAP.get(persona, ("Free trial", "Social media"))
        color   = CLUSTER_COLORS[c_id % len(CLUSTER_COLORS)]
        conv_pct = prof.get("conversion_rate", 0) * 100
        avg_spend = prof.get("avg_spend", 0)

        st.markdown(f"""
        <div style="background:#1a1a2e;border:1px solid {color}55;border-left:4px solid {color};
                    border-radius:12px;padding:1rem 1.2rem;margin-bottom:10px">
          <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
            <div>
              <span style="background:{color}22;color:{color};font-size:0.7rem;font-weight:600;
                           padding:2px 8px;border-radius:10px">CLUSTER {c_id}</span>
              <span style="font-size:1.05rem;font-weight:600;color:#fff;margin-left:10px">{persona}</span>
            </div>
            <span style="font-size:0.8rem;color:#aaa">{prof['size']:,} respondents</span>
          </div>
          <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:10px">
            <div style="text-align:center">
              <div style="font-size:1.2rem;font-weight:700;color:{color}">{conv_pct:.0f}%</div>
              <div style="font-size:0.72rem;color:#aaa">Conversion rate</div>
            </div>
            <div style="text-align:center">
              <div style="font-size:1.2rem;font-weight:700;color:{color}">₹{avg_spend:.0f}</div>
              <div style="font-size:0.72rem;color:#aaa">Avg monthly spend</div>
            </div>
            <div style="text-align:center">
              <div style="font-size:1.2rem;font-weight:700;color:{color}">{prof['avg_pod_interest']:.1f}/5</div>
              <div style="font-size:0.72rem;color:#aaa">Pod interest</div>
            </div>
            <div style="text-align:center">
              <div style="font-size:1.2rem;font-weight:700;color:{color}">{prof['avg_income']:.1f}/5</div>
              <div style="font-size:0.72rem;color:#aaa">Income score</div>
            </div>
          </div>
          <div style="font-size:0.8rem;color:#ccc">
            <span style="color:{color};font-weight:600">Best offer:</span> {disc}<br>
            <span style="color:{color};font-weight:600">Reach via:</span> {channel}
          </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Cluster profile radar ─────────────────────────────────────────────────
    st.markdown('<div class="section-header">Cluster Radar — Feature Profiles</div>',
                unsafe_allow_html=True)

    radar_feats  = ["age_num","income_num","role_num","practice_num",
                    "data_importance","pod_interest","spend_num","tech_num"]
    radar_labels = ["Age","Income","Cricket Role","Practice Days",
                    "Data Importance","Pod Interest","Rec Spend","Tech Adoption"]

    if "cluster" in df_enc.columns:
        avail_r = [c for c in radar_feats if c in df_enc.columns]
        avail_l = [radar_labels[radar_feats.index(c)] for c in avail_r]
        cluster_means = df_enc.groupby("cluster")[avail_r].mean()
        # normalise 0-1 per feature
        norm = (cluster_means - cluster_means.min()) / (cluster_means.max() - cluster_means.min() + 1e-9)

        fig_radar = go.Figure()
        for c_id, row in norm.iterrows():
            persona = persona_map.get(int(c_id), f"Cluster {c_id}")
            color   = CLUSTER_COLORS[int(c_id) % len(CLUSTER_COLORS)]
            vals    = row.tolist() + [row.tolist()[0]]
            lbls    = avail_l + [avail_l[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals, theta=lbls, fill="toself",
                name=f"C{c_id}: {persona}",
                line=dict(color=color), fillcolor=color,
                opacity=0.25,
            ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(visible=True, range=[0, 1], color="#666"),
                angularaxis=dict(color="#ccc"),
            ),
            paper_bgcolor="rgba(0,0,0,0)",
            height=440, legend=dict(font=dict(color="#ccc", size=10)),
            font=dict(color="#ccc"), margin=dict(t=20, b=20, l=0, r=0),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Cluster vs true segment validation ───────────────────────────────────
    st.markdown('<div class="section-header">Cluster Purity vs True Segment (Validation)</div>',
                unsafe_allow_html=True)

    if "cluster" in df_enc.columns and "true_segment" in df_enc.columns:
        ct = pd.crosstab(df_enc["cluster"].map(lambda x: persona_map.get(x, str(x))),
                         df_enc["true_segment"],
                         normalize="index").round(3) * 100
        st.dataframe(ct.style.background_gradient(cmap="Greens", axis=1),
                     use_container_width=True)
        st.markdown("""
        <div class="insight-box">
        Each row shows what % of that cluster came from each true segment.
        High diagonal values = good cluster purity. This validates that K-Means
        is recovering the underlying customer types correctly.
        </div>""", unsafe_allow_html=True)
