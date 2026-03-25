import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

PRIMARY = "#1D9E75"; SECONDARY = "#7F77DD"; ACCENT = "#EF9F27"; DANGER = "#D85A30"

LABEL_MAP = {
    "feat_ai":"AI Analysis","feat_bowlingmachine":"Bowling Machine","feat_batspeed":"Bat Speed",
    "feat_footwork":"Footwork","feat_videoreplay":"Video Replay","feat_leaderboard":"Leaderboard",
    "feat_progressreport":"Progress Report","feat_appbooking":"App Booking",
    "addon_smartbat":"Smart Bat","addon_wearables":"Wearables Kit","addon_aicoaching":"AI Coaching Sub",
    "addon_highlights":"Video Highlights","addon_fitness":"Fitness Program","addon_merch":"Merchandise",
    "use_academy":"Uses Academy","use_boxcricket":"Box Cricket","use_bowlingmachine":"Bowling Machine Net",
    "use_gym":"Gym","use_mobilegame":"Mobile Cricket Game","use_videoanalysis":"Video Analysis",
    "act_gym":"Gym","act_yoga":"Yoga","act_othersport":"Other Sport","act_swimming":"Swimming",
    "act_videogaming":"Video Gaming","act_running":"Running",
    "stream_hotstar":"Hotstar","stream_netflix":"Netflix","stream_jiocinema":"JioCinema",
    "stream_prime":"Amazon Prime","stream_youtube":"YouTube","stream_sonyliv":"Sony LIV",
    "past_boxcricket":"Paid Box Cricket","past_trampoline":"Trampoline Park","past_vr":"VR Gaming",
    "past_bowling":"Bowling Alley","past_gokarting":"Go-Karting","past_fitclass":"Fitness Class",
    "past_academy":"Paid Academy","past_golf":"Golf Range",
    "frust_nodata":"Frustrated: No Data","frust_coachattention":"Frustrated: Coach Inattentive",
    "frust_timing":"Frustrated: Timing","frust_notracking":"Frustrated: No Tracking",
    "bar_price":"Barrier: Price","bar_aidistrust":"Barrier: AI Distrust",
    "bar_location":"Barrier: Location","bar_humancoach":"Barrier: Wants Human Coach",
    "disc_freetrial":"Prefers Free Trial","disc_referral":"Prefers Referral Disc",
    "disc_student":"Prefers Student Disc","disc_family":"Prefers Family Bundle",
    "hh_self":"Self User","hh_child":"Child User","hh_spouse":"Spouse User",
    "brand_mrf":"Brand MRF","brand_sg":"Brand SG","brand_decathlon":"Brand Decathlon",
}

def pretty(items_str):
    parts = [s.strip() for s in items_str.split(",")]
    return " + ".join([LABEL_MAP.get(p, p) for p in parts])

def show(df, df_enc, models):
    st.title("🔗 Association Rule Mining")
    st.markdown("Discovering what products, features and behaviours go together — using **Apriori** algorithm.")
    st.markdown("---")

    rules = models.get("assoc_rules")
    if rules is None or (hasattr(rules, "__len__") and len(rules) == 0):
        st.warning("Association rules not found in models. Re-run model_trainer.py.")
        return

    if not isinstance(rules, pd.DataFrame):
        rules = pd.DataFrame(rules)

    if rules.empty:
        st.warning("No rules found with current thresholds.")
        return

    # ── Filter controls ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Filter Rules</div>', unsafe_allow_html=True)
    col_f1, col_f2, col_f3 = st.columns(3)
    min_sup  = col_f1.slider("Min Support",    0.01, 0.30, 0.05, 0.01)
    min_conf = col_f2.slider("Min Confidence", 0.30, 0.95, 0.50, 0.05)
    min_lift = col_f3.slider("Min Lift",       1.0,  5.0,  1.2,  0.1)

    filtered = rules[
        (rules["support"]    >= min_sup) &
        (rules["confidence"] >= min_conf) &
        (rules["lift"]       >= min_lift)
    ].copy()
    filtered = filtered.sort_values("lift", ascending=False).reset_index(drop=True)

    st.markdown(f"**{len(filtered)} rules** match current filters.")
    st.markdown("---")

    # ── KPI summary ───────────────────────────────────────────────────────────
    if len(filtered) > 0:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Rules",     len(filtered))
        c2.metric("Max Lift",        f"{filtered['lift'].max():.3f}")
        c3.metric("Max Confidence",  f"{filtered['confidence'].max():.3f}")
        c4.metric("Max Support",     f"{filtered['support'].max():.3f}")

    # ── Top rules table ───────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Top Rules — Ranked by Lift</div>',
                unsafe_allow_html=True)

    if len(filtered) > 0:
        display_rules = filtered.head(30).copy()
        display_rules["antecedents"] = display_rules["antecedents"].apply(
            lambda x: pretty(x) if isinstance(x, str) else str(x))
        display_rules["consequents"] = display_rules["consequents"].apply(
            lambda x: pretty(x) if isinstance(x, str) else str(x))
        display_rules = display_rules[["antecedents","consequents","support","confidence","lift"]].copy()
        display_rules["support"]    = display_rules["support"].round(4)
        display_rules["confidence"] = display_rules["confidence"].round(4)
        display_rules["lift"]       = display_rules["lift"].round(4)
        display_rules.columns       = ["IF (Antecedent)","THEN (Consequent)","Support","Confidence","Lift"]
        st.dataframe(display_rules, use_container_width=True, hide_index=True)
    else:
        st.info("No rules match current filter. Try lowering thresholds.")

    st.markdown("---")

    # ── Scatter: Support vs Confidence (bubble = lift) ────────────────────────
    st.markdown('<div class="section-header">Support vs Confidence — Bubble Size = Lift</div>',
                unsafe_allow_html=True)

    if len(filtered) > 0:
        plot_df = filtered.head(60).copy()
        plot_df["ant_pretty"] = plot_df["antecedents"].apply(
            lambda x: pretty(x)[:40] + "..." if len(pretty(str(x))) > 40 else pretty(str(x)))
        plot_df["con_pretty"] = plot_df["consequents"].apply(
            lambda x: pretty(x)[:40] + "..." if len(pretty(str(x))) > 40 else pretty(str(x)))

        fig_sc = go.Figure(go.Scatter(
            x=plot_df["support"],
            y=plot_df["confidence"],
            mode="markers",
            marker=dict(
                size=np.clip(plot_df["lift"] * 6, 6, 30),
                color=plot_df["lift"],
                colorscale=[[0,"#333"],[0.5, SECONDARY],[1, PRIMARY]],
                showscale=True,
                colorbar=dict(title="Lift", tickfont=dict(color="#ccc")),
                opacity=0.8,
            ),
            text=[f"IF: {a}<br>THEN: {c}<br>Support={s:.3f} | Conf={cf:.3f} | Lift={l:.3f}"
                  for a, c, s, cf, l in zip(
                      plot_df["ant_pretty"], plot_df["con_pretty"],
                      plot_df["support"], plot_df["confidence"], plot_df["lift"])],
            hoverinfo="text",
        ))
        fig_sc.update_layout(
            height=400, plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Support", color="#ccc", showgrid=True, gridcolor="#333"),
            yaxis=dict(title="Confidence", color="#aaa", showgrid=True, gridcolor="#333"),
            font=dict(color="#ccc"), margin=dict(t=10, b=40, l=0, r=0),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("---")

    # ── Business-specific rule categories ────────────────────────────────────
    st.markdown('<div class="section-header">Business Rule Categories</div>',
                unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🏏 Feature → Add-on Bundles",
                                  "😤 Frustration → Conversion Signals",
                                  "🎯 Discount → Segment Patterns"])

    with tab1:
        feat_rules = filtered[
            filtered["antecedents"].str.contains("feat_", na=False) &
            filtered["consequents"].str.contains("addon_", na=False)
        ].head(15).copy()
        if len(feat_rules) > 0:
            feat_rules["antecedents"] = feat_rules["antecedents"].apply(lambda x: pretty(str(x)))
            feat_rules["consequents"] = feat_rules["consequents"].apply(lambda x: pretty(str(x)))
            feat_rules = feat_rules[["antecedents","consequents","support","confidence","lift"]].round(4)
            feat_rules.columns = ["Feature Interest","Bundle Add-on","Support","Confidence","Lift"]
            st.dataframe(feat_rules, use_container_width=True, hide_index=True)
            st.markdown("""
            <div class="insight-box">
            📌 <strong>Bundle strategy:</strong> Players interested in AI Analysis and Bowling Machine
            features should be shown the Smart Bat + AI Coaching subscription bundle at checkout.
            </div>""", unsafe_allow_html=True)
        else:
            st.info("Increase min support or lower confidence to see feature→addon rules.")

    with tab2:
        frust_rules = filtered[
            filtered["antecedents"].str.contains("frust_", na=False)
        ].head(15).copy()
        if len(frust_rules) > 0:
            frust_rules["antecedents"] = frust_rules["antecedents"].apply(lambda x: pretty(str(x)))
            frust_rules["consequents"] = frust_rules["consequents"].apply(lambda x: pretty(str(x)))
            frust_rules = frust_rules[["antecedents","consequents","support","confidence","lift"]].round(4)
            frust_rules.columns = ["Frustration","Associated Behaviour/Interest","Support","Confidence","Lift"]
            st.dataframe(frust_rules, use_container_width=True, hide_index=True)
        else:
            st.info("No frustration rules found at current thresholds.")

    with tab3:
        disc_rules = filtered[
            filtered["antecedents"].str.contains("disc_", na=False) |
            filtered["consequents"].str.contains("disc_", na=False)
        ].head(15).copy()
        if len(disc_rules) > 0:
            disc_rules["antecedents"] = disc_rules["antecedents"].apply(lambda x: pretty(str(x)))
            disc_rules["consequents"] = disc_rules["consequents"].apply(lambda x: pretty(str(x)))
            disc_rules = disc_rules[["antecedents","consequents","support","confidence","lift"]].round(4)
            disc_rules.columns = ["Pattern A","Pattern B","Support","Confidence","Lift"]
            st.dataframe(disc_rules, use_container_width=True, hide_index=True)
        else:
            st.info("No discount pattern rules found at current thresholds.")

    st.markdown("---")

    # ── Top 10 rules bar chart by lift ────────────────────────────────────────
    st.markdown('<div class="section-header">Top 10 Rules by Lift</div>', unsafe_allow_html=True)

    if len(filtered) >= 3:
        top10 = filtered.head(10).copy()
        top10["rule_label"] = [
            f"{pretty(str(a))[:30]}… → {pretty(str(c))[:20]}…"
            for a, c in zip(top10["antecedents"], top10["consequents"])
        ]
        top10_sorted = top10.sort_values("lift", ascending=True)

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            y=top10_sorted["rule_label"], x=top10_sorted["lift"],
            orientation="h", name="Lift",
            marker_color=PRIMARY,
            text=[f"{v:.3f}" for v in top10_sorted["lift"]], textposition="outside",
        ))
        fig_bar.add_trace(go.Bar(
            y=top10_sorted["rule_label"], x=top10_sorted["confidence"],
            orientation="h", name="Confidence",
            marker_color=SECONDARY,
            text=[f"{v:.3f}" for v in top10_sorted["confidence"]], textposition="outside",
            visible="legendonly",
        ))
        fig_bar.update_layout(
            height=400, barmode="overlay",
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, color="#aaa", title="Score"),
            yaxis=dict(color="#ccc"),
            legend=dict(font=dict(color="#ccc")),
            font=dict(color="#ccc"), margin=dict(t=10, b=20, l=0, r=80),
        )
        st.plotly_chart(fig_bar, use_container_width=True)
