import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

PRIMARY = "#1D9E75"; SECONDARY = "#7F77DD"; ACCENT = "#EF9F27"; DANGER = "#D85A30"

PSM_TC_MID = {"Below 50":50,"50-99":75,"100-149":125,"150-199":175,"200-249":225}
PSM_R_MID  = {"100-149":125,"150-199":175,"200-249":225,"250-299":275,"300-349":325}
PSM_E_MID  = {"200-299":250,"300-399":350,"400-499":450,"500-599":550,"600+":650}
PSM_TE_MID = {"300-399":350,"400-499":450,"500-599":550,"600-799":700,"800+":900}

def show(df, df_enc, models):
    st.title("📊 Descriptive Analysis")
    st.markdown("Understanding the customer landscape — demographics, behaviours, and pricing signals.")
    st.markdown("---")

    # ── Demographic overview ──────────────────────────────────────────────────
    st.markdown('<div class="section-header">Demographics</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        age_c = df["age_group"].value_counts()
        order = ["Under 15","15-18","19-25","26-35","36-50","50+"]
        age_c = age_c.reindex([o for o in order if o in age_c.index])
        fig = go.Figure(go.Bar(x=age_c.index, y=age_c.values,
                               marker_color=PRIMARY, text=age_c.values, textposition="outside"))
        fig.update_layout(title="Age distribution", height=280,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(color="#ccc"), yaxis=dict(showgrid=False, color="#aaa"),
                          font=dict(color="#ccc"), margin=dict(t=35,b=10,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        gen_c = df["gender"].value_counts()
        fig = go.Figure(go.Pie(labels=gen_c.index, values=gen_c.values,
                               hole=0.4, marker_colors=[SECONDARY, PRIMARY, ACCENT]))
        fig.update_layout(title="Gender split", height=280,
                          paper_bgcolor="rgba(0,0,0,0)",
                          legend=dict(font=dict(color="#ccc",size=11)),
                          font=dict(color="#ccc"), margin=dict(t=35,b=10,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)

    with c3:
        inc_c = df["income_bracket"].value_counts()
        order_i = ["Below 20K","20K-40K","40K-75K","75K-150K","Above 150K"]
        inc_c = inc_c.reindex([o for o in order_i if o in inc_c.index])
        fig = go.Figure(go.Bar(x=inc_c.values, y=inc_c.index, orientation="h",
                               marker_color=ACCENT, text=inc_c.values, textposition="outside"))
        fig.update_layout(title="Income bracket", height=280,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(showgrid=False, color="#aaa"),
                          yaxis=dict(color="#ccc"),
                          font=dict(color="#ccc"), margin=dict(t=35,b=10,l=0,r=40))
        st.plotly_chart(fig, use_container_width=True)

    c4, c5, c6 = st.columns(3)
    with c4:
        role_c = df["cricket_role"].value_counts()
        fig = go.Figure(go.Pie(labels=role_c.index, values=role_c.values,
                               hole=0.4,
                               marker_colors=[PRIMARY,SECONDARY,ACCENT,DANGER,"#888","#5DCAA5"]))
        fig.update_layout(title="Cricket role", height=280,
                          paper_bgcolor="rgba(0,0,0,0)",
                          legend=dict(font=dict(color="#ccc",size=10)),
                          font=dict(color="#ccc"), margin=dict(t=35,b=10,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)

    with c5:
        occ_c = df["occupation"].value_counts().head(8)
        fig = go.Figure(go.Bar(x=occ_c.values, y=occ_c.index, orientation="h",
                               marker_color=SECONDARY, text=occ_c.values, textposition="outside"))
        fig.update_layout(title="Occupation", height=280,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(showgrid=False, color="#aaa"),
                          yaxis=dict(color="#ccc"),
                          font=dict(color="#ccc"), margin=dict(t=35,b=10,l=0,r=40))
        st.plotly_chart(fig, use_container_width=True)

    with c6:
        city_c = df["city_tier"].value_counts()
        fig = go.Figure(go.Bar(x=city_c.index, y=city_c.values,
                               marker_color=DANGER, text=city_c.values, textposition="outside"))
        fig.update_layout(title="City tier", height=280,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(color="#ccc"), yaxis=dict(showgrid=False, color="#aaa"),
                          font=dict(color="#ccc"), margin=dict(t=35,b=10,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)

    # ── Cricket behaviour ─────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Cricket Behaviour</div>', unsafe_allow_html=True)
    c7, c8 = st.columns(2)

    with c7:
        pd_c = df["practice_days"].value_counts()
        order_p = ["0","1-2","3-4","5-6","Daily"]
        pd_c = pd_c.reindex([o for o in order_p if o in pd_c.index])
        fig = go.Figure(go.Bar(x=pd_c.index, y=pd_c.values,
                               marker_color=[PRIMARY if v in ["3-4","5-6","Daily"] else "#888"
                                             for v in pd_c.index],
                               text=pd_c.values, textposition="outside"))
        fig.update_layout(title="Practice days per week", height=260,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(color="#ccc"), yaxis=dict(showgrid=False,color="#aaa"),
                          font=dict(color="#ccc"), margin=dict(t=35,b=10,l=0,r=0))
        st.plotly_chart(fig, use_container_width=True)

    with c8:
        feat_cols = ["feat_ai","feat_bowlingmachine","feat_batspeed","feat_footwork",
                     "feat_videoreplay","feat_leaderboard","feat_progressreport","feat_appbooking"]
        feat_labels = ["AI Analysis","Bowling Machine","Bat Speed","Footwork",
                       "Video Replay","Leaderboard","Progress Report","App Booking"]
        avail = [c for c in feat_cols if c in df.columns]
        labels_avail = [feat_labels[feat_cols.index(c)] for c in avail]
        feat_pcts = df[avail].fillna(0).mean() * 100
        fig = go.Figure(go.Bar(
            x=feat_pcts.values, y=labels_avail, orientation="h",
            marker_color=PRIMARY, text=[f"{v:.1f}%" for v in feat_pcts.values],
            textposition="outside"))
        fig.update_layout(title="Feature interest (% respondents)", height=280,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(range=[0,100], showgrid=False, color="#aaa"),
                          yaxis=dict(color="#ccc"),
                          font=dict(color="#ccc"), margin=dict(t=35,b=10,l=0,r=60))
        st.plotly_chart(fig, use_container_width=True)

    # ── Barriers ranked ────────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Barriers to Adoption (Q26)</div>', unsafe_allow_html=True)
    bar_cols   = ["bar_price","bar_location","bar_humancoach","bar_aidistrust",
                  "bar_time","bar_notserious","bar_academy","bar_safety","bar_social"]
    bar_labels = ["Too expensive","No pod nearby","Prefer human coach","AI distrust",
                  "No time","Not serious","Already in academy","Safety concern","Want friends along"]
    avail_b = [c for c in bar_cols if c in df.columns]
    labels_b = [bar_labels[bar_cols.index(c)] for c in avail_b]
    bar_pcts = df[avail_b].fillna(0).mean() * 100
    bar_series = pd.Series(bar_pcts.values, index=labels_b).sort_values(ascending=True)

    fig_bar = go.Figure(go.Bar(
        x=bar_series.values, y=bar_series.index, orientation="h",
        marker_color=[DANGER if v > 30 else ACCENT if v > 20 else "#888"
                      for v in bar_series.values],
        text=[f"{v:.1f}%" for v in bar_series.values], textposition="outside"))
    fig_bar.update_layout(height=320,
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                          xaxis=dict(range=[0,70], showgrid=False, color="#aaa"),
                          yaxis=dict(color="#ccc"),
                          font=dict(color="#ccc"), margin=dict(t=10,b=10,l=0,r=60))
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Van Westendorp PSM ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Van Westendorp Price Sensitivity Meter — Optimal Launch Price</div>',
                unsafe_allow_html=True)

    df_psm = df.copy()
    df_psm["tc_mid"] = df_psm["psm_too_cheap"].map(PSM_TC_MID)
    df_psm["r_mid"]  = df_psm["psm_reasonable"].map(PSM_R_MID)
    df_psm["e_mid"]  = df_psm["psm_expensive"].map(PSM_E_MID)
    df_psm["te_mid"] = df_psm["psm_too_expensive"].map(PSM_TE_MID)
    df_psm = df_psm.dropna(subset=["tc_mid","r_mid","e_mid","te_mid"])

    prices = np.arange(50, 900, 10)
    n_valid = len(df_psm)

    pct_tc  = [(df_psm["tc_mid"] >= p).sum() / n_valid * 100 for p in prices]
    pct_r   = [(df_psm["r_mid"]  <= p).sum() / n_valid * 100 for p in prices]
    pct_e   = [(df_psm["e_mid"]  <= p).sum() / n_valid * 100 for p in prices]
    pct_te  = [(df_psm["te_mid"] <= p).sum() / n_valid * 100 for p in prices]

    fig_psm = go.Figure()
    fig_psm.add_trace(go.Scatter(x=prices, y=pct_tc, name="Too cheap",
                                  line=dict(color="#5DCAA5", width=2, dash="dot")))
    fig_psm.add_trace(go.Scatter(x=prices, y=pct_r,  name="Reasonable",
                                  line=dict(color=PRIMARY, width=2.5)))
    fig_psm.add_trace(go.Scatter(x=prices, y=pct_e,  name="Expensive",
                                  line=dict(color=ACCENT, width=2, dash="dash")))
    fig_psm.add_trace(go.Scatter(x=prices, y=pct_te, name="Too expensive",
                                  line=dict(color=DANGER, width=2.5)))

    # Find acceptable price range (intersection of too_cheap & too_expensive)
    tc_arr  = np.array(pct_tc)
    te_arr  = np.array(pct_te)
    r_arr   = np.array(pct_r)
    e_arr   = np.array(pct_e)

    # OPP = intersection of reasonable and expensive
    diff_r_e = np.abs(r_arr - e_arr)
    opp_idx  = np.argmin(diff_r_e)
    opp_price = int(prices[opp_idx])

    # APR lower = intersection of too_cheap ascending & expensive
    diff_tc_e = r_arr - tc_arr
    apr_lo_idx = np.where(diff_tc_e >= 0)[0]
    apr_lo = int(prices[apr_lo_idx[0]]) if len(apr_lo_idx) else 100

    # APR upper = where too_expensive crosses reasonable
    diff_te_r = te_arr - (100 - r_arr)
    apr_hi_idx = np.where(diff_te_r >= 0)[0]
    apr_hi = int(prices[apr_hi_idx[0]]) if len(apr_hi_idx) else 500

    fig_psm.add_vrect(x0=apr_lo, x1=apr_hi, fillcolor=PRIMARY,
                       opacity=0.08, layer="below", line_width=0)
    fig_psm.add_vline(x=opp_price, line_color=PRIMARY, line_dash="dash", line_width=2,
                       annotation_text=f"Optimal: ₹{opp_price}",
                       annotation_font_color=PRIMARY, annotation_position="top right")

    fig_psm.update_layout(
        height=380,
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(title="Price per 30-min session (₹)", color="#ccc",
                   tickprefix="₹", showgrid=True, gridcolor="#333"),
        yaxis=dict(title="% respondents", color="#aaa",
                   showgrid=True, gridcolor="#333"),
        legend=dict(font=dict(color="#ccc")),
        font=dict(color="#ccc"),
        margin=dict(t=20,b=40,l=0,r=0),
    )
    st.plotly_chart(fig_psm, use_container_width=True)

    col_p1, col_p2, col_p3 = st.columns(3)
    col_p1.metric("Acceptable Price Range", f"₹{apr_lo} – ₹{apr_hi}")
    col_p2.metric("Optimal Price Point (OPP)", f"₹{opp_price}")
    col_p3.metric("Recommended Launch Price", f"₹{opp_price - 10} – ₹{opp_price + 15}")

    st.markdown(f"""
    <div class="insight-box">
    📌 <strong>Founder action:</strong> Launch price of ₹{opp_price} per 30-minute session
    maximises conversion. The acceptable range ₹{apr_lo}–₹{apr_hi} gives pricing flexibility
    for student discounts (lower bound) and premium weekend slots (upper bound).
    </div>""", unsafe_allow_html=True)
