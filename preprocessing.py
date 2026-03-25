import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib, os

# ── Ordinal mappings ──────────────────────────────────────────────────────────
AGE_ORD       = {"Under 15":1,"15-18":2,"19-25":3,"26-35":4,"36-50":5,"50+":6}
CITY_ORD      = {"Rural":1,"Tier 3":2,"Tier 2":3,"Tier 1":4,"Metro":5}
INC_ORD       = {"Below 20K":1,"20K-40K":2,"40K-75K":3,"75K-150K":4,"Above 150K":5}
EDU_ORD       = {"Up to 10th":1,"12th/Diploma":2,"Bachelors":3,"Masters+":4}
PDAYS_ORD     = {"0":0,"1-2":1,"3-4":2,"5-6":3,"Daily":4}
ROLE_ORD      = {"Not interested":0,"Fan only":1,"Occasional":2,"Regular":3,"Competitive":4,"Coach":3}
SPEND_ORD     = {"0":0,"1-500":1,"501-1000":2,"1001-2500":3,"2501-5000":4,"Above 5000":5}
SPEND_MID     = {"0":0,"1-500":150,"501-1000":450,"1001-2500":900,"2501-5000":1850,"Above 5000":3000}
MEM_ORD       = {"Would not subscribe":0,"Up to 499":1,"500-999":2,"1000-1999":3,"2000-3000":4,"Above 3000":5}
DIG_ORD       = {"0":0,"1-200":1,"201-500":2,"501-1000":3,"Above 1000":4}
FD_ORD        = {"Rarely":1,"1-2/week":2,"3-4/week":3,"Daily":4}
TECH_ORD      = {"Tech avoider":0,"Laggard":1,"Late majority":2,"Early majority":3,"Early adopter":4}
DIST_ORD      = {"Within 1km":1,"Up to 3km":2,"Up to 5km":3,"Up to 10km":4,"Any distance":5}
FC_ORD        = {"Not interested":0,"Aware not using":1,"Occasional":2,"Active":3}
TS_ORD        = {"Early morning":1,"Morning":2,"Afternoon":3,"Evening":4,"Night":5}

PSM_TC_ORD    = {"Below 50":1,"50-99":2,"100-149":3,"150-199":4,"200-249":5}
PSM_R_ORD     = {"100-149":1,"150-199":2,"200-249":3,"250-299":4,"300-349":5}
PSM_E_ORD     = {"200-299":1,"300-399":2,"400-499":3,"500-599":4,"600+":5}
PSM_TE_ORD    = {"300-399":1,"400-499":2,"500-599":3,"600-799":4,"800+":5}

GENDER_ORD    = {"Male":0,"Female":1,"Other/PNS":2}

MULTI_SELECT_COLS = [
    "use_academy","use_boxcricket","use_bowlingmachine","use_homenet",
    "use_videoanalysis","use_mobilegame","use_gym",
    "disc_freetrial","disc_buy5get1","disc_student","disc_family",
    "disc_offpeak","disc_referral","disc_academy","disc_corporate",
    "feat_ai","feat_bowlingmachine","feat_batspeed","feat_footwork",
    "feat_videoreplay","feat_leaderboard","feat_progressreport","feat_appbooking",
    "brand_mrf","brand_sg","brand_ss","brand_kookaburra","brand_graynicolls",
    "brand_adidasnike","brand_decathlon","brand_nopref",
    "addon_smartbat","addon_wearables","addon_aicoaching",
    "addon_highlights","addon_fitness","addon_merch",
    "stream_hotstar","stream_jiocinema","stream_netflix",
    "stream_prime","stream_youtube","stream_sonyliv",
    "act_gym","act_yoga","act_othersport","act_swimming","act_videogaming","act_running",
    "past_boxcricket","past_trampoline","past_vr","past_bowling",
    "past_gokarting","past_fitclass","past_academy","past_golf",
    "frust_nodata","frust_coachattention","frust_timing","frust_crowded",
    "frust_distance","frust_cost","frust_equipment","frust_notracking",
    "bar_price","bar_location","bar_humancoach","bar_aidistrust",
    "bar_time","bar_notserious","bar_academy","bar_safety","bar_social",
    "hh_self","hh_child","hh_spouse","hh_sibling","hh_parent",
]

CLUSTERING_FEATURES = [
    "age_num","income_num","city_num","role_num","practice_num",
    "data_importance","pod_interest","spend_num","tech_num",
    "dist_num","nps_score","digital_num","fd_num",
    "addon_count","past_exp_count","barrier_count","feat_count",
]

CLASSIFICATION_FEATURES = [
    "age_num","gender_num","city_num","income_num","edu_num",
    "role_num","practice_num","data_importance","pod_interest",
    "spend_num","mem_num","digital_num","fd_num","tech_num",
    "dist_num","nps_score","addon_count","feat_count",
    "past_exp_count","barrier_count","frust_count",
    "past_boxcricket","past_trampoline","past_vr",
    "feat_ai","feat_bowlingmachine","feat_progressreport",
    "bar_aidistrust","bar_price","bar_location","bar_notserious",
    "use_academy","use_boxcricket",
]

REGRESSION_FEATURES = [
    "age_num","income_num","city_num","role_num","practice_num",
    "data_importance","pod_interest","spend_num","tech_num",
    "digital_num","nps_score","addon_count","feat_count",
    "past_exp_count","frust_count","barrier_count",
    "mem_num","dist_num",
]

def encode(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["age_num"]      = d["age_group"].map(AGE_ORD).fillna(3)
    d["city_num"]     = d["city_tier"].map(CITY_ORD).fillna(3)
    d["income_num"]   = d["income_bracket"].map(INC_ORD).fillna(3)
    d["edu_num"]      = d["education"].map(EDU_ORD).fillna(2)
    d["role_num"]     = d["cricket_role"].map(ROLE_ORD).fillna(1)
    d["practice_num"] = d["practice_days"].map(PDAYS_ORD).fillna(1)
    d["spend_num"]    = d["monthly_rec_spend"].map(SPEND_ORD).fillna(2)
    d["mem_num"]      = d["membership_wtp"].map(MEM_ORD).fillna(1)
    d["digital_num"]  = d["digital_spend"].map(DIG_ORD).fillna(1)
    d["fd_num"]       = d["food_delivery_freq"].map(FD_ORD).fillna(2)
    d["tech_num"]     = d["tech_adoption"].map(TECH_ORD).fillna(2)
    d["dist_num"]     = d["distance_tolerance"].map(DIST_ORD).fillna(3)
    d["gender_num"]   = d["gender"].map(GENDER_ORD).fillna(0)
    d["fc_num"]       = d["fantasy_cricket"].map(FC_ORD).fillna(1)
    d["psm_tc_num"]   = d["psm_too_cheap"].map(PSM_TC_ORD).fillna(3)
    d["psm_r_num"]    = d["psm_reasonable"].map(PSM_R_ORD).fillna(3)
    d["psm_e_num"]    = d["psm_expensive"].map(PSM_E_ORD).fillna(3)
    d["psm_te_num"]   = d["psm_too_expensive"].map(PSM_TE_ORD).fillna(3)

    # aggregate counts
    addon_cols   = ["addon_smartbat","addon_wearables","addon_aicoaching",
                    "addon_highlights","addon_fitness","addon_merch"]
    feat_cols    = ["feat_ai","feat_bowlingmachine","feat_batspeed","feat_footwork",
                    "feat_videoreplay","feat_leaderboard","feat_progressreport","feat_appbooking"]
    past_cols    = ["past_boxcricket","past_trampoline","past_vr","past_bowling",
                    "past_gokarting","past_fitclass","past_academy","past_golf"]
    barrier_cols = ["bar_price","bar_location","bar_humancoach","bar_aidistrust",
                    "bar_time","bar_notserious","bar_academy","bar_safety","bar_social"]
    frust_cols   = ["frust_nodata","frust_coachattention","frust_timing","frust_crowded",
                    "frust_distance","frust_cost","frust_equipment","frust_notracking"]

    for clist, cname in [(addon_cols,"addon_count"),(feat_cols,"feat_count"),
                         (past_cols,"past_exp_count"),(barrier_cols,"barrier_count"),
                         (frust_cols,"frust_count")]:
        available = [c for c in clist if c in d.columns]
        d[cname] = d[available].fillna(0).sum(axis=1)

    return d

def get_cluster_features(df_enc: pd.DataFrame):
    available = [c for c in CLUSTERING_FEATURES if c in df_enc.columns]
    return df_enc[available].fillna(df_enc[available].median())

def get_classification_features(df_enc: pd.DataFrame):
    available = [c for c in CLASSIFICATION_FEATURES if c in df_enc.columns]
    return df_enc[available].fillna(df_enc[available].median())

def get_regression_features(df_enc: pd.DataFrame):
    available = [c for c in REGRESSION_FEATURES if c in df_enc.columns]
    return df_enc[available].fillna(df_enc[available].median())

def get_basket_df(df: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in MULTI_SELECT_COLS if c in df.columns]
    basket = df[available].fillna(0).astype(bool)
    return basket

def get_conversion_target(df: pd.DataFrame):
    return df["pod_conversion_binary"].dropna()

def get_spend_target(df: pd.DataFrame):
    return df["realistic_monthly_spend"].fillna(df["realistic_monthly_spend"].median())
