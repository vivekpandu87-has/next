import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix,
                              roc_curve, mean_squared_error, r2_score,
                              mean_absolute_error)
from sklearn.model_selection import train_test_split
try:
    from mlxtend.frequent_patterns import apriori, association_rules
except ImportError:
    from apriori_utils import apriori, association_rules
import joblib, os, warnings
warnings.filterwarnings("ignore")

from preprocessing import (encode, get_cluster_features, get_classification_features,
                            get_regression_features, get_basket_df,
                            get_conversion_target, get_spend_target)

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_all(df: pd.DataFrame):
    results = {}
    df_enc = encode(df)

    # ── CLUSTERING ────────────────────────────────────────────────────────────
    X_clust = get_cluster_features(df_enc)
    scaler_clust = StandardScaler()
    X_clust_s = scaler_clust.fit_transform(X_clust)

    inertias, silhouettes = [], []
    from sklearn.metrics import silhouette_score
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labs = km.fit_predict(X_clust_s)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_clust_s, labs))

    best_k = 5
    km_final = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    cluster_labels = km_final.fit_predict(X_clust_s)
    df_enc["cluster"] = cluster_labels

    joblib.dump(km_final,      f"{MODEL_DIR}/kmeans.pkl")
    joblib.dump(scaler_clust,  f"{MODEL_DIR}/scaler_clust.pkl")
    joblib.dump(X_clust.columns.tolist(), f"{MODEL_DIR}/cluster_features.pkl")

    # Persona naming
    cluster_profiles = []
    persona_map = {}
    for c in range(best_k):
        mask = df_enc["cluster"] == c
        sub  = df_enc[mask]
        prof = {
            "cluster": c,
            "size": int(mask.sum()),
            "avg_income": float(sub["income_num"].mean()),
            "avg_role": float(sub["role_num"].mean()),
            "avg_pod_interest": float(sub["pod_interest"].mean()),
            "avg_spend": float(sub["realistic_monthly_spend"].mean()) if "realistic_monthly_spend" in sub else 0,
            "avg_age": float(sub["age_num"].mean()),
            "conversion_rate": float((sub["pod_conversion_binary"]==1).mean()) if "pod_conversion_binary" in sub else 0,
        }
        cluster_profiles.append(prof)
        # Auto persona name
        if prof["avg_role"] >= 3.5 and prof["avg_income"] <= 2.5:
            name = "Rising Star"
        elif prof["avg_role"] >= 3.0 and prof["avg_income"] >= 3.5:
            name = "Elite Competitor"
        elif prof["avg_income"] >= 4.0 and prof["avg_role"] <= 2.0:
            name = "Corporate Cricket Fan"
        elif prof["avg_pod_interest"] <= 2.5:
            name = "Sceptic / Disengaged"
        else:
            name = "Recreational Player"
        persona_map[c] = name

    joblib.dump(cluster_profiles, f"{MODEL_DIR}/cluster_profiles.pkl")
    joblib.dump(persona_map,       f"{MODEL_DIR}/persona_map.pkl")

    results["clustering"] = {
        "inertias": inertias, "silhouettes": silhouettes,
        "best_k": best_k, "labels": cluster_labels,
        "profiles": cluster_profiles, "persona_map": persona_map,
    }

    # ── CLASSIFICATION ────────────────────────────────────────────────────────
    valid_mask = df_enc["pod_conversion_binary"].notna()
    df_clf     = df_enc[valid_mask].copy()
    X_clf_raw  = get_classification_features(df_clf)
    y_clf      = df_clf["pod_conversion_binary"].astype(int)

    scaler_clf = StandardScaler()
    X_clf_s    = scaler_clf.fit_transform(X_clf_raw)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_clf_s, y_clf, test_size=0.2, random_state=42, stratify=y_clf)
    X_tr_raw, X_te_raw, _, _ = train_test_split(
        X_clf_raw, y_clf, test_size=0.2, random_state=42, stratify=y_clf)

    rf = RandomForestClassifier(n_estimators=200, max_depth=10,
                                class_weight="balanced", random_state=42)
    rf.fit(X_tr, y_tr)
    rf_pred  = rf.predict(X_te)
    rf_prob  = rf.predict_proba(X_te)[:,1]
    fpr, tpr, thresh = roc_curve(y_te, rf_prob)

    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_tr, y_tr)
    lr_pred  = lr.predict(X_te)
    lr_prob  = lr.predict_proba(X_te)[:,1]
    fpr_lr, tpr_lr, _ = roc_curve(y_te, lr_prob)

    feat_imp = pd.Series(rf.feature_importances_,
                         index=X_clf_raw.columns).sort_values(ascending=False)

    joblib.dump(rf,          f"{MODEL_DIR}/rf_classifier.pkl")
    joblib.dump(lr,          f"{MODEL_DIR}/lr_classifier.pkl")
    joblib.dump(scaler_clf,  f"{MODEL_DIR}/scaler_clf.pkl")
    joblib.dump(X_clf_raw.columns.tolist(), f"{MODEL_DIR}/clf_features.pkl")

    results["classification"] = {
        "rf": {"acc": accuracy_score(y_te, rf_pred),
               "prec": precision_score(y_te, rf_pred),
               "rec":  recall_score(y_te, rf_pred),
               "f1":   f1_score(y_te, rf_pred),
               "auc":  roc_auc_score(y_te, rf_prob),
               "cm":   confusion_matrix(y_te, rf_pred).tolist(),
               "fpr":  fpr.tolist(), "tpr": tpr.tolist(),},
        "lr": {"acc": accuracy_score(y_te, lr_pred),
               "prec": precision_score(y_te, lr_pred),
               "rec":  recall_score(y_te, lr_pred),
               "f1":   f1_score(y_te, lr_pred),
               "auc":  roc_auc_score(y_te, lr_prob),
               "cm":   confusion_matrix(y_te, lr_pred).tolist(),
               "fpr":  fpr_lr.tolist(), "tpr": tpr_lr.tolist(),},
        "feat_imp": feat_imp.head(20).to_dict(),
        "y_test": y_te.tolist(), "rf_prob": rf_prob.tolist(),
    }

    # ── ASSOCIATION RULES ─────────────────────────────────────────────────────
    basket = get_basket_df(df)
    freq_items = apriori(basket, min_support=0.05, use_colnames=True, max_len=4)
    rules = association_rules(freq_items, metric="lift", min_threshold=1.2)
    rules = rules[rules["confidence"] >= 0.50].sort_values("lift", ascending=False)
    rules["antecedents"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
    rules["consequents"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))

    joblib.dump(rules, f"{MODEL_DIR}/assoc_rules.pkl")
    results["association"] = {"rules": rules}

    # ── REGRESSION ────────────────────────────────────────────────────────────
    X_reg_raw = get_regression_features(df_enc)
    y_reg     = get_spend_target(df_enc)

    scaler_reg = StandardScaler()
    X_reg_s    = scaler_reg.fit_transform(X_reg_raw)

    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
        X_reg_s, y_reg, test_size=0.2, random_state=42)

    ridge = Ridge(alpha=1.0)
    ridge.fit(Xr_tr, yr_tr)
    ridge_pred = ridge.predict(Xr_te)

    from sklearn.linear_model import LinearRegression
    lr_reg = LinearRegression()
    lr_reg.fit(Xr_tr, yr_tr)
    lr_reg_pred = lr_reg.predict(Xr_te)

    coef_imp = pd.Series(np.abs(ridge.coef_),
                         index=X_reg_raw.columns).sort_values(ascending=False)

    joblib.dump(ridge,       f"{MODEL_DIR}/ridge_regressor.pkl")
    joblib.dump(lr_reg,      f"{MODEL_DIR}/lr_regressor.pkl")
    joblib.dump(scaler_reg,  f"{MODEL_DIR}/scaler_reg.pkl")
    joblib.dump(X_reg_raw.columns.tolist(), f"{MODEL_DIR}/reg_features.pkl")

    results["regression"] = {
        "ridge": {"r2":   r2_score(yr_te, ridge_pred),
                  "rmse": np.sqrt(mean_squared_error(yr_te, ridge_pred)),
                  "mae":  mean_absolute_error(yr_te, ridge_pred),
                  "y_test": yr_te.tolist(), "y_pred": ridge_pred.tolist()},
        "lr":    {"r2":   r2_score(yr_te, lr_reg_pred),
                  "rmse": np.sqrt(mean_squared_error(yr_te, lr_reg_pred)),
                  "mae":  mean_absolute_error(yr_te, lr_reg_pred)},
        "coef_imp": coef_imp.head(15).to_dict(),
    }

    joblib.dump(results, f"{MODEL_DIR}/all_results.pkl")
    print("All models trained and saved.")
    return results, df_enc

if __name__ == "__main__":
    df = pd.read_csv("cricket_pod_survey_data.csv")
    train_all(df)
