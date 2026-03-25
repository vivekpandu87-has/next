"""
Pure-pandas Apriori + Association Rules.
Drop-in replacement for mlxtend when mlxtend is unavailable.
On Streamlit Cloud, mlxtend installs fine from requirements.txt.
This module is used as fallback.
"""
import pandas as pd
import numpy as np
from itertools import combinations

def apriori(df: pd.DataFrame, min_support: float = 0.05,
            use_colnames: bool = True, max_len: int = 4) -> pd.DataFrame:
    """Frequent itemset mining via Apriori algorithm."""
    n = len(df)
    cols = df.columns.tolist() if use_colnames else list(range(df.shape[1]))
    arr  = df.values.astype(bool)

    # 1-itemsets
    freq = {}
    for i, col in enumerate(cols):
        sup = arr[:, i].sum() / n
        if sup >= min_support:
            freq[frozenset([col])] = sup

    result = list(freq.items())
    prev_level = list(freq.keys())

    for length in range(2, max_len + 1):
        if not prev_level:
            break
        candidates = []
        prev_list  = sorted([sorted(fs) for fs in prev_level])
        for i in range(len(prev_list)):
            for j in range(i + 1, len(prev_list)):
                a, b = prev_list[i], prev_list[j]
                if a[:-1] == b[:-1]:
                    candidates.append(frozenset(a) | frozenset(b))

        next_level = []
        for cand in candidates:
            idx = [cols.index(c) for c in cand if c in cols]
            if len(idx) != len(cand):
                continue
            sup = (arr[:, idx].all(axis=1)).sum() / n
            if sup >= min_support:
                freq[cand] = sup
                result.append((cand, sup))
                next_level.append(cand)
        prev_level = next_level

    rows = [{"itemsets": fs, "support": sup} for fs, sup in result]
    return pd.DataFrame(rows)


def association_rules(freq_itemsets: pd.DataFrame,
                      metric: str = "lift",
                      min_threshold: float = 1.0) -> pd.DataFrame:
    """Generate association rules from frequent itemsets."""
    rows = []
    item_support = {frozenset(row["itemsets"]): row["support"]
                    for _, row in freq_itemsets.iterrows()}

    for _, row in freq_itemsets.iterrows():
        itemset = frozenset(row["itemsets"])
        if len(itemset) < 2:
            continue
        sup_ab = row["support"]
        for i in range(1, len(itemset)):
            for ant in combinations(itemset, i):
                ant = frozenset(ant)
                con = itemset - ant
                sup_a = item_support.get(ant, np.nan)
                sup_b = item_support.get(con, np.nan)
                if np.isnan(sup_a) or np.isnan(sup_b) or sup_a == 0:
                    continue
                conf = sup_ab / sup_a
                lift = conf / sup_b if sup_b > 0 else 0
                lev  = sup_ab - sup_a * sup_b
                conv = (1 - sup_b) / (1 - conf) if conf < 1 else np.inf
                rows.append({
                    "antecedents":  ant,
                    "consequents":  con,
                    "support":      sup_ab,
                    "confidence":   conf,
                    "lift":         lift,
                    "leverage":     lev,
                    "conviction":   conv,
                })

    rules = pd.DataFrame(rows)
    if rules.empty:
        return rules

    if metric == "lift":
        rules = rules[rules["lift"] >= min_threshold]
    elif metric == "confidence":
        rules = rules[rules["confidence"] >= min_threshold]
    elif metric == "support":
        rules = rules[rules["support"] >= min_threshold]

    return rules.sort_values("lift", ascending=False).reset_index(drop=True)
