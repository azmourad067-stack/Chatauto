"""HorseProno V5.1 — Meta-Consensus (shadow mode).

V5.1 ne remplace pas la sélection officielle V4.5/V4.6. Elle exploite uniquement
l'union du Top7 marché et du Top7 fondamental pour mesurer :
- la force du consensus ;
- la divergence marché ↔ fondamental ;
- un score meta Top5 par candidat ;
- un score meta Winner par candidat ;
- un Top7 / Top3 alternatifs en shadow mode.

Le méta-modèle a été entraîné sur les 30 premières courses walk-forward et évalué
sur les 28 suivantes. Comme il ne bat pas encore la baseline sur ce bloc, il reste
strictement informatif : `selected_top7` et `quinte_rank` ne sont jamais modifiés.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

V51_CONFIG = {
    "mode": "META_CONSENSUS_SHADOW",
    "official_source": "V4.5_V4.6",
    "candidate_pool": "UNION_MARKET_FUNDAMENTAL_TOP7",
    "shadow_top7_size": 7,
    "shadow_top3_size": 3,
    "trained_races": 30,
    "temporal_check_races": 28,
}


def load_v51_artifact(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def _predict_exported(model: dict[str, Any], X: pd.DataFrame, features: list[str]) -> np.ndarray:
    x = X[features].to_numpy(dtype=float)
    mean = np.asarray(model["mean"], dtype=float)
    scale = np.asarray(model["scale"], dtype=float)
    scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
    coef = np.asarray(model["coef"], dtype=float)
    z = ((x - mean) / scale) @ coef + float(model["intercept"])
    return _sigmoid(z)


def _rank_to_pct(rank: pd.Series, max_rank: int = 7) -> pd.Series:
    r = pd.to_numeric(rank, errors="coerce")
    return pd.Series(np.where(r.le(max_rank), 1.0 - (r - 1.0) / float(max_rank), 0.0), index=rank.index)


def _relative(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() < 2:
        return pd.Series(0.5, index=series.index, dtype=float)
    lo, hi = float(s.min()), float(s.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(0.5, index=series.index, dtype=float)
    return ((s - lo) / (hi - lo)).fillna(0.5)


def _build_features(out: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=out.index)
    X["market_pct"] = pd.to_numeric(out.get("market_rankpct"), errors="coerce").fillna(0.0)
    X["logit_pct"] = pd.to_numeric(out.get("model_rankpct"), errors="coerce").fillna(0.0)
    X["ranker_pct"] = pd.to_numeric(out.get("ranker_percentile"), errors="coerce").fillna(0.0)
    X["form_pct"] = pd.to_numeric(out.get("form_pct"), errors="coerce").fillna(0.0)
    X["entity_pct"] = pd.to_numeric(out.get("entity_pct"), errors="coerce").fillna(0.0)
    X["context_pct"] = pd.to_numeric(out.get("context_score"), errors="coerce").fillna(0.0)
    X["model_cons_pct"] = pd.to_numeric(out.get("model_cons_pct"), errors="coerce").fillna(0.0)

    X["fund_top5_prob"] = pd.to_numeric(out.get("v46_fund_top5_probability"), errors="coerce").fillna(0.0)
    X["fund_model_pct"] = pd.to_numeric(out.get("v46_fund_model_pct"), errors="coerce").fillna(0.0)
    X["fund_ranker_pct"] = pd.to_numeric(out.get("v46_fund_ranker_pct"), errors="coerce").fillna(0.0)
    X["fund_context"] = pd.to_numeric(out.get("v46_fund_context"), errors="coerce").fillna(0.0)
    X["fund_shortlist_score"] = pd.to_numeric(out.get("v46_fundamental_score"), errors="coerce").fillna(0.0)
    X["fund_winner_score"] = pd.to_numeric(out.get("v46_fundamental_winner_score"), errors="coerce").fillna(0.0)

    X["in_market_top7"] = out.get("v46_market_top7", False).astype(bool).astype(int)
    X["in_fund_top7"] = out.get("v46_fundamental_top7", False).fillna(False).astype(bool).astype(int)
    X["consensus"] = out.get("v46_consensus", False).astype(bool).astype(int)

    # Le méta-modèle historique utilisait l'ordre V4.4 avant ajustement dynamique.
    # v45_original_rank conserve précisément cet ordre de référence.
    market_rank = pd.to_numeric(out.get("v45_original_rank", out.get("quinte_rank")), errors="coerce")
    market_rank = market_rank.where(X["in_market_top7"].eq(1), 8.0).fillna(8.0)
    fund_rank = pd.to_numeric(out.get("v46_fundamental_rank"), errors="coerce")
    fund_rank = fund_rank.where(X["in_fund_top7"].eq(1), 8.0).fillna(8.0)
    X["market_rank_pct7"] = _rank_to_pct(market_rank)
    X["fund_rank_pct7"] = _rank_to_pct(fund_rank)
    X["rank_gap_abs"] = (market_rank - fund_rank).abs() / 7.0
    X["rank_gap_signed"] = (fund_rank - market_rank) / 7.0

    X["model_fund_gap"] = X["logit_pct"] - X["fund_model_pct"]
    X["ranker_fund_gap"] = X["ranker_pct"] - X["fund_ranker_pct"]
    X["context_fund_gap"] = X["context_pct"] - X["fund_context"]
    X["avg_model_pct"] = (X["logit_pct"] + X["fund_model_pct"]) / 2.0
    X["avg_ranker_pct"] = (X["ranker_pct"] + X["fund_ranker_pct"]) / 2.0
    X["avg_context"] = (X["context_pct"] + X["fund_context"]) / 2.0
    X["consensus_strength"] = X["consensus"] * (X["market_rank_pct7"] + X["fund_rank_pct7"]) / 2.0

    odds = pd.to_numeric(out.get("odds"), errors="coerce").clip(lower=1.01)
    X["log_odds"] = np.log(odds).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    distance = pd.to_numeric(out.get("distance"), errors="coerce")
    field = pd.to_numeric(out.get("field_size"), errors="coerce")
    if field.isna().all():
        field = pd.Series(float(len(out)), index=out.index)
    age = pd.to_numeric(out.get("age"), errors="coerce")
    X["distance_norm"] = distance.fillna(float(distance.median()) if distance.notna().any() else 2000.0) / 3000.0
    X["field_norm"] = field.fillna(float(len(out))) / 20.0
    X["draw_rel"] = _relative(out.get("draw", pd.Series(np.nan, index=out.index)))
    X["weight_rel"] = _relative(out.get("weight", pd.Series(np.nan, index=out.index)))
    X["age_norm"] = age.fillna(float(age.median()) if age.notna().any() else 5.0) / 10.0

    discipline = out.get("discipline", pd.Series("", index=out.index)).astype(str).str.upper()
    X["is_plat"] = discipline.eq("PLAT").astype(int)
    X["is_trot"] = discipline.str.startswith("ATTELE").astype(int)
    X["is_obstacle"] = discipline.isin(["HAIES", "STEEPLECHASE", "CROSS_COUNTRY"]).astype(int)
    return X.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def add_v51_meta_consensus(ranked_v46: pd.DataFrame, artifact: dict[str, Any]) -> pd.DataFrame:
    """Ajoute V5.1 en shadow mode sans modifier l'ordre officiel ni le Top7."""
    out = ranked_v46.copy().reset_index(drop=True)
    if out.empty:
        return out

    features = list(artifact["features"])
    X = _build_features(out)
    missing = [c for c in features if c not in X.columns]
    if missing:
        raise RuntimeError("Variables V5.1 absentes: " + ", ".join(missing))

    union_mask = out.get("v46_market_top7", False).astype(bool) | out.get("v46_fundamental_top7", False).fillna(False).astype(bool)
    out["v51_meta_top5_score"] = np.nan
    out["v51_meta_winner_score"] = np.nan
    out.loc[union_mask, "v51_meta_top5_score"] = _predict_exported(artifact["top5_model"], X.loc[union_mask], features)
    out.loc[union_mask, "v51_meta_winner_score"] = _predict_exported(artifact["winner_model"], X.loc[union_mask], features)

    union = out[union_mask].copy()
    consensus = union[union.get("v46_consensus", False).astype(bool)].sort_values(
        ["v51_meta_top5_score", "horse_number"], ascending=[False, True]
    )
    disagreements = union[~union.get("v46_consensus", False).astype(bool)].sort_values(
        ["v51_meta_top5_score", "horse_number"], ascending=[False, True]
    )
    shadow = pd.concat([consensus, disagreements], axis=0).drop_duplicates("horse_number").head(int(V51_CONFIG["shadow_top7_size"]))
    shadow_nums = shadow["horse_number"].astype(int).tolist()
    winner_shadow = shadow.sort_values(["v51_meta_winner_score", "v51_meta_top5_score", "horse_number"], ascending=[False, False, True])
    shadow_top3 = winner_shadow.head(int(V51_CONFIG["shadow_top3_size"]))["horse_number"].astype(int).tolist()

    out["v51_shadow_top7"] = out["horse_number"].astype(int).isin(shadow_nums)
    shadow_rank_map = {h: i + 1 for i, h in enumerate(winner_shadow["horse_number"].astype(int).tolist())}
    out["v51_shadow_rank"] = out["horse_number"].astype(int).map(shadow_rank_map)
    out["v51_shadow_top3"] = out["horse_number"].astype(int).isin(shadow_top3)

    official_nums = out[out.get("selected_top7", False).astype(bool)]["horse_number"].astype(int).tolist()
    swap_in = [h for h in shadow_nums if h not in official_nums]
    swap_out = [h for h in official_nums if h not in shadow_nums]
    out["v51_shadow_swap_in"] = ",".join(map(str, swap_in)) if swap_in else ""
    out["v51_shadow_swap_out"] = ",".join(map(str, swap_out)) if swap_out else ""

    # Diagnostic de consensus de la course.
    consensus_count = int(out.get("v46_consensus", False).astype(bool).sum())
    consensus_rows = X.loc[out.get("v46_consensus", False).astype(bool)]
    if len(consensus_rows):
        rank_agreement = float(np.clip(1.0 - consensus_rows["rank_gap_abs"].mean(), 0.0, 1.0))
    else:
        rank_agreement = 0.0
    ordered_union = union.sort_values("v51_meta_top5_score", ascending=False)
    if len(ordered_union) >= 8:
        boundary_margin = float(ordered_union.iloc[6]["v51_meta_top5_score"] - ordered_union.iloc[7]["v51_meta_top5_score"])
    else:
        boundary_margin = 0.15
    boundary_conf = float(np.clip(boundary_margin / 0.15, 0.0, 1.0))
    confidence = float(np.clip(0.50 * (consensus_count / 7.0) + 0.30 * rank_agreement + 0.20 * boundary_conf, 0.0, 1.0))
    divergence = float(np.clip(0.60 * (1.0 - consensus_count / 7.0) + 0.40 * (1.0 - rank_agreement), 0.0, 1.0))

    market_only = out[out.get("v46_market_only", False).astype(bool)]
    fund_only = out[out.get("v46_fundamental_only", False).astype(bool)]
    mscore = float(pd.to_numeric(market_only["v51_meta_top5_score"], errors="coerce").mean()) if len(market_only) else np.nan
    fscore = float(pd.to_numeric(fund_only["v51_meta_top5_score"], errors="coerce").mean()) if len(fund_only) else np.nan
    if consensus_count >= 6 and divergence < 0.25:
        stance = "CONSENSUS_FORT"
    elif np.isfinite(mscore) and np.isfinite(fscore) and (mscore - fscore) >= 0.08:
        stance = "MARCHE_META_PLUS_FORT"
    elif np.isfinite(mscore) and np.isfinite(fscore) and (fscore - mscore) >= 0.08:
        stance = "FONDAMENTAL_META_A_SURVEILLER"
    elif divergence >= 0.45:
        stance = "DIVERGENCE_ELEVEE"
    else:
        stance = "CONSENSUS_MODERE"

    out["v51_consensus_confidence"] = confidence
    out["v51_divergence_index"] = divergence
    out["v51_stance"] = stance
    out["v51_consensus_count"] = consensus_count
    out["v51_union_count"] = int(union_mask.sum())
    out["v51_mode"] = V51_CONFIG["mode"]

    role = pd.Series("HORS_UNION", index=out.index, dtype=object)
    role.loc[out.get("v46_consensus", False).astype(bool)] = "NOYAU_META_CONSENSUS"
    role.loc[out.get("v46_market_only", False).astype(bool)] = "MARCHE_SEUL_META"
    role.loc[out.get("v46_fundamental_only", False).astype(bool)] = "FONDAMENTAL_SEUL_META"
    role.loc[out.get("v46_fundamental_only", False).astype(bool) & out["v51_shadow_top7"]] = "FONDAMENTAL_SHADOW_TOP7"
    role.loc[out.get("v46_market_only", False).astype(bool) & ~out["v51_shadow_top7"]] = "MARCHE_FRAGILE_SHADOW"
    out["v51_meta_role"] = role

    # GARANTIE : V5.1 ne modifie jamais ces colonnes officielles.
    return out
