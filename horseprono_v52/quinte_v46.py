"""HorseProno Quinté V4.6 — Dual Engine Fundamental + Market.

V4.6 ajoute un moteur fondamental strictement sans cotes au-dessus de V4.5.
La sélection officielle reste V4.5 tant que la fusion n'a pas été validée forward.
Le moteur fondamental fournit :
- un classement 100 % sans cotes ;
- un noyau de consensus avec la sélection V4.5 ;
- une réserve fondamentale (chevaux Top7 fondamental absents du Top7 V4.5).

Aucune variable odds / market_prob / dynamique de cote n'entre dans le moteur fondamental.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

from v46_fundamental_features import prepare_fundamental

V46_CONFIG = {
    "mode": "DUAL_ENGINE_SHADOW",
    "fundamental_shortlist_size": 7,
    "official_shortlist_source": "V4.5",
    "fundamental_weights": {
        "shortlist_top5": 0.52,
        "shortlist_ranker": 0.28,
        "shortlist_context": 0.20,
        "winner_top5": 0.45,
        "winner_ranker": 0.40,
        "winner_context": 0.15,
    },
}


def load_fundamental_artifact(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def _rankpct(a: np.ndarray) -> np.ndarray:
    s = pd.Series(np.asarray(a, dtype=float))
    n = len(s)
    if n <= 1:
        return np.ones(n, dtype=float)
    r = s.rank(method="average", ascending=False).to_numpy(dtype=float)
    return 1.0 - (r - 1.0) / (n - 1.0)


def _predict_logistic(artifact: dict[str, Any], X: pd.DataFrame) -> np.ndarray:
    lg = artifact["top5_logistic"]
    cols = lg["features"]
    missing = [c for c in cols if c not in X.columns]
    if missing:
        raise RuntimeError("Variables fondamentales absentes: " + ", ".join(missing))
    x = X[cols].to_numpy(dtype=float)
    mean = np.asarray(lg["mean"], dtype=float)
    scale = np.asarray(lg["scale"], dtype=float)
    scale = np.where(np.abs(scale) < 1e-12, 1.0, scale)
    coef = np.asarray(lg["coef"], dtype=float)
    return _sigmoid(((x - mean) / scale) @ coef + float(lg["intercept"]))


def add_v46_fundamental(
    history: pd.DataFrame,
    race: pd.DataFrame,
    ranked_v45: pd.DataFrame,
    artifact: dict[str, Any],
    ranker: Any,
) -> pd.DataFrame:
    """Ajoute le moteur fondamental sans modifier l'ordre officiel V4.5."""
    if ranker is None:
        raise ValueError("V4.6 nécessite le ranker fondamental.")

    race = race.copy().reset_index(drop=True)
    X = prepare_fundamental(history, race).reset_index(drop=True)
    p = _predict_logistic(artifact, X)

    rank_features = artifact.get("ranker", {}).get("features", [])
    missing = [c for c in rank_features if c not in X.columns]
    if missing:
        raise RuntimeError("Variables ranker fondamental absentes: " + ", ".join(missing))
    rs = np.asarray(ranker.predict(X[rank_features]), dtype=float)

    p_pct = _rankpct(p)
    r_pct = _rankpct(rs)
    ctx = (
        pd.to_numeric(X["form_rankpct"], errors="coerce").fillna(0.0).to_numpy()
        + pd.to_numeric(X["music_rankpct"], errors="coerce").fillna(0.0).to_numpy()
        + pd.to_numeric(X["entity_context_mean"], errors="coerce").fillna(0.0).to_numpy()
        + pd.to_numeric(X["fundamental_consensus"], errors="coerce").fillna(0.0).to_numpy()
    ) / 4.0

    w = artifact.get("score_weights", V46_CONFIG["fundamental_weights"])
    sw = w.get("shortlist", {})
    ww = w.get("winner", {})
    shortlist = (
        float(sw.get("top5", 0.52)) * p_pct
        + float(sw.get("ranker", 0.28)) * r_pct
        + float(sw.get("context", 0.20)) * ctx
    )
    winner = (
        float(ww.get("top5", 0.45)) * p_pct
        + float(ww.get("ranker", 0.40)) * r_pct
        + float(ww.get("context", 0.15)) * ctx
    )

    f = race[["horse_number"]].copy()
    f["v46_fund_top5_probability"] = p
    f["v46_fund_model_pct"] = p_pct
    f["v46_fund_ranker_score"] = rs
    f["v46_fund_ranker_pct"] = r_pct
    f["v46_fund_context"] = ctx
    f["v46_fundamental_score"] = shortlist
    f["v46_fundamental_winner_score"] = winner
    f = f.sort_values(
        ["v46_fundamental_score", "v46_fund_top5_probability", "horse_number"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    f["v46_fundamental_rank"] = np.arange(1, len(f) + 1)
    f["v46_fundamental_top7"] = f["v46_fundamental_rank"].le(int(V46_CONFIG["fundamental_shortlist_size"]))

    out = ranked_v45.copy().reset_index(drop=True)
    out = out.merge(f, on="horse_number", how="left", validate="one_to_one")
    out["v46_market_top7"] = out["selected_top7"].astype(bool)
    out["v46_consensus"] = out["v46_market_top7"] & out["v46_fundamental_top7"].fillna(False).astype(bool)
    out["v46_market_only"] = out["v46_market_top7"] & ~out["v46_fundamental_top7"].fillna(False).astype(bool)
    out["v46_fundamental_only"] = (~out["v46_market_top7"]) & out["v46_fundamental_top7"].fillna(False).astype(bool)
    out["v46_dual_role"] = "HORS_DOUBLE_TOP7"
    out.loc[out["v46_consensus"], "v46_dual_role"] = "NOYAU_CONSENSUS"
    out.loc[out["v46_market_only"], "v46_dual_role"] = "CHALLENGER_MARCHE"
    out.loc[out["v46_fundamental_only"], "v46_dual_role"] = "RESERVE_FONDAMENTALE"

    # Meilleur cheval de réserve fondamentale : information shadow uniquement.
    reserve = out[out["v46_fundamental_only"]].sort_values(
        ["v46_fundamental_rank", "v46_fundamental_score", "horse_number"],
        ascending=[True, False, True],
    )
    reserve_num = np.nan if reserve.empty else int(reserve.iloc[0]["horse_number"])
    out["v46_reserve_number"] = reserve_num
    out["v46_consensus_count"] = int(out["v46_consensus"].sum())
    out["v46_mode"] = V46_CONFIG["mode"]

    # IMPORTANT : ordre et selected_top7 restent exactement ceux de V4.5.
    return out
