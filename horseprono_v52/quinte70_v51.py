"""HorseProno V5.1 — 70 combinaisons Quinté rétro-calibrées.

Le générateur sépare deux choses :
1) le signal du jour (V5.1/V4.6 + rang officiel) ;
2) un prior historique appris uniquement sur les 30 premières courses walk-forward.

Le pool combinatoire n'est plus un simple Top10 de marché :
- les 7 premiers officiels sont conservés ;
- les 3 dernières places sont attribuées aux meilleures réserves fondamentales
  (avec repli sur le classement officiel si nécessaire).

Parmi les 252 combinaisons possibles de 5 chevaux dans ce pool de 10, on score
chaque ticket avec 60% de signal courant et 40% de prior historique, puis on
retient les 70 meilleurs tickets. Ce score n'est PAS une probabilité de gain.
"""
from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Any
import json
import math

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = BASE_DIR / "v51_quinte70_history_artifact.json"

_FALLBACK = {
    "rank_probabilities": {
        "1": 0.5322580645, "2": 0.6290322581, "3": 0.3709677419,
        "4": 0.4032258065, "5": 0.6612903226, "6": 0.4677419355,
        "7": 0.3709677419, "8": 0.1451612903, "9": 0.2096774194,
        "10": 0.2419354839,
    },
    "top5_structure_probabilities": {
        "0": 0.0151515152, "1": 0.1060606061, "2": 0.2575757576,
        "3": 0.5303030303, "4": 0.0757575758, "5": 0.0151515152,
    },
    "weights": {"current_smart_signal": 0.60, "historical_pattern": 0.40},
}

RANK_PRIOR = [1.00, 0.94, 0.88, 0.82, 0.76, 0.65, 0.57, 0.50, 0.43, 0.36]


def _load_history_artifact() -> dict[str, Any]:
    if ARTIFACT_PATH.exists():
        try:
            with ARTIFACT_PATH.open("r", encoding="utf-8") as fh:
                obj = json.load(fh)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return _FALLBACK


def _num_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)


def _bool_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    return df[col].fillna(False).astype(bool)


def _relative(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.notna().sum() < 2:
        return pd.Series(0.5, index=s.index, dtype=float)
    lo, hi = float(s.min()), float(s.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(0.5, index=s.index, dtype=float)
    return ((s - lo) / (hi - lo)).fillna(0.5)


def _percentile(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if len(values) <= 1:
        return np.full_like(values, 0.5, dtype=float)
    # rank stable, 0..1
    order = np.argsort(values, kind="mergesort")
    out = np.empty(len(values), dtype=float)
    out[order] = np.arange(len(values), dtype=float) / float(len(values) - 1)
    return out


def _prepare_pool(ranked: pd.DataFrame) -> pd.DataFrame:
    """Top10 combinatoire = Top7 officiel + 3 meilleures réserves fondamentales."""
    work = ranked.copy()
    if "quinte_rank" in work.columns:
        work["_official_rank"] = pd.to_numeric(work["quinte_rank"], errors="coerce")
        work = work.sort_values(["_official_rank", "horse_number"], na_position="last")
    else:
        work = work.reset_index(drop=True)
        work["_official_rank"] = np.arange(1, len(work) + 1)

    work["horse_number"] = pd.to_numeric(work["horse_number"], errors="coerce").astype("Int64")
    work = work[work["horse_number"].notna()].drop_duplicates("horse_number").copy()
    if len(work) < 10:
        return work.head(10).copy()

    top7 = work.head(7).copy()
    used = set(top7["horse_number"].astype(int).tolist())
    rest = work[~work["horse_number"].astype(int).isin(used)].copy()

    # Les réserves fondamentales ont montré plus de complémentarité historique
    # que de simples fillers de marché. Le méta-score départage les égalités.
    if "v46_fundamental_rank" in rest.columns:
        rest["_fund_rank"] = pd.to_numeric(rest["v46_fundamental_rank"], errors="coerce").fillna(999)
    else:
        rest["_fund_rank"] = 999.0
    rest["_fund_score_sort"] = _num_series(rest, "v46_fundamental_score", 0.0)
    rest["_meta_sort"] = _num_series(rest, "v51_meta_top5_score", 0.0)
    rest = rest.sort_values(
        ["_fund_rank", "_fund_score_sort", "_meta_sort", "_official_rank"],
        ascending=[True, False, False, True],
        na_position="last",
    )
    reserve = rest.head(3).copy()

    pool = pd.concat([top7, reserve], ignore_index=True)
    if len(pool) < 10:
        missing = 10 - len(pool)
        pool_ids = set(pool["horse_number"].astype(int).tolist())
        fallback = work[~work["horse_number"].astype(int).isin(pool_ids)].head(missing)
        pool = pd.concat([pool, fallback], ignore_index=True)

    pool = pool.head(10).copy().reset_index(drop=True)
    pool["_rank"] = np.arange(1, len(pool) + 1)
    pool["_rank_prior"] = RANK_PRIOR[: len(pool)]

    meta_top5 = _relative(_num_series(pool, "v51_meta_top5_score", np.nan))
    meta_win = _relative(_num_series(pool, "v51_meta_winner_score", np.nan))
    fund = _relative(_num_series(pool, "v46_fundamental_score", np.nan))
    dyn_raw = _num_series(pool, "v45_dynamics_score", 0.0).clip(-1.0, 1.0)
    dyn = (dyn_raw + 1.0) / 2.0
    consensus = _bool_series(pool, "v46_consensus").astype(float)
    shadow = _bool_series(pool, "v51_shadow_top7").astype(float)
    fund7 = _bool_series(pool, "v46_fundamental_top7").astype(float)

    pool["_smart_score"] = (
        0.48 * pool["_rank_prior"]
        + 0.17 * meta_top5
        + 0.08 * meta_win
        + 0.08 * fund
        + 0.08 * consensus
        + 0.05 * shadow
        + 0.03 * fund7
        + 0.03 * dyn
    ).clip(0.0, 1.2)
    return pool


def _historical_pattern_score(combo_ranks: tuple[int, ...], artifact: dict[str, Any]) -> float:
    rank_p = artifact.get("rank_probabilities", _FALLBACK["rank_probabilities"])
    struct_p = artifact.get("top5_structure_probabilities", _FALLBACK["top5_structure_probabilities"])
    score = 0.0
    for r in combo_ranks:
        p = float(rank_p.get(str(r), 0.5))
        p = min(max(p, 1e-5), 1.0 - 1e-5)
        score += math.log(p / (1.0 - p))
    k = sum(r <= 5 for r in combo_ranks)
    sp = max(float(struct_p.get(str(k), 1e-5)), 1e-5)
    score += math.log(sp)
    return float(score)


def _scenario(k_top5: int) -> str:
    return {
        5: "Noyau pur 5+0",
        4: "Noyau fort 4+1",
        3: "Équilibré 3+2",
        2: "Ouvert 2+3",
        1: "Très ouvert 1+4",
        0: "Extrême 0+5",
    }.get(int(k_top5), f"Structure {k_top5}+{5-k_top5}")


def generate_quinte_70(ranked: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Retourne (70 tickets, exposition du pool10, résumé des scénarios)."""
    if ranked is None or ranked.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    work = _prepare_pool(ranked)
    if len(work) < 10:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    artifact = _load_history_artifact()
    weights = artifact.get("weights", _FALLBACK["weights"])
    w_current = float(weights.get("current_smart_signal", 0.60))
    w_history = float(weights.get("historical_pattern", 0.40))
    total = max(w_current + w_history, 1e-9)
    w_current, w_history = w_current / total, w_history / total

    horses = work["horse_number"].astype(int).tolist()
    smart = {int(row["horse_number"]): float(row["_smart_score"]) for _, row in work.iterrows()}
    rank_to_horse = {i + 1: h for i, h in enumerate(horses)}

    candidates: list[dict[str, Any]] = []
    history_raw: list[float] = []
    current_raw: list[float] = []
    combos = list(combinations(range(1, 11), 5))

    for ranks in combos:
        hs = tuple(rank_to_horse[r] for r in ranks)
        current_score = float(np.mean([smart[h] for h in hs]))
        history_score = _historical_pattern_score(ranks, artifact)
        current_raw.append(current_score)
        history_raw.append(history_score)
        candidates.append({"ranks": ranks, "horses": hs, "current_raw": current_score, "history_raw": history_score})

    current_pct = _percentile(np.asarray(current_raw, dtype=float))
    history_pct = _percentile(np.asarray(history_raw, dtype=float))
    for i, c in enumerate(candidates):
        c["current_pct"] = float(current_pct[i])
        c["history_pct"] = float(history_pct[i])
        c["final_score"] = w_current * c["current_pct"] + w_history * c["history_pct"]
        c["k_top5"] = sum(r <= 5 for r in c["ranks"])

    chosen = sorted(
        candidates,
        key=lambda c: (-c["final_score"], -c["current_pct"], -c["history_pct"], c["ranks"]),
    )[:70]

    rows: list[dict[str, Any]] = []
    exposure = {h: 0 for h in horses}
    for idx, c in enumerate(chosen, start=1):
        for h in c["horses"]:
            exposure[h] += 1
        rows.append({
            "Ticket": idx,
            "Scénario": _scenario(c["k_top5"]),
            "C1": c["horses"][0], "C2": c["horses"][1], "C3": c["horses"][2],
            "C4": c["horses"][3], "C5": c["horses"][4],
            "Combinaison": " - ".join(map(str, c["horses"])),
            "Top5 présents": c["k_top5"],
            "Indice modèle": round(100.0 * c["current_pct"], 1),
            "Indice historique": round(100.0 * c["history_pct"], 1),
            "Score combiné": round(100.0 * c["final_score"], 1),
        })

    names = {}
    if "horse_name" in work.columns:
        names = {int(r["horse_number"]): str(r.get("horse_name", "")) for _, r in work.iterrows()}

    exposure_rows = []
    for i, h in enumerate(horses, start=1):
        row = work.iloc[i - 1]
        exposure_rows.append({
            "Rang pool": i,
            "N°": h,
            "Cheval": names.get(h, ""),
            "Présence /70": exposure[h],
            "Taux": exposure[h] / 70.0,
            "Force intelligente": round(100.0 * float(row["_smart_score"]), 1),
            "Rang officiel": int(row["_official_rank"]) if pd.notna(row.get("_official_rank")) else None,
            "Consensus": bool(row.get("v46_consensus", False)),
            "V5.1 shadow": bool(row.get("v51_shadow_top7", False)),
        })

    ticket_df = pd.DataFrame(rows)
    scenario_order = [
        "Noyau pur 5+0", "Noyau fort 4+1", "Équilibré 3+2",
        "Ouvert 2+3", "Très ouvert 1+4", "Extrême 0+5",
    ]
    counts = ticket_df["Scénario"].value_counts().to_dict()
    scenario_rows = [
        {"Scénario": s, "Tickets": int(counts.get(s, 0)), "Part": float(counts.get(s, 0)) / 70.0}
        for s in scenario_order if counts.get(s, 0) > 0
    ]
    return ticket_df, pd.DataFrame(exposure_rows), pd.DataFrame(scenario_rows)
