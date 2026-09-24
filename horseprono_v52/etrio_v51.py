"""Générateur e-Trio V5.1.

Règle demandée : 10 combinaisons de 3 chevaux, avec exactement deux bases
prises parmi les cinq premiers du classement officiel. Les 10 paires possibles
parmi le Top5 sont utilisées une fois chacune. Le troisième cheval est choisi
dans un pool d'associés hors Top5, diversifié et priorisé par les signaux V5.1/V4.6.
"""
from __future__ import annotations

from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd


def _num(series: pd.Series | Any, default: float = 0.0) -> pd.Series:
    if isinstance(series, pd.Series):
        return pd.to_numeric(series, errors="coerce").fillna(default)
    return pd.Series(dtype=float)


def _bool_col(df: pd.DataFrame, name: str) -> pd.Series:
    if name not in df.columns:
        return pd.Series(False, index=df.index)
    return df[name].fillna(False).astype(bool)


def _associate_priority(df: pd.DataFrame) -> pd.Series:
    """Score de priorité transparent pour les associés hors Top5.

    Le score ne modifie aucun pronostic officiel : il sert uniquement à ordonner
    le pool e-Trio. Il privilégie le consensus, le Top7 V5.1 shadow, le Top7 officiel,
    le Top7 fondamental, puis les scores meta/fondamentaux disponibles.
    """
    score = pd.Series(0.0, index=df.index, dtype=float)
    score += 0.35 * _bool_col(df, "v46_consensus").astype(float)
    score += 0.25 * _bool_col(df, "v51_shadow_top7").astype(float)
    score += 0.20 * _bool_col(df, "selected_top7").astype(float)
    score += 0.15 * _bool_col(df, "v46_fundamental_top7").astype(float)

    if "v51_meta_top5_score" in df.columns:
        meta = pd.to_numeric(df["v51_meta_top5_score"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
        score += 0.20 * meta
    if "v46_fundamental_score" in df.columns:
        fund = pd.to_numeric(df["v46_fundamental_score"], errors="coerce")
        if fund.notna().any():
            lo, hi = float(fund.min()), float(fund.max())
            norm = (fund - lo) / (hi - lo) if hi > lo else pd.Series(0.5, index=df.index)
            score += 0.10 * norm.fillna(0.0)

    # Petit bonus de rang officiel pour départager proprement les ex-aequo.
    if "quinte_rank" in df.columns:
        rank = pd.to_numeric(df["quinte_rank"], errors="coerce").fillna(len(df) + 1)
        score += 0.05 * (1.0 - ((rank - 1.0) / max(float(len(df)), 1.0))).clip(0.0, 1.0)
    return score


def generate_etrio_10(ranked: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retourne (10 combinaisons, pool d'associés).

    - Les cinq premières positions officielles servent de bases potentielles.
    - Les 10 paires C(5,2) sont toutes utilisées exactement une fois.
    - Les associés sont hors Top5 et les cinq meilleurs sont retenus quand possible.
    - Chaque associé est utilisé de façon équilibrée (deux fois si 5 associés).
    - Les paires les plus fortes reçoivent d'abord les associés les plus aventureux,
      et les paires les plus faibles les associés les plus solides, afin d'équilibrer
      les tickets.
    """
    if ranked is None or ranked.empty:
        return pd.DataFrame(), pd.DataFrame()

    work = ranked.copy()
    if "quinte_rank" in work.columns:
        work = work.sort_values(["quinte_rank", "horse_number"], na_position="last").reset_index(drop=True)
    else:
        work = work.reset_index(drop=True)

    work["horse_number"] = pd.to_numeric(work["horse_number"], errors="coerce").astype("Int64")
    work = work[work["horse_number"].notna()].copy()
    work["horse_number"] = work["horse_number"].astype(int)
    work = work.drop_duplicates("horse_number").reset_index(drop=True)

    if len(work) < 6:
        return pd.DataFrame(), pd.DataFrame()

    bases = work.head(5)["horse_number"].astype(int).tolist()
    base_set = set(bases)

    assoc = work[~work["horse_number"].isin(base_set)].copy()
    if assoc.empty:
        return pd.DataFrame(), pd.DataFrame()

    assoc["etrio_priority"] = _associate_priority(assoc)
    # Tie-breaks : meilleur score meta, meilleur rang fondamental puis rang officiel.
    assoc["_meta"] = pd.to_numeric(assoc.get("v51_meta_top5_score"), errors="coerce").fillna(-1.0) if "v51_meta_top5_score" in assoc.columns else -1.0
    assoc["_fund_rank"] = pd.to_numeric(assoc.get("v46_fundamental_rank"), errors="coerce").fillna(999.0) if "v46_fundamental_rank" in assoc.columns else 999.0
    assoc["_official_rank"] = pd.to_numeric(assoc.get("quinte_rank"), errors="coerce").fillna(999.0) if "quinte_rank" in assoc.columns else 999.0
    assoc = assoc.sort_values(
        ["etrio_priority", "_meta", "_fund_rank", "_official_rank", "horse_number"],
        ascending=[False, False, True, True, True],
    ).head(5).reset_index(drop=True)

    associate_nums = assoc["horse_number"].astype(int).tolist()
    if not associate_nums:
        return pd.DataFrame(), pd.DataFrame()

    # Les 10 paires sont naturellement ordonnées de la plus forte à la plus ouverte
    # selon les positions dans le Top5.
    pair_rows = []
    for i, j in combinations(range(5), 2):
        pair_rows.append((bases[i], bases[j], i + j))
    pair_rows.sort(key=lambda x: x[2])

    # Assoc. du moins prioritaire au plus prioritaire pour équilibrer les bases :
    # paire forte -> associé plus aventureux ; paire faible -> associé plus solide.
    reverse_assoc = list(reversed(associate_nums))
    schedule = [reverse_assoc[k % len(reverse_assoc)] for k in range(10)]

    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int]] = set()
    for idx, ((b1, b2, _), a) in enumerate(zip(pair_rows, schedule), start=1):
        combo = tuple(sorted((int(b1), int(b2), int(a))))
        if combo in seen:
            # Cas rare si le pool associé est très petit : chercher le premier associé
            # qui conserve l'unicité du ticket.
            for alt in associate_nums:
                candidate = tuple(sorted((int(b1), int(b2), int(alt))))
                if candidate not in seen:
                    a = int(alt)
                    combo = candidate
                    break
        seen.add(combo)
        rows.append({
            "Ticket": idx,
            "Base 1": int(b1),
            "Base 2": int(b2),
            "Associé": int(a),
            "Combinaison": f"{int(b1)} - {int(b2)} - {int(a)}",
        })

    pool_cols = [c for c in [
        "horse_number", "horse_name", "quinte_rank", "etrio_priority",
        "v51_meta_top5_score", "v46_fundamental_rank", "v46_dual_role",
        "v51_meta_role", "selected_top7", "v51_shadow_top7", "v46_fundamental_top7",
    ] if c in assoc.columns]
    pool = assoc[pool_cols].copy()
    return pd.DataFrame(rows), pool
