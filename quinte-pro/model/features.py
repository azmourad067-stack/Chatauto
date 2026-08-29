"""Construction des features par partant.

CONTRAT DE COHERENCE : la liste FEATURES est la reference unique, partagee
par le generateur de donnees (dataset.py), l'entrainement (train.py) et
l'inference (predict.py). Ne jamais l'etendre d'un cote sans l'etendre
des autres, sinon le modele plante ou predit n'importe quoi.
"""
from __future__ import annotations

import math

FEATURES = [
    # --- cheval ---
    "horse_win_rate",        # taux de victoires en carriere
    "horse_place_rate",      # taux d'arrivees dans les 3 (ou 5)
    "horse_last_rank",       # place du dernier outing (1 = vainqueur)
    "horse_avg_rank_6",      # place moyenne sur les 6 dernieres courses
    "horse_career_gains_log",# log(1 + gains en carriere, euros)
    "horse_age",             # age en annees
    # --- driver ---
    "driver_win_rate",
    "driver_place_rate",
    "driver_recent_rank_avg",
    "driver_horse_courses",  # nb de courses communes driver/cheval
    "combo_win_rate",        # taux de reussite du duo driver/cheval
    # --- contexte ---
    "discipline_win_rate",   # reussite du cheval dans la discipline du jour
    "hippodrome_win_rate",   # reussite du cheval sur cet hippodrome
    "dist_ratio",            # distance du jour / distance preferee du cheval
    "rest_days",             # jours de repos depuis la derniere course
    "music_top5_ratio",      # part de top-5 sur la musique recente
    # --- course ---
    "odds_proxy",            # proxy de cote (gains relatifs), plus petit = favori
    "n_partants",
    "distance_log",          # log(distance du jour en metres)
    # --- recherche web temps reel ---
    "web_sentiment",         # sentiment des snippets (-1..1)
    "data_quality",          # 0..1 sur la richesse des donnees collectees
]


def build_features_df(race: dict, partants: list[dict], medians: dict) -> "pd.DataFrame":
    """Transforme une course + partants (dicts enrichis par le scraper)
    en DataFrame avec EXACTEMENT les colonnes FEATURES.

    Toute valeur manquante est remplacee par la mediane d'entrainement
    stockee dans le joblib (medians).
    """
    import pandas as pd

    rows = []
    n_part = len(partants)
    for p in partants:
        rows.append(_one_row(race, p, n_part))
    df = pd.DataFrame(rows, columns=FEATURES)

    # imputation par mediane d'entrainement (jamais de NaN)
    for col in FEATURES:
        if col in medians and medians[col] is not None:
            df[col] = df[col].fillna(medians[col])
    return df


def _one_row(race: dict, p: dict, n_part: int) -> list:
    gains = _num(p.get("gains"))
    distance = _num(race.get("distance")) or 2000.0
    dist_pref = _num(p.get("distance_pref"))
    if not dist_pref or dist_pref <= 0:
        dist_pref = distance  # ratio neutre = 1.0
    music = _music(p.get("musique"))

    horse_win = _clamp01(_num(p.get("horse_win_rate")))
    horse_place = _clamp01(_num(p.get("horse_place_rate")))
    driver_win = _clamp01(_num(p.get("driver_win_rate")))
    driver_place = _clamp01(_num(p.get("driver_place_rate")))
    combo = _clamp01(_num(p.get("combo_win_rate")))

    return [
        horse_win,
        horse_place,
        _num(p.get("horse_last_rank"), 10.0),
        _num(p.get("horse_avg_rank_6"), 10.0),
        math.log(1.0 + max(0.0, gains)),
        _num(p.get("age"), 6.0),
        driver_win,
        driver_place,
        _num(p.get("driver_recent_rank_avg"), 10.0),
        _num(p.get("driver_horse_courses"), 5.0),
        combo,
        _clamp01(_num(p.get("discipline_win_rate"))),
        _clamp01(_num(p.get("hippodrome_win_rate"))),
        _clamp01(distance / max(1.0, dist_pref)),
        _num(p.get("rest_days"), 21.0),
        music["top5_ratio"],
        _num(p.get("odds_proxy"), 10.0),
        float(n_part),
        math.log(max(100.0, distance)),
        _clamp(_num(p.get("web_sentiment")), -1.0, 1.0),
        _clamp01(_num(p.get("data_quality"))),
    ]


# ----------------------------------------------------------------------
# petits helpers
# ----------------------------------------------------------------------
def _num(v, default: float | None = None) -> float | None:
    if v is None:
        return default
    try:
        f = float(v)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _clamp01(v: float | None) -> float:
    return 0.0 if v is None else max(0.0, min(1.0, v))


def _clamp(v: float | None, lo: float, hi: float) -> float:
    if v is None:
        return 0.0
    return max(lo, min(hi, v))


def _music(musique) -> dict:
    from scraper.engine import parse_musique  # import tardif : pas de cycle

    return parse_musique(musique)
