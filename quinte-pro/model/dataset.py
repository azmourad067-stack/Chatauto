"""Generateur de jeu de donnees d'entrainement (simulation documentee).

POURQUOI UNE SIMULATION ? Le projet est livre SANS base historique fournie :
aucune API publique ne distribue de dataset PMU complet (les arrives
detaillees d'open-pmu-api ne couvrent qu'une partie et demandent a etre
auto-hebergees). Ce module produit donc un corpus d'entrainement
SYNTHETIQUE, structure exactement comme le sera un vrai historique.

MODELE DE GENERATION (realiste, sans triche) :
1. chaque partant possede des "niveaux latents" (cheval, driver, forme,
   affinite discipline, affinite hippodrome) ~ N(0,1) ;
2. la place reelle est tiree par classement d'un score latent bruite :
   score = 0.35*cheval + 0.22*driver + 0.20*forme + 0.13*discipline
           + 0.10*hippodrome + bruit(1.2)
3. le modele n'observe QUE des features NOYAUTEES derivees des latents
   (taux de reussite, classements, gains, musique...), comme dans la
   vraie vie : il doit apprendre a travers le bruit.

A VENIR : remplacer generate_synthetic_races par un chargement de votre
vrai historique (CSV) via load_real_data("chemin.csv") - voir train.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from model.features import FEATURES

DEFAULT_N_RACES = 3000


def _musique_from_form(form_level: np.ndarray, rng: np.random.Generator) -> list[str]:
    """Génère une musique crédible corrélée à la forme latente du cheval."""
    musics = []
    for f in form_level:
        n = int(rng.integers(4, 9))
        tokens = []
        for _ in range(n):
            p = 1.0 / (1.0 + np.exp(-(f * 1.4 + rng.normal(0, 0.9))))
            if p > 0.72:
                tokens.append(f"{int(rng.integers(1, 4))}p")
            elif p > 0.45:
                tokens.append(f"{int(rng.integers(4, 7))}p")
            elif p > 0.25:
                tokens.append(f"{int(rng.integers(7, 12))}p")
            else:
                tokens.append(rng.choice(["0a", "Dm", "Da", "9p", "12p"]))
            if rng.random() < 0.25:
                tokens.append(f"({1900 + int(rng.integers(20, 30))})")
        musics.append(" ".join(tokens))
    return musics


def generate_synthetic_races(n_races: int = DEFAULT_N_RACES, seed: int = 42) -> pd.DataFrame:
    """Cree le corpus d'entrainement (1 ligne par partant)."""
    rng = np.random.default_rng(seed)
    rows = []

    for race_id in range(n_races):
        n = int(rng.integers(8, 21))
        hippodrome_id = int(rng.integers(0, 40))
        discipline = rng.choice(["plat", "trot_attele", "trot_monte", "obstacle"])
        distance = int(rng.integers(1400, 3200))

        # --- niveaux latents (invisibles pour le modele) ---
        horse_level = rng.normal(0, 1, n)
        driver_level = rng.normal(0, 1, n)
        form_level = rng.normal(0, 1, n)
        disc_aff = rng.normal(0, 1, n)
        hippo_aff = rng.normal(0, 1, n)
        age = rng.integers(3, 11, n)

        # --- place reelle ---
        score = (0.35 * horse_level + 0.22 * driver_level + 0.20 * form_level
                 + 0.13 * disc_aff + 0.10 * hippo_aff + rng.normal(0, 1.2, n))
        order = np.argsort(-score)
        place = np.empty(n, dtype=int)
        place[order] = np.arange(1, n + 1)
        place_norm = (place - 1) / (n - 1)
        top5 = (place <= 5).astype(int)

        # --- features observables NOYAUTEES (le modele ne voit que ca) ---
        e = lambda: rng.normal(0, 1, n)  # noqa: E731
        horse_win_rate = _clamp01(1 / (1 + np.exp(-(2.2 * horse_level + 0.6 * e()))))
        horse_place_rate = _clamp01(1 / (1 + np.exp(-(1.9 * horse_level + 0.7 * e()))))
        horse_last_rank = _clip_int(1 + (n - 1) / (1 + np.exp(-(1.6 * form_level + 0.8 * e()))), 1, n)
        horse_avg_rank_6 = _clip_int(1 + (n - 1) / (1 + np.exp(-(1.4 * (0.6 * horse_level + 0.4 * form_level) + 0.9 * e()))), 1, n)
        horse_career_gains_log = np.log1p(np.exp(6.5 + 1.5 * horse_level + rng.normal(0, 0.8, n)))
        driver_win_rate = _clamp01(1 / (1 + np.exp(-(2.0 * driver_level + 0.7 * e()))))
        driver_place_rate = _clamp01(1 / (1 + np.exp(-(1.8 * driver_level + 0.8 * e()))))
        driver_recent_rank_avg = _clip_int(1 + (n - 1) / (1 + np.exp(-(1.5 * driver_level + 0.9 * e()))), 1, n)
        _lam_courses = np.maximum(0.5, 6 + 2 * (horse_level + driver_level))  # lambda Poisson > 0
        driver_horse_courses = np.maximum(1, rng.poisson(_lam_courses, n))
        combo_win_rate = _clamp01(1 / (1 + np.exp(-(2.0 * (0.5 * horse_level + 0.5 * driver_level) + 1.0 * e()))))
        discipline_win_rate = _clamp01(1 / (1 + np.exp(-(1.8 * disc_aff + 0.9 * e()))))
        hippodrome_win_rate = _clamp01(1 / (1 + np.exp(-(1.6 * hippo_aff + 1.0 * e()))))
        dist_pref = distance * np.exp(rng.normal(0, 0.18, n))
        dist_ratio = _clamp01(distance / dist_pref)
        rest_days = np.maximum(7, (30 + 40 * np.clip(1 - form_level / 3, 0, 1) + rng.normal(0, 12, n)).astype(int))
        music = _musique_from_form(form_level, rng)

        from scraper.engine import parse_musique

        m = np.array([parse_musique(x)["top5_ratio"] for x in music])
        odds_proxy = np.exp(1.2 - 0.5 * horse_level + rng.normal(0, 0.5, n))
        web_sentiment = np.clip(0.5 * form_level + rng.normal(0, 0.6, n), -1, 1)
        data_quality = _clamp01(0.55 + 0.25 * rng.normal(0, 1, n))

        for i in range(n):
            rows.append({
                "race_id": race_id,
                "num": i + 1,
                "horse_name": f"CHEVAL_{race_id}_{i}",
                "driver_name": f"DRIVER_{race_id}_{i}",
                "discipline": discipline,
                "hippodrome": hippodrome_id,
                "distance": distance,
                "place": int(place[i]),
                "place_norm": float(place_norm[i]),
                "top5": int(top5[i]),
                "horse_win_rate": float(horse_win_rate[i]),
                "horse_place_rate": float(horse_place_rate[i]),
                "horse_last_rank": float(horse_last_rank[i]),
                "horse_avg_rank_6": float(horse_avg_rank_6[i]),
                "horse_career_gains_log": float(horse_career_gains_log[i]),
                "horse_age": float(age[i]),
                "driver_win_rate": float(driver_win_rate[i]),
                "driver_place_rate": float(driver_place_rate[i]),
                "driver_recent_rank_avg": float(driver_recent_rank_avg[i]),
                "driver_horse_courses": float(driver_horse_courses[i]),
                "combo_win_rate": float(combo_win_rate[i]),
                "discipline_win_rate": float(discipline_win_rate[i]),
                "hippodrome_win_rate": float(hippodrome_win_rate[i]),
                "dist_ratio": float(dist_ratio[i]),
                "rest_days": float(rest_days[i]),
                "music_top5_ratio": float(m[i]),
                "odds_proxy": float(odds_proxy[i]),
                "n_partants": float(n),
                "distance_log": float(np.log(distance)),
                "web_sentiment": float(web_sentiment[i]),
                "data_quality": float(data_quality[i]),
            })

    df = pd.DataFrame(rows)
    df = df[["race_id", "num", "horse_name", "driver_name", "discipline", "hippodrome",
             "distance", "place", "place_norm", "top5"] + FEATURES]
    return df


def load_real_data(path: str) -> pd.DataFrame:
    """Point d'extension : charger un vrai historique CSV.

    Format attendu : une ligne par partant, colonnes identiques a celles
    generees par generate_synthetic_races (les colonnes de FEATURES doivent
    exister). Placez votre fichier au meme schema pour que train.py
    fonctionne sans modification.
    """
    df = pd.read_csv(path)
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV: {missing}")
    return df


def _clamp01(a: np.ndarray) -> np.ndarray:
    return np.clip(a, 0.0, 1.0)


def _clip_int(a: np.ndarray, lo: int, hi: int) -> np.ndarray:
    return np.clip(np.round(a), lo, hi)
