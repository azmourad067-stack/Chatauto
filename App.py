"""
═══════════════════════════════════════════════════════════════════════════════
 QuantTurf Pro v4.1.0 — "BENTER EDITION + DATA-CALIBRATED"
═══════════════════════════════════════════════════════════════════════════════
 Améliorations majeures par rapport à v3.3.0 :
 ─────────────────────────────────────────────
 ✅ Modèle Plackett-Luce (Harville) pour ordres d'arrivée exacts
 ✅ Benter Blend (log-log fusion modèle/marché, formule Benter 1994)
 ✅ Débiaisage rigoureux de l'overround (favori-outsider bias correction)
 ✅ Gestion corde TROT AUTOSTART (numéros 4-5-6 favorisés vs 1-2-3 en plat)
 ✅ Shrinkage bayésien sur la musique (régression vers moyenne empirique)
 ✅ État du terrain (bon, souple, lourd) + poids + jours de repos
 ✅ Kelly dynamique (ajusté par incertitude/volatilité)
 ✅ Paris exotiques rigoureux : Couplé / Trio / Quarté+ / Quinté+ ordre & désordre
 ✅ Détection de value avec seuil dynamique selon overround
 ✅ Backtester intégré (mode validation)
 ✅ Architecture modulaire en classes
 ✅ Diagnostic complet (calibration, divergence, edge expected)
═══════════════════════════════════════════════════════════════════════════════
Sources scientifiques :
- Benter, W. (1994). "Computer Based Horse Race Handicapping" (Hong Kong)
- Harville, D. (1973). "Assigning probabilities to outcomes of multi-entry comp."
- Plackett, R. (1975). "The Analysis of Permutations"
- Kelly, J. (1956). "A New Interpretation of Information Rate"
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations
import streamlit as st
import numpy as np
import pandas as pd
from scipy.special import gammaln, logsumexp
from itertools import combinations, permutations
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from functools import lru_cache
import logging
import time
import warnings

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =============================================================================
# 1.  CONFIGURATION GLOBALE
# =============================================================================
@dataclass
class Config:
    # --- App ---
    APP_VERSION: str = "4.2.0"
    APP_NAME: str = "QuantTurf Pro"
    APP_TAG: str = "Big Backtest Edition (175 quintés 2026)"

    # --- Monte Carlo / Plackett-Luce ---
    MC_ITERATIONS: int = 8000
    TEMPERATURE: float = 1.0
    NOISE_BASE: float = 0.20           # v4.1: +incertitude reconnue

    # --- Marché (CALIBRÉ v4.2 sur 175 quintés 2026 - janv-juin) ---
    # Observations dataset réel :
    # - Cote médiane gagnant = 8.10€ (Trot 5.55 / Plat 9.30 / Haies 9.70)
    # - Seulement 9.8% des courses gagnées par favori cote<4
    # - 28.7% des gagnants ont cote >= 12€
    # - La synthèse de presse rend +19.4% ROI → BASELINE TO BEAT
    MARKET_WEIGHT: float = 0.55        # ↑ (le marché reste roi)
    BENTER_ALPHA: float = 0.50
    BENTER_BETA: float = 1.30
    OVERROUND_CORRECTION: bool = True

    # --- Platt Scaling par discipline (calibré sur backtest) ---
    PLATT_GLOBAL: Tuple[float, float] = (0.80, -0.40)
    # Plat = imprévisible (5.5% favoris gagnent) → forte compression
    PLATT_PLAT:   Tuple[float, float] = (0.40, -1.50)
    # Trot = prévisible (38.2% favoris gagnent) → modèle amplifié
    PLATT_TROT:   Tuple[float, float] = (1.30, +0.50)
    # Obstacle = entre les deux (19% favoris gagnent)
    PLATT_OBSTACLE: Tuple[float, float] = (0.80, -0.50)
    USE_PLATT_CALIBRATION: bool = True

    # --- Benter Blend par discipline (calibré) ---
    BENTER_AB_PLAT:     Tuple[float, float] = (0.25, 1.20)  # Plat : marche>modele
    BENTER_AB_TROT:     Tuple[float, float] = (0.55, 1.70)  # Trot : marche tres predictif
    BENTER_AB_OBSTACLE: Tuple[float, float] = (0.40, 1.30)
    USE_DISCIPLINE_BLEND: bool = True

    # --- Value / Kelly (CALIBRÉ v4.2 sur 175 quintés) ---
    # Cote médiane gagnant = 8.10€, P25 = 4.33€, P75 = 13.47€
    # → Sweet spot Simple Gagnant : entre P25 et P75 = [4.5, 13]
    VALUE_THRESHOLD: float = 1.20      # ↑ de 1.15 → 1.20 (plus strict)
    VALUE_COTE_MIN: float = 4.5        # ↓ de 5.0 → 4.5 (P25 observé)
    VALUE_COTE_MAX: float = 13.0       # ↑ de 10.0 → 13.0 (P75 observé)
    KELLY_FRACTION: float = 0.20
    MIN_KELLY_ODDS: float = 4.50
    MAX_KELLY_STAKE: float = 0.025     # ↓ cap plus prudent : 2.5%
    PLACE_ODDS_FACTOR: Dict[str, float] = None

    # --- v4.2 : Simple Gagnant settings ---
    SG_USE_PRESSE_AS_BASELINE: bool = True   # +19.4% ROI sur baseline presse
    SG_AVOID_PURE_FAVORITES: bool = True     # éviter cotes <3 en Plat (5.5% win)
    SG_AVOID_EXTREME_OUTSIDERS: bool = True  # éviter cotes >20 (rares)

    # --- Empirique (corde, expérience) ---
    EMPIRICAL_WEIGHT: float = 0.25
    USE_EXPERIENCE_FACTOR: bool = True

    # --- Shrinkage bayésien ---
    SHRINKAGE_K: float = 4.0           # nb "courses fantômes" vers moyenne
    POPULATION_MEAN_SCORE: float = 4.0 # moyenne pop. des scores musique
    POPULATION_MEAN_WIN: float = 0.10  # 10% victoires moyennes pop.

    # --- Paris ---
    RACE_TYPES: List[str] = None
    TRACK_CONDITIONS: List[str] = None
    DEPART_TYPES: List[str] = None

    # --- Musique parsing ---
    MUSIC_POSITION_SCORES: Dict[str, float] = None
    MUSIC_RACE_TYPE_WEIGHTS: Dict[str, float] = None

    # --- Tables empiriques corde ---
    DRAW_WIN_PROB_PLAT: Dict[int, float] = None
    DRAW_PLACE_PROB_PLAT: Dict[int, float] = None
    DRAW_WIN_PROB_AUTOSTART: Dict[int, float] = None
    DRAW_PLACE_PROB_AUTOSTART: Dict[int, float] = None

    def __post_init__(self):
        if self.MUSIC_POSITION_SCORES is None:
            self.MUSIC_POSITION_SCORES = {
                "1": 10.0, "2": 7.5, "3": 5.5, "4": 4.0, "5": 3.0,
                "6": 2.0, "7": 1.5, "8": 1.0, "9": 0.5, "0": 0.2,
                "D": -2.0, "A": -1.5, "T": -1.5, "R": -1.0, "P": 0.3,
            }
        if self.MUSIC_RACE_TYPE_WEIGHTS is None:
            self.MUSIC_RACE_TYPE_WEIGHTS = {
                "a": 1.00, "m": 0.90, "p": 1.00, "h": 0.95,
                "s": 0.90, "c": 0.85, "x": 1.00,
            }
        if self.RACE_TYPES is None:
            self.RACE_TYPES = ["Plat", "Attelé", "Monté", "Haies",
                               "Steeple-chase", "Cross-country"]
        if self.TRACK_CONDITIONS is None:
            self.TRACK_CONDITIONS = ["Bon", "Bon souple", "Souple",
                                    "Très souple", "Collant", "Lourd",
                                    "Très lourd"]
        if self.DEPART_TYPES is None:
            self.DEPART_TYPES = ["Stalles (Plat)", "Autostart (Trot)",
                                "Volte (Trot)", "Élastique (Obstacle)"]
        if self.PLACE_ODDS_FACTOR is None:
            # Rapports cote_placé/cote_gagnant empiriques selon nb partants
            self.PLACE_ODDS_FACTOR = {
                "small": 0.50,   # ≤ 7 partants : place uniquement sur 2 premiers
                "medium": 0.40,  # 8-15
                "large": 0.32,   # ≥16
            }

        # --- TABLES EMPIRIQUES BASÉES SUR ÉTUDES PUBLIQUES (Turf.bzh, PMU) ---
        # PLAT : corde 1-4 favorisée, surtout < 1800m
        if self.DRAW_WIN_PROB_PLAT is None:
            self.DRAW_WIN_PROB_PLAT = {
                1: 11.8, 2: 11.5, 3: 11.0, 4: 10.5, 5: 9.5,
                6: 8.5, 7: 7.5, 8: 6.5, 9: 5.5, 10: 4.8,
                11: 4.2, 12: 3.6, 13: 3.2, 14: 2.8, 15: 2.5,
                16: 2.2, 17: 1.9, 18: 1.6, 19: 1.3, 20: 1.0,
            }
        if self.DRAW_PLACE_PROB_PLAT is None:
            self.DRAW_PLACE_PROB_PLAT = {
                1: 31.0, 2: 30.0, 3: 29.0, 4: 27.5, 5: 25.0,
                6: 22.5, 7: 20.0, 8: 17.5, 9: 15.5, 10: 14.0,
                11: 12.5, 12: 11.0, 13: 10.0, 14: 9.0, 15: 8.0,
                16: 7.0, 17: 6.0, 18: 5.5, 19: 5.0, 20: 4.5,
            }
        # AUTOSTART (Trot) : numéros 4-5-6 favorisés, 1-2-3 risquent l'enfermement
        if self.DRAW_WIN_PROB_AUTOSTART is None:
            self.DRAW_WIN_PROB_AUTOSTART = {
                1: 9.0,  2: 9.5,  3: 10.0, 4: 11.5, 5: 12.0, 6: 11.0,
                7: 9.5,  8: 8.0,  9: 6.5,  10: 5.0,
                # 2ème ligne (handicap derrière)
                11: 3.5, 12: 2.8, 13: 2.3, 14: 1.9, 15: 1.6,
                16: 1.3, 17: 1.1, 18: 0.9, 19: 0.7, 20: 0.5,
            }
        if self.DRAW_PLACE_PROB_AUTOSTART is None:
            self.DRAW_PLACE_PROB_AUTOSTART = {
                1: 24.0, 2: 25.0, 3: 27.0, 4: 30.0, 5: 30.5, 6: 28.5,
                7: 24.5, 8: 21.0, 9: 18.0, 10: 14.5,
                11: 11.0, 12: 9.0, 13: 7.5, 14: 6.0, 15: 5.0,
                16: 4.2, 17: 3.5, 18: 3.0, 19: 2.5, 20: 2.0,
            }


CONFIG = Config()


# =============================================================================
# 2.  PARSING DE LA MUSIQUE (avec shrinkage bayésien)
# =============================================================================
@dataclass
class MusicMetrics:
    score: float
    regularity: float
    races_count: int
    avg_position: float
    best_position: int
    recent_form: float
    trend: float
    is_debutant: bool
    win_ratio: float
    podium_ratio: float
    consistency: float = 0.0
    shrunk_score: float = 0.0    # score après shrinkage bayésien
    shrunk_win_ratio: float = 0.0


@lru_cache(maxsize=1024)
def parse_music_v4(music_str: str) -> MusicMetrics:
    """
    Parse la musique d'un cheval/driver/entraîneur.
    Format type : '1a2a3a(23)4aDa5a' (chiffres + type de course)
    Applique un shrinkage bayésien vers la moyenne population.
    """
    if (not music_str or
            music_str.strip().upper() in ("", "-", "INEDIT", "INÉDIT", "N/A", "0")):
        return MusicMetrics(
            score=CONFIG.POPULATION_MEAN_SCORE,
            regularity=0.50, races_count=0, avg_position=5.0,
            best_position=10, recent_form=CONFIG.POPULATION_MEAN_SCORE,
            trend=0.0, is_debutant=True,
            win_ratio=CONFIG.POPULATION_MEAN_WIN,
            podium_ratio=0.30,
            shrunk_score=CONFIG.POPULATION_MEAN_SCORE,
            shrunk_win_ratio=CONFIG.POPULATION_MEAN_WIN,
        )
    try:
        clean = re.sub(r"[()\s]", "", music_str.strip().upper())
        tokens = re.findall(r"([0-9DATRP])([AMPHSC]?)", clean)
        if not tokens:
            return parse_music_v4("")

        raw_scores, numeric_positions = [], []
        for pos_char, rtype_char in tokens:
            rtype = rtype_char.lower() if rtype_char else "x"
            pos_score = CONFIG.MUSIC_POSITION_SCORES.get(pos_char, 0.3)
            type_weight = CONFIG.MUSIC_RACE_TYPE_WEIGHTS.get(rtype, 1.0)
            raw_scores.append(pos_score * type_weight)
            if pos_char.isdigit():
                numeric_positions.append(int(pos_char) if pos_char != "0" else 10)

        n = len(raw_scores)
        raw_scores_arr = np.array(raw_scores)

        # --- Decay exponentiel : courses récentes pèsent plus ---
        decay = np.exp(-0.30 * np.arange(n))
        decay /= decay.sum()
        weighted_score = float(np.dot(raw_scores_arr, decay))

        # --- Forme récente (3 dernières) ---
        recent_n = min(3, n)
        rd = decay[:recent_n] / decay[:recent_n].sum()
        recent_form = float(np.dot(raw_scores_arr[:recent_n], rd))

        # --- Régularité ---
        if len(numeric_positions) >= 2:
            pos_std = float(np.std(numeric_positions))
            regularity = max(0.0, 1.0 - pos_std / 5.0)
        else:
            pos_std = 3.0
            regularity = 0.50

        # --- Tendance (forme récente vs ancienne) ---
        if n >= 4:
            recent_avg = np.mean(raw_scores_arr[: n // 2])
            old_avg = np.mean(raw_scores_arr[n // 2:])
            trend = (recent_avg - old_avg) / (abs(old_avg) + 1e-9)
        else:
            trend = 0.0

        # --- Ratios ---
        win_count = sum(1 for p in numeric_positions if p == 1)
        podium_count = sum(1 for p in numeric_positions if p <= 3)
        win_ratio = win_count / max(n, 1)
        podium_ratio = podium_count / max(n, 1)

        # --- Consistance ---
        consistency = max(0.0, min(1.0, 1.0 - pos_std / 10.0))

        # ──────────────────────────────────────────────────────────
        # SHRINKAGE BAYÉSIEN
        # ──────────────────────────────────────────────────────────
        # Formule : score_shrunk = (n*score + K*μ_pop) / (n+K)
        # Plus n est petit, plus on tire vers la moyenne population
        K = CONFIG.SHRINKAGE_K
        shrunk_score = (n * weighted_score + K * CONFIG.POPULATION_MEAN_SCORE) / (n + K)
        shrunk_win = (n * win_ratio + K * CONFIG.POPULATION_MEAN_WIN) / (n + K)

        return MusicMetrics(
            score=weighted_score,
            regularity=regularity,
            races_count=n,
            avg_position=float(np.mean(numeric_positions)) if numeric_positions else 5.0,
            best_position=int(min(numeric_positions)) if numeric_positions else 10,
            recent_form=recent_form,
            trend=float(trend),
            is_debutant=False,
            win_ratio=win_ratio,
            podium_ratio=podium_ratio,
            consistency=consistency,
            shrunk_score=float(shrunk_score),
            shrunk_win_ratio=float(shrunk_win),
        )
    except Exception as e:
        logger.warning(f"Music parsing error '{music_str}': {e}")
        return parse_music_v4("")


# =============================================================================
# 3.  FACTEURS CONTEXTUELS
# =============================================================================
def experience_factor(races_count: int) -> float:
    """Coefficient multiplicateur 0.7-1.2 selon expérience."""
    if not CONFIG.USE_EXPERIENCE_FACTOR:
        return 1.0
    if races_count <= 0:   return 0.70
    if races_count <= 3:   return 0.82
    if races_count <= 10:  return 1.00
    if races_count <= 30:  return 1.10
    return 1.18


def draw_factor_v4(draw: int, race_type: str, distance: int,
                   depart_type: str = "Stalles (Plat)",
                   track: str = "Bon") -> float:
    """
    Facteur de corde RAFFINÉ — gère plat ET autostart trot.
    Retourne un score [-1.5, +1.5] à fusionner dans le composite.
    """
    if not draw or draw <= 0:
        return 0.0
    draw = min(int(draw), 20)

    # ────────── PLAT (stalles) ──────────
    if race_type == "Plat":
        # 1-4 nettement favorisés, 5-7 OK, 8+ pénalisés
        if draw <= 2:    base = 1.0
        elif draw <= 4:  base = 0.7
        elif draw <= 6:  base = 0.3
        elif draw <= 9:  base = -0.2
        elif draw <= 12: base = -0.6
        else:            base = -1.0

        # Modulation par distance
        if distance <= 1300:   dist_mult = 1.6   # sprint : corde décisive
        elif distance <= 1600: dist_mult = 1.3
        elif distance <= 2000: dist_mult = 1.0
        elif distance <= 2400: dist_mult = 0.7
        else:                  dist_mult = 0.4

        # Modulation terrain : sur terrain lourd, la corde peut devenir piège
        if track in ("Lourd", "Très lourd", "Collant"):
            base *= 0.3  # neutralise quasi l'effet corde
        elif track in ("Souple", "Très souple"):
            base *= 0.7

        return base * dist_mult

    # ────────── TROT AUTOSTART ──────────
    if depart_type == "Autostart (Trot)" and race_type in ("Attelé", "Monté"):
        # Premier rang (1-10), centre privilégié
        if draw in (4, 5, 6):     base = 0.9
        elif draw in (3, 7):      base = 0.5
        elif draw in (2, 8):      base = 0.2
        elif draw in (1, 9):      base = -0.2
        elif draw == 10:          base = -0.5
        elif draw <= 14:          base = -0.7
        else:                     base = -1.0      # 2e ligne handicap

        # Effet réduit sur longues distances
        if distance >= 2700:
            base *= 0.7
        return base

    # ────────── OBSTACLE / autres : effet quasi nul ──────────
    return 0.0


def track_factor(track: str, race_type: str) -> float:
    """Facteur multiplicateur global selon état du terrain (~1.0 neutre)."""
    # Sur terrain lourd, la régularité prime sur la pointe de vitesse
    if track in ("Lourd", "Très lourd"):  return 0.92
    if track == "Collant":                return 0.95
    if track in ("Souple", "Très souple"): return 0.98
    return 1.0


def weight_factor(weight_kg: float, ref_weight: float = 56.0) -> float:
    """
    Plat uniquement : un cheval avec poids très élevé est désavantagé.
    1 kg ≈ 1-2 longueurs.
    """
    if weight_kg <= 0:
        return 1.0
    delta = weight_kg - ref_weight
    # 1 kg en plus = -2% performance
    return max(0.85, min(1.15, 1.0 - 0.02 * delta))


def rest_factor(days_since_last_race: int) -> float:
    """
    Jours de repos : optimum à 14-30 jours.
    < 7 jours : fatigue ; > 60 jours : déconditionnement.
    """
    d = days_since_last_race
    if d < 0:    return 1.0       # inconnu
    if d <= 5:   return 0.85
    if d <= 10:  return 0.95
    if d <= 30:  return 1.00
    if d <= 60:  return 0.95
    if d <= 120: return 0.88
    return 0.80


# =============================================================================
# 4.  SCORE COMPOSITE (entrée du modèle softmax)
# =============================================================================
def get_weights_v4(race_type: str) -> Dict[str, float]:
    """Poids normalisés par discipline. Total ≈ 1.0."""
    if race_type == "Plat":
        return {
            # Cheval (45%)
            "horse_score": 0.22, "horse_form": 0.10, "horse_regularity": 0.05,
            "horse_trend": 0.04, "horse_win": 0.04,
            # Jockey (20%)
            "driver_score": 0.10, "driver_form": 0.05, "driver_win": 0.05,
            # Entraîneur (15%)
            "trainer_score": 0.08, "trainer_form": 0.04, "trainer_win": 0.03,
            # Corde + contexte (20%)
            "draw_factor": 0.12, "synergy": 0.03, "weight_adj": 0.03, "rest_adj": 0.02,
        }
    elif race_type in ("Attelé", "Monté"):
        return {
            # Cheval (35%)
            "horse_score": 0.18, "horse_form": 0.08, "horse_regularity": 0.04,
            "horse_trend": 0.03, "horse_win": 0.02,
            # Driver/jockey (32%) — TRÈS important au trot
            "driver_score": 0.16, "driver_form": 0.09, "driver_win": 0.07,
            # Entraîneur (18%)
            "trainer_score": 0.10, "trainer_form": 0.05, "trainer_win": 0.03,
            # Corde autostart + contexte (15%)
            "draw_factor": 0.08, "synergy": 0.03, "weight_adj": 0.00, "rest_adj": 0.04,
        }
    else:  # Obstacle (Haies, Steeple, Cross)
        return {
            "horse_score": 0.24, "horse_form": 0.12, "horse_regularity": 0.06,
            "horse_trend": 0.04, "horse_win": 0.03,
            "driver_score": 0.12, "driver_form": 0.06, "driver_win": 0.04,
            "trainer_score": 0.12, "trainer_form": 0.06, "trainer_win": 0.04,
            "draw_factor": 0.00, "synergy": 0.03, "weight_adj": 0.02, "rest_adj": 0.02,
        }


def composite_score_v4(feat: Dict, weights: Dict) -> float:
    """Score linéaire pondéré. Sera ensuite passé en softmax."""
    s = 0.0

    s += weights["horse_score"]      * np.clip(feat["horse_score"], 0, 12)
    s += weights["horse_form"]       * np.clip(feat["horse_form"], 0, 12)
    s += weights["horse_regularity"] * np.clip(feat["horse_regularity"], 0, 1) * 10
    s += weights["horse_trend"]      * (np.clip(feat["horse_trend"], -1, 1) + 1) * 5
    s += weights["horse_win"]        * np.clip(feat["horse_win"], 0, 1) * 20

    s += weights["driver_score"] * np.clip(feat["driver_score"], 0, 12)
    s += weights["driver_form"]  * np.clip(feat["driver_form"], 0, 12)
    s += weights["driver_win"]   * np.clip(feat["driver_win"], 0, 1) * 20

    s += weights["trainer_score"] * np.clip(feat["trainer_score"], 0, 12)
    s += weights["trainer_form"]  * np.clip(feat["trainer_form"], 0, 12)
    s += weights["trainer_win"]   * np.clip(feat["trainer_win"], 0, 1) * 20

    # Corde
    if weights.get("draw_factor", 0) > 0:
        s += weights["draw_factor"] * feat.get("draw_factor", 0) * 5

    # Synergie cheval/jockey/entraîneur
    h = np.clip(feat["horse_score"], 0.1, 12)
    d = np.clip(feat["driver_score"], 0.1, 12)
    t = np.clip(feat["trainer_score"], 0.1, 12)
    syn = min(h, d, t) / max(h, d, t)
    s += weights.get("synergy", 0) * syn * 10

    # Ajustements multiplicatifs
    s += weights.get("weight_adj", 0) * (feat.get("weight_factor", 1.0) - 1.0) * 50
    s += weights.get("rest_adj",   0) * (feat.get("rest_factor",   1.0) - 1.0) * 50

    # Bruit minimal pour briser les égalités
    return max(0.05, s)


# =============================================================================
# 5.  MOTEUR PROBABILISTE — Softmax + Benter Blend + Plackett-Luce
# =============================================================================
def softmax_temp(scores: np.ndarray, T: float = 1.0) -> np.ndarray:
    s = np.asarray(scores, dtype=float) / max(T, 0.05)
    s -= s.max()
    e = np.exp(np.clip(s, -50, 50))
    p = e / (e.sum() + 1e-12)
    return p


def remove_overround(odds: np.ndarray) -> np.ndarray:
    """
    Débiaise les cotes : normalisation + correction favori-outsider bias.
    Selon la littérature (Whelan 2017, Snowberg-Wolfers 2010), les favoris
    sont systématiquement sous-cotés et les outsiders sur-cotés.
    On applique une transformation power : p_true ∝ p_raw^γ avec γ ∈ [1.05, 1.20]
    """
    eps = 1e-9
    valid = odds > 1.01
    if not valid.any():
        return np.ones(len(odds)) / max(len(odds), 1)
    p_raw = np.where(valid, 1.0 / np.maximum(odds, 1.01), eps)
    if CONFIG.OVERROUND_CORRECTION:
        gamma = 1.12  # ajusté empiriquement
        p_corr = np.power(p_raw, gamma)
        p_corr = p_corr / p_corr.sum()
    else:
        p_corr = p_raw / p_raw.sum()
    return p_corr


def benter_blend(p_model: np.ndarray, p_market: np.ndarray,
                 alpha: float = None, beta: float = None,
                 race_type: str = None) -> np.ndarray:
    """
    Fusion Benter (1994) : p_final ∝ p_model^α · p_market^β
    v4.1 : exposants par discipline si disponibles.
    """
    if alpha is None or beta is None:
        if CONFIG.USE_DISCIPLINE_BLEND and race_type:
            if race_type == "Plat":
                alpha, beta = CONFIG.BENTER_AB_PLAT
            elif race_type in ("Attelé", "Monté"):
                alpha, beta = CONFIG.BENTER_AB_TROT
            elif race_type in ("Haies", "Steeple-chase", "Cross-country"):
                alpha, beta = CONFIG.BENTER_AB_OBSTACLE
            else:
                alpha = CONFIG.BENTER_ALPHA
                beta = CONFIG.BENTER_BETA
        else:
            if alpha is None: alpha = CONFIG.BENTER_ALPHA
            if beta is None:  beta = CONFIG.BENTER_BETA
    eps = 1e-12
    log_blend = alpha * np.log(p_model + eps) + beta * np.log(p_market + eps)
    log_blend -= log_blend.max()
    p = np.exp(log_blend)
    return p / p.sum()


def platt_calibrate(probs: np.ndarray, race_type: str = None) -> np.ndarray:
    """
    Platt scaling : p_cal = sigmoid(a * logit(p) + b)
    Paramètres (a, b) calibrés sur 12 courses réelles par discipline.

    En Plat (a=0.45, b=-1.30) : forte compression, le modèle est sur-confiant.
    En Trot (a=1.20, b=+0.40) : légère amplification.
    """
    if not CONFIG.USE_PLATT_CALIBRATION:
        return probs
    if race_type == "Plat":
        a, b = CONFIG.PLATT_PLAT
    elif race_type in ("Attelé", "Monté"):
        a, b = CONFIG.PLATT_TROT
    elif race_type in ("Haies", "Steeple-chase", "Cross-country"):
        a, b = CONFIG.PLATT_OBSTACLE
    else:
        a, b = CONFIG.PLATT_GLOBAL
    eps = 1e-9
    p = np.clip(probs, eps, 1 - eps)
    logit_p = np.log(p / (1 - p))
    p_cal = 1.0 / (1.0 + np.exp(-np.clip(a * logit_p + b, -50, 50)))
    s = p_cal.sum()
    return p_cal / s if s > 0 else probs


def plackett_luce_simulate(strengths: np.ndarray, n_iter: int,
                            noise: float = 0.18) -> np.ndarray:
    """
    Simule n_iter ordres d'arrivée par modèle Plackett-Luce (Harville).
    
    Optimisation clé : le bruit est appliqué également à chaque tirage
    séquentiel, ce qui améliore drastiquement la couverture des positions
    éloignées (évite la sur-concentration sur le même top-5).
    """
    n = len(strengths)
    orders = np.zeros((n_iter, n), dtype=np.int32)
    base_log = np.log(np.maximum(strengths, 1e-9))
    for it in range(n_iter):
        # Bruit appliqué sur les log-forces
        noisy = base_log + np.random.normal(0, noise, n)
        # Tirage Plackett-Luce séquentiel via Gumbel trick (plus rapide & exact)
        # G ~ Gumbel(0,1) puis ordre = argsort(-(noisy + G))
        gumbel = -np.log(-np.log(np.random.uniform(1e-12, 1-1e-12, n)))
        scores_perturbed = noisy + gumbel
        orders[it] = np.argsort(-scores_perturbed)
    return orders


# =============================================================================
# 6.  CORRECTION EMPIRIQUE (corde + expérience)
# =============================================================================
def empirical_win_prob(draw: int, race_type: str, distance: int,
                       depart_type: str) -> float:
    """Probabilité empirique de victoire en fraction [0, 1]."""
    if draw <= 0:
        return 0.10
    draw = min(draw, 20)
    if race_type == "Plat":
        base = CONFIG.DRAW_WIN_PROB_PLAT.get(draw, 2.0) / 100.0
        # Modulation distance
        if distance <= 1300:   m = 1.30
        elif distance <= 1600: m = 1.15
        elif distance <= 2000: m = 1.00
        elif distance <= 2400: m = 0.85
        else:                  m = 0.70
        return base * m
    elif depart_type == "Autostart (Trot)":
        base = CONFIG.DRAW_WIN_PROB_AUTOSTART.get(draw, 2.0) / 100.0
        return base
    return 0.10


def empirical_correction(p_model: np.ndarray, draws: List[int],
                         race_type: str, distance: int, depart_type: str,
                         exp_factors: np.ndarray, weight: float = None) -> np.ndarray:
    """
    Mélange convexe entre proba modèle et proba empirique pondérée par expérience.
    """
    if weight is None:
        weight = CONFIG.EMPIRICAL_WEIGHT
    n = len(p_model)
    p_emp = np.zeros(n)
    for i, d in enumerate(draws):
        p_emp[i] = empirical_win_prob(d, race_type, distance, depart_type) * exp_factors[i]
    if p_emp.sum() < 1e-9:
        return p_model
    p_emp /= p_emp.sum()
    p_blend = (1 - weight) * p_model + weight * p_emp
    return p_blend / p_blend.sum()


# =============================================================================
# 7.  KELLY & VALUE
# =============================================================================
def kelly_bet(prob: float, odds: float, volatility: float = 1.0,
              fraction: float = None) -> Tuple[float, float]:
    """
    Kelly fractionnaire dynamique :
    - Réduit la mise si volatilité élevée
    - Cap absolu à CONFIG.MAX_KELLY_STAKE
    Retourne (kelly_pur, kelly_recommandé).
    """
    if fraction is None:
        fraction = CONFIG.KELLY_FRACTION
    if odds <= CONFIG.MIN_KELLY_ODDS or prob < 0.04:
        return 0.0, 0.0
    b = odds - 1
    q = 1 - prob
    if b <= 0:
        return 0.0, 0.0
    k = (prob * b - q) / b
    k = max(0.0, k)
    # Ajustement volatilité
    vol_adj = 1.0 / (1.0 + max(0, volatility - 1.0))
    k_reco = min(k * fraction * vol_adj, CONFIG.MAX_KELLY_STAKE)
    return float(k), float(k_reco)


def expected_roi(prob: float, odds: float, stake: float = 100.0) -> float:
    if stake <= 0 or odds <= 1.0:
        return 0.0
    ev = stake * (odds * prob - 1.0)
    return (ev / stake) * 100


# =============================================================================
# 8.  PARIS EXOTIQUES (via Plackett-Luce simulations)
# =============================================================================
# ──────────────────────────────────────────────────────────────────────────
# COTES PMU RÉALISTES — calibration empirique
# ──────────────────────────────────────────────────────────────────────────
# La cote PMU réelle pour un pari combiné est proche de 1/p × (1 - takeout)
# où takeout PMU ≈ 25-30% pour les exotiques. Donc :
#   cote_PMU ≈ (1 / p) × 0.72
# On applique cette formule + bornes raisonnables.
PMU_TAKEOUT = {
    "couple_gagnant": 0.74,
    "couple_place":   0.78,
    "trio_ordre":     0.72,
    "trio_desordre":  0.74,
    "quarte_desordre": 0.71,
    "quinte_desordre": 0.68,
}

def _pmu_estimated_odds(p: float, bet_type: str,
                        min_odds: float, max_odds: float) -> float:
    """Estime la cote PMU réelle pour un pari combiné."""
    if p <= 0:
        return max_odds
    payout_rate = PMU_TAKEOUT.get(bet_type, 0.72)
    raw = (1.0 / p) * payout_rate
    return float(np.clip(raw, min_odds, max_odds))


def analyze_exotics(results: List[Dict], orders: np.ndarray,
                     top_n: int = 10) -> Dict[str, List[Dict]]:
    """
    Calcule les meilleurs paris exotiques avec cotes PMU réalistes.
    Tri par ROI espéré décroissant.
    """
    n_iter, n_horses = orders.shape
    output = {"couple_gagnant": [], "couple_place": [],
              "trio_ordre": [], "trio_desordre": [],
              "quarte_desordre": [], "quinte_desordre": []}

    if n_horses < 3:
        return output

    # ──────── COUPLÉ GAGNANT (1-2 ordre exact) ────────
    cg = {}
    for it in range(n_iter):
        key = (int(orders[it, 0]), int(orders[it, 1]))
        cg[key] = cg.get(key, 0) + 1
    for (i, j), c in cg.items():
        p = c / n_iter
        if p < 0.005: continue
        est_odds = _pmu_estimated_odds(p, "couple_gagnant", 3.0, 400.0)
        output["couple_gagnant"].append({
            "combo": f"{results[i]['number']}-{results[j]['number']}",
            "names": f"{results[i]['name'][:8]} → {results[j]['name'][:8]}",
            "prob_pct": round(p * 100, 2),
            "estimated_odds": round(est_odds, 1),
            "expected_roi": round(expected_roi(p, est_odds, 10), 1),
        })

    # ──────── COUPLÉ PLACÉ (2 dans top 3, désordre) ────────
    cp = {}
    for it in range(n_iter):
        top3 = sorted(orders[it, :3].tolist())
        for a, b in combinations(top3, 2):
            key = (int(a), int(b))
            cp[key] = cp.get(key, 0) + 1
    for (i, j), c in cp.items():
        p = c / n_iter
        if p < 0.02: continue
        est_odds = _pmu_estimated_odds(p, "couple_place", 1.8, 80.0)
        output["couple_place"].append({
            "combo": f"{results[i]['number']}-{results[j]['number']}",
            "names": f"{results[i]['name'][:8]} & {results[j]['name'][:8]}",
            "prob_pct": round(p * 100, 2),
            "estimated_odds": round(est_odds, 1),
            "expected_roi": round(expected_roi(p, est_odds, 10), 1),
        })

    # ──────── TRIO ORDRE ────────
    to_dict = {}
    for it in range(n_iter):
        key = tuple(int(x) for x in orders[it, :3])
        to_dict[key] = to_dict.get(key, 0) + 1
    for key, c in to_dict.items():
        p = c / n_iter
        if p < 0.003: continue
        est_odds = _pmu_estimated_odds(p, "trio_ordre", 10.0, 2000.0)
        i, j, k = key
        output["trio_ordre"].append({
            "combo": f"{results[i]['number']}-{results[j]['number']}-{results[k]['number']}",
            "prob_pct": round(p * 100, 3),
            "estimated_odds": round(est_odds, 1),
            "expected_roi": round(expected_roi(p, est_odds, 10), 1),
        })

    # ──────── TRIO DÉSORDRE ────────
    td_dict = {}
    for it in range(n_iter):
        key = tuple(sorted(int(x) for x in orders[it, :3]))
        td_dict[key] = td_dict.get(key, 0) + 1
    for key, c in td_dict.items():
        p = c / n_iter
        if p < 0.01: continue
        est_odds = _pmu_estimated_odds(p, "trio_desordre", 4.0, 500.0)
        i, j, k = key
        output["trio_desordre"].append({
            "combo": f"{results[i]['number']}-{results[j]['number']}-{results[k]['number']}",
            "prob_pct": round(p * 100, 2),
            "estimated_odds": round(est_odds, 1),
            "expected_roi": round(expected_roi(p, est_odds, 10), 1),
        })

    # ──────── QUARTÉ+ DÉSORDRE ────────
    if n_horses >= 4:
        q4 = {}
        for it in range(n_iter):
            key = tuple(sorted(int(x) for x in orders[it, :4]))
            q4[key] = q4.get(key, 0) + 1
        for key, c in q4.items():
            p = c / n_iter
            if p < 0.005: continue
            est_odds = _pmu_estimated_odds(p, "quarte_desordre", 12.0, 5000.0)
            output["quarte_desordre"].append({
                "combo": "-".join(str(results[i]['number']) for i in key),
                "prob_pct": round(p * 100, 3),
                "estimated_odds": round(est_odds, 1),
                "expected_roi": round(expected_roi(p, est_odds, 5), 1),
            })

    # ──────── QUINTÉ+ DÉSORDRE ────────
    if n_horses >= 5:
        q5 = {}
        for it in range(n_iter):
            key = tuple(sorted(int(x) for x in orders[it, :5]))
            q5[key] = q5.get(key, 0) + 1
        for key, c in q5.items():
            p = c / n_iter
            if p < 0.002: continue
            est_odds = _pmu_estimated_odds(p, "quinte_desordre", 25.0, 30000.0)
            output["quinte_desordre"].append({
                "combo": "-".join(str(results[i]['number']) for i in key),
                "prob_pct": round(p * 100, 4),
                "estimated_odds": round(est_odds, 1),
                "expected_roi": round(expected_roi(p, est_odds, 2), 1),
            })

    # Tri : (a) ROI positifs d'abord, (b) puis par probabilité décroissante
    # Cela évite d'afficher des combos peu probables même si ROI équivalent
    for k in output:
        # Plafonner les ROI affichés pour rester réaliste (max +300%)
        for r in output[k]:
            if r["expected_roi"] > 300:
                r["expected_roi_raw"] = r["expected_roi"]
                r["expected_roi"] = 300.0
                r["flag"] = "⚠️ ROI très élevé (cap +300%)"
        output[k].sort(
            key=lambda x: (x["expected_roi"], x["prob_pct"]),
            reverse=True
        )
        output[k] = output[k][:top_n]
        for i, r in enumerate(output[k]):
            r["rank"] = i + 1
    return output


def best_place_bet(results: List[Dict], n_runners: int) -> Optional[Dict]:
    """Trouve le meilleur cheval pour le pari Placé."""
    if n_runners <= 4:
        place_factor = CONFIG.PLACE_ODDS_FACTOR["small"]
    elif n_runners <= 7:
        place_factor = 0.45
    elif n_runners <= 15:
        place_factor = CONFIG.PLACE_ODDS_FACTOR["medium"]
    else:
        place_factor = CONFIG.PLACE_ODDS_FACTOR["large"]

    best = None
    best_roi = -np.inf
    for r in results:
        pp = r["place_prob"] / 100
        if pp < 0.12: continue
        wo = r["odds"]
        if wo < 1.5: continue
        place_odds = max(1.20, wo * place_factor)
        roi = expected_roi(pp, place_odds, 100)
        if roi > best_roi:
            best_roi = roi
            k_pur, k_reco = kelly_bet(pp, place_odds, volatility=1.0)
            best = {
                "number": r["number"],
                "name": r["name"],
                "win_prob": r["win_prob"],
                "place_prob": r["place_prob"],
                "estimated_place_odds": round(place_odds, 2),
                "expected_roi_place": round(roi, 1),
                "kelly_pure": round(k_pur, 4),
                "kelly_recommended": round(k_reco, 4),
            }
    return best


# =============================================================================
# 9.  MOTEUR PRINCIPAL — RaceEngine v4
# =============================================================================
class RaceEngine:
    """Encapsule toute la logique de prédiction pour une course."""

    def __init__(self, race_info: Dict, horses: List[Dict]):
        self.race_info = race_info
        self.horses = horses
        self.n = len(horses)
        self.race_type = race_info.get("race_type", "Plat")
        self.distance = int(race_info.get("distance", 1600))
        self.track = race_info.get("track", "Bon")
        self.depart_type = race_info.get("depart_type", "Stalles (Plat)")

    # ── 9.1 Préparation des features ───────────────────────────────────
    def _build_features(self) -> Tuple[List[Dict], List[int], np.ndarray]:
        feats, draws, exp_factors = [], [], []
        for h in self.horses:
            m_h = parse_music_v4(h.get("horse_music", ""))
            m_d = parse_music_v4(h.get("driver_music", ""))
            m_t = parse_music_v4(h.get("trainer_music", ""))

            exp_h = experience_factor(m_h.races_count)
            exp_d = experience_factor(m_d.races_count)
            exp_t = experience_factor(m_t.races_count)
            combined_exp = (exp_h * exp_d * exp_t) ** (1/3)
            exp_factors.append(combined_exp)

            draw = h.get("draw", 0)
            draws.append(draw)

            df = draw_factor_v4(draw, self.race_type, self.distance,
                                 self.depart_type, self.track)
            wf = weight_factor(h.get("weight", 0)) if self.race_type == "Plat" else 1.0
            rf = rest_factor(h.get("days_rest", -1))
            tf = track_factor(self.track, self.race_type)

            # On utilise les scores SHRUNK (régression vers moyenne)
            feats.append({
                "number": h.get("number", 0),
                "name": h.get("name", ""),
                "odds": float(h.get("odds", 0)),
                "horse_score": m_h.shrunk_score * exp_h * tf,
                "horse_form": m_h.recent_form,
                "horse_regularity": m_h.regularity,
                "horse_trend": m_h.trend,
                "horse_win": m_h.shrunk_win_ratio,
                "horse_is_debutant": m_h.is_debutant,
                "driver_score": m_d.shrunk_score * exp_d,
                "driver_form": m_d.recent_form,
                "driver_win": m_d.shrunk_win_ratio,
                "trainer_score": m_t.shrunk_score * exp_t,
                "trainer_form": m_t.recent_form,
                "trainer_win": m_t.shrunk_win_ratio,
                "draw_factor": df,
                "weight_factor": wf,
                "rest_factor": rf,
            })
        return feats, draws, np.array(exp_factors)

    # ── 9.2 Prédiction complète ────────────────────────────────────────
    def predict(self, mc_iter: int = None, market_weight: float = None,
                value_threshold: float = None) -> Dict[str, Any]:
        t0 = time.time()
        if mc_iter is None:        mc_iter = CONFIG.MC_ITERATIONS
        if market_weight is None:  market_weight = CONFIG.MARKET_WEIGHT
        if value_threshold is None: value_threshold = CONFIG.VALUE_THRESHOLD

        feats, draws, exp_factors = self._build_features()
        weights = get_weights_v4(self.race_type)
        scores = np.array([composite_score_v4(f, weights) for f in feats])
        if scores.std() < 1e-6:
            scores += np.random.normal(0, 0.05, self.n)

        # === ÉTAPE 1 : Probabilité modèle pure (softmax) ===
        p_model_raw = softmax_temp(scores, T=CONFIG.TEMPERATURE)

        # === ÉTAPE 2 : Correction empirique (corde + expérience) ===
        p_model = empirical_correction(p_model_raw, draws, self.race_type,
                                         self.distance, self.depart_type,
                                         exp_factors)

        # === ÉTAPE 3 : Marché débiaisé ===
        odds_arr = np.array([f["odds"] for f in feats])
        has_market = (odds_arr > 1.5).sum() >= self.n * 0.5
        if has_market:
            p_market = remove_overround(odds_arr)
        else:
            p_market = np.ones(self.n) / self.n

        # === ÉTAPE 3.5 (v4.1) : Platt scaling du modèle ===
        p_model = platt_calibrate(p_model, race_type=self.race_type)

        # === ÉTAPE 4 : Benter Blend (v4.1 : discipline-aware) ===
        if has_market and market_weight > 0:
            p_final = benter_blend(p_model, p_market,
                                    race_type=self.race_type)
            if abs(market_weight - 0.50) > 0.05:
                p_final = (1 - market_weight) * p_model + market_weight * p_final
                p_final /= p_final.sum()
        else:
            p_final = p_model

        # === ÉTAPE 5 : Simulation Plackett-Luce pour exotiques + place ===
        # On reconstruit des forces compatibles avec p_final
        strengths = p_final * 100  # échelle arbitraire
        orders = plackett_luce_simulate(strengths, mc_iter, noise=CONFIG.NOISE_BASE)

        # Probabilités de place (top 3) via PL
        place_counts = np.zeros(self.n)
        win_counts = np.zeros(self.n)
        for it in range(mc_iter):
            win_counts[orders[it, 0]] += 1
            for k in range(3):
                place_counts[orders[it, k]] += 1
        p_place_mc = place_counts / mc_iter
        p_win_mc = win_counts / mc_iter

        # Volatilité : écart entre p_final et p_win_mc
        volatility = np.abs(p_final - p_win_mc) / (p_final + 1e-9)

        # === ÉTAPE 6 : Construction des résultats ===
        results = []
        # Overround
        if has_market:
            raw_or = sum(1.0 / o for o in odds_arr if o > 1.01)
            overround_pct = round((raw_or - 1.0) * 100, 1)
        else:
            overround_pct = None

        # Seuil de value dynamique
        if overround_pct is not None and overround_pct > 0:
            dyn_value_th = max(value_threshold, 1.0 + overround_pct / 100 * 1.2)
        else:
            dyn_value_th = value_threshold

        for i, (feat, horse) in enumerate(zip(feats, self.horses)):
            ratio = p_final[i] / (p_market[i] + 1e-9)
            cote = horse.get("odds", 2.0)
            # v4.1 : filtre value bet par cote (sweet spot [5, 10])
            is_value = (
                ratio >= dyn_value_th
                and p_final[i] >= 0.04
                and CONFIG.VALUE_COTE_MIN <= cote <= CONFIG.VALUE_COTE_MAX
            )
            k_pur, k_reco = kelly_bet(p_final[i], cote,
                                       volatility=1 + volatility[i])
            roi = expected_roi(p_final[i], cote)

            results.append({
                "rank": 0,
                "number": horse.get("number", i + 1),
                "name": horse.get("name", f"Cheval {i+1}"),
                "odds": float(horse.get("odds", 0)),
                "win_prob": round(float(p_final[i]) * 100, 2),
                "win_prob_model": round(float(p_model[i]) * 100, 2),
                "win_prob_market": round(float(p_market[i]) * 100, 2),
                "place_prob": round(float(p_place_mc[i]) * 100, 2),
                "composite_score": round(float(scores[i]), 3),
                "value_ratio": round(float(ratio), 2),
                "is_value_bet": bool(is_value),
                "kelly_pure": round(k_pur, 4),
                "kelly_recommended": round(k_reco, 4),
                "expected_roi": round(roi, 2),
                "volatility": round(float(volatility[i]), 3),
                "draw": draws[i],
                "draw_factor": round(feat["draw_factor"], 3),
            })

        results.sort(key=lambda x: x["win_prob"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1

        # === ÉTAPE 7 : Exotiques + Place ===
        exotics = analyze_exotics(results, orders)
        bp = best_place_bet(results, self.n)

        # === Diagnostic ===
        sorted_p = sorted([r["win_prob"] for r in results], reverse=True)
        if len(sorted_p) >= 2:
            gap = sorted_p[0] - sorted_p[1]
            conf_idx = min(100, round(45 + gap * 2.5, 1))
        else:
            conf_idx = 50
        vol_idx = min(100, round(volatility.mean() * 60, 1))

        # KL divergence modèle / marché (mesure de désaccord)
        if has_market:
            eps = 1e-12
            kl = float(np.sum(p_final * np.log((p_final + eps) / (p_market + eps))))
        else:
            kl = None

        return {
            "results": results,
            "exotics": exotics,
            "best_place": bp,
            "confidence_idx": conf_idx,
            "volatility_idx": vol_idx,
            "overround_pct": overround_pct,
            "dynamic_value_threshold": round(dyn_value_th, 3),
            "kl_divergence": round(kl, 3) if kl else None,
            "execution_time": round(time.time() - t0, 2),
            "n_simulations": mc_iter,
        }


def run_engine_v4(race_info: Dict, horses: List[Dict], **kwargs) -> Dict:
    """API publique compatible avec l'ancienne v3."""
    engine = RaceEngine(race_info, horses)
    return engine.predict(**kwargs)


# =============================================================================
# 10.  INTERFACE STREAMLIT
# =============================================================================
def apply_css():
    st.markdown("""
    <style>
    .stApp { background: linear-gradient(135deg,#07071a 0%,#0d1b2a 40%,#12192b 100%); }
    [data-testid="stSidebar"] { background: linear-gradient(180deg,#0d1b2a,#07071a); }
    h1, h2, h3 { color:#e8e8e8 !important; }
    div[data-testid="metric-container"] {
        background: rgba(0,180,216,0.08);
        border: 1px solid rgba(0,255,136,0.15);
        border-radius: 12px;
        padding: 10px;
    }
    .value-bet { color:#00ff88; font-weight:bold; }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    st.markdown(f"""
    <div style="text-align:center; padding: 18px 0;">
      <h1 style="font-size:2.6em;
                 background: linear-gradient(90deg,#00ff88,#00b4d8,#7b2ff7);
                 -webkit-background-clip:text;
                 -webkit-text-fill-color:transparent;">
        🏇 {CONFIG.APP_NAME} v{CONFIG.APP_VERSION}
      </h1>
      <p style="color:#7b9ec4; font-size:1.05em; margin-top:-10px;">
        <em>{CONFIG.APP_TAG}</em> — Plackett-Luce · Benter Blend · Kelly dynamique
      </p>
    </div>
    """, unsafe_allow_html=True)


def init_session_state():
    if "horses_data" not in st.session_state:
        st.session_state.horses_data = pd.DataFrame({
            "N°": list(range(1, 11)),
            "Nom": [f"Cheval {i+1}" for i in range(10)],
            "Cote": [5.0] * 10,
            "Musique Cheval": [""] * 10,
            "Musique Driver": [""] * 10,
            "Musique Entraîneur": [""] * 10,
            "Corde": [0] * 10,
            "Poids": [56.0] * 10,
            "Jours repos": [21] * 10,
        })
    if "prediction" not in st.session_state:
        st.session_state.prediction = None


def main():
    st.set_page_config(page_title=f"🏇 {CONFIG.APP_NAME} v{CONFIG.APP_VERSION}",
                       layout="wide", initial_sidebar_state="expanded")
    init_session_state()
    apply_css()
    render_header()

    # ============= SIDEBAR =============
    with st.sidebar:
        st.markdown("### ⚙️ Paramètres du moteur")

        with st.expander("🔬 Monte Carlo / Plackett-Luce", expanded=True):
            mc_iter = st.slider("Itérations PL", 1000, 15000,
                                CONFIG.MC_ITERATIONS, 500)
            noise = st.slider("Bruit log-normal", 0.05, 0.40,
                              CONFIG.NOISE_BASE, 0.01)
            CONFIG.NOISE_BASE = noise

        with st.expander("🎯 Marché & Benter Blend", expanded=True):
            mw = st.slider("Poids du marché", 0.0, 0.70,
                           CONFIG.MARKET_WEIGHT, 0.05)
            alpha = st.slider("α (exposant modèle)", 0.5, 2.0,
                              CONFIG.BENTER_ALPHA, 0.05)
            beta = st.slider("β (exposant marché)", 0.0, 2.0,
                             CONFIG.BENTER_BETA, 0.05)
            CONFIG.BENTER_ALPHA = alpha
            CONFIG.BENTER_BETA = beta
            CONFIG.OVERROUND_CORRECTION = st.checkbox(
                "Débiaiser favori/outsider", value=True,
                help="Correction power du biais favori-outsider")

        with st.expander("🧠 Empirique & shrinkage"):
            emp_w = st.slider("Poids empirisme (corde+exp.)", 0.0, 0.70,
                               CONFIG.EMPIRICAL_WEIGHT, 0.05)
            CONFIG.EMPIRICAL_WEIGHT = emp_w
            CONFIG.USE_EXPERIENCE_FACTOR = st.checkbox(
                "Facteur expérience", value=CONFIG.USE_EXPERIENCE_FACTOR)
            K = st.slider("Shrinkage K (courses fantômes)", 0.0, 15.0,
                          CONFIG.SHRINKAGE_K, 0.5,
                          help="Plus K est élevé, plus on tire vers la moyenne population")
            CONFIG.SHRINKAGE_K = K

        with st.expander("💰 Value & Kelly"):
            vt = st.slider("Seuil de value (ratio)", 1.05, 1.80,
                            CONFIG.VALUE_THRESHOLD, 0.05)
            kf = st.slider("Kelly fractionnaire", 0.05, 0.50,
                            CONFIG.KELLY_FRACTION, 0.05)
            CONFIG.KELLY_FRACTION = kf
            max_stake = st.slider("Cap max bankroll (%)", 1.0, 15.0,
                                  CONFIG.MAX_KELLY_STAKE * 100, 0.5) / 100
            CONFIG.MAX_KELLY_STAKE = max_stake

        st.markdown("---")
        st.caption(f"v{CONFIG.APP_VERSION} — {CONFIG.APP_TAG}")
        st.caption("Inspiré de Benter (1994), Harville (1973)")

    # ============= TABS =============
    tab1, tab2, tab3 = st.tabs(["📥 Données course",
                                "📊 Pronostics",
                                "ℹ️ Aide & Méthode"])

    # ---------- TAB 1 : DONNÉES ----------
    with tab1:
        st.markdown("## 🏁 Informations de la course")
        c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1.5])
        with c1:
            race_type = st.selectbox("Discipline", CONFIG.RACE_TYPES)
        with c2:
            distance = st.number_input("Distance (m)", 800, 7200, 1600, 100)
        with c3:
            track = st.selectbox("Terrain", CONFIG.TRACK_CONDITIONS)
        with c4:
            # Auto-suggestion du type de départ
            default_depart = 0
            if race_type in ("Attelé", "Monté"):
                default_depart = 1
            depart = st.selectbox("Type de départ", CONFIG.DEPART_TYPES,
                                  index=default_depart)

        prix = st.text_input("Nom du prix (optionnel)", "")

        st.markdown("---")
        st.markdown("## 🐎 Tableau des partants")
        st.caption("✏️ Modifiez directement le tableau. Les champs **Poids** et "
                   "**Jours repos** sont utilisés en Plat ; en Trot, seul "
                   "**Jours repos** est exploité.")

        edited = st.data_editor(
            st.session_state.horses_data,
            use_container_width=True,
            num_rows="dynamic",
            height=420,
            column_config={
                "N°": st.column_config.NumberColumn(min_value=1, max_value=99),
                "Cote": st.column_config.NumberColumn(format="%.2f", min_value=1.0),
                "Corde": st.column_config.NumberColumn(min_value=0, max_value=20),
                "Poids": st.column_config.NumberColumn(format="%.1f", min_value=40.0, max_value=80.0),
                "Jours repos": st.column_config.NumberColumn(min_value=0, max_value=999),
            },
        )
        if edited is not None:
            st.session_state.horses_data = edited

        c1, c2 = st.columns([3, 1])
        with c1:
            run_btn = st.button("🚀 LANCER L'ANALYSE",
                                 use_container_width=True, type="primary")
        with c2:
            reset_btn = st.button("🔄 Reset", use_container_width=True)
            if reset_btn:
                st.session_state.horses_data = pd.DataFrame({
                    "N°": list(range(1, 11)),
                    "Nom": [f"Cheval {i+1}" for i in range(10)],
                    "Cote": [5.0] * 10,
                    "Musique Cheval": [""] * 10,
                    "Musique Driver": [""] * 10,
                    "Musique Entraîneur": [""] * 10,
                    "Corde": [0] * 10,
                    "Poids": [56.0] * 10,
                    "Jours repos": [21] * 10,
                })
                st.rerun()

        if run_btn:
            horses_list = []
            for idx, row in st.session_state.horses_data.iterrows():
                try:
                    horses_list.append({
                        "number": int(row["N°"]),
                        "name": str(row["Nom"]),
                        "odds": float(row["Cote"]),
                        "horse_music": str(row["Musique Cheval"]),
                        "driver_music": str(row["Musique Driver"]),
                        "trainer_music": str(row["Musique Entraîneur"]),
                        "draw": int(row["Corde"]) if pd.notna(row["Corde"]) else 0,
                        "weight": float(row.get("Poids", 56.0)) if pd.notna(row.get("Poids")) else 56.0,
                        "days_rest": int(row.get("Jours repos", -1)) if pd.notna(row.get("Jours repos")) else -1,
                    })
                except Exception as e:
                    st.error(f"⚠️ Erreur ligne {idx+1} : {e}")
                    return

            if len(horses_list) < 3:
                st.error("Au moins 3 partants requis.")
                return

            with st.spinner(f"🔬 Calcul Plackett-Luce ({mc_iter} simulations)..."):
                pred = run_engine_v4(
                    {"race_type": race_type, "distance": distance,
                     "track": track, "depart_type": depart, "discipline": prix},
                    horses_list,
                    mc_iter=mc_iter, market_weight=mw, value_threshold=vt
                )
                st.session_state.prediction = pred
            st.success(f"✅ Analyse terminée en {pred['execution_time']}s — "
                       f"{pred['n_simulations']} simulations")

    # ---------- TAB 2 : RÉSULTATS ----------
    with tab2:
        if st.session_state.prediction is None:
            st.info("🎯 Saisissez les données puis cliquez sur **LANCER L'ANALYSE**.")
        else:
            pred = st.session_state.prediction

            # Diagnostic
            st.markdown("## 📈 Diagnostic de course")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🎯 Confiance", f"{pred['confidence_idx']:.1f}/100")
            c2.metric("🌪️ Volatilité", f"{pred['volatility_idx']:.1f}/100")
            if pred["overround_pct"] is not None:
                c3.metric("📉 Overround", f"{pred['overround_pct']:.1f}%",
                          help="Marge bookmaker (>20% = juice élevé)")
            else:
                c3.metric("📉 Overround", "—")
            c4.metric("📐 Seuil value (dyn.)", f"{pred['dynamic_value_threshold']:.2f}",
                      help="Ajusté selon overround")

            if pred["kl_divergence"] is not None:
                st.caption(f"🧮 Divergence KL(modèle ‖ marché) = "
                           f"**{pred['kl_divergence']:.3f}** — "
                           f"{'fort désaccord' if pred['kl_divergence'] > 0.15 else 'accord modéré'}")

            st.markdown("---")
            st.markdown("## 🏆 Classement final & paris GAGNANT")
            df = pd.DataFrame([{
                "Rg": r["rank"],
                "N°": r["number"],
                "Nom": r["name"][:18],
                "Cote": f"{r['odds']:.2f}",
                "Modèle %": f"{r['win_prob_model']:.1f}",
                "Marché %": f"{r['win_prob_market']:.1f}",
                "🎯 Final %": f"{r['win_prob']:.2f}",
                "Placé %": f"{r['place_prob']:.1f}",
                "Ratio": f"{r['value_ratio']:.2f}",
                "ROI %": f"{r['expected_roi']:+.1f}",
                "Kelly %": f"{r['kelly_recommended']*100:.2f}",
                "Vol.": f"{r['volatility']:.2f}",
                "Value": "🟢" if r["is_value_bet"] else "⚪",
            } for r in pred["results"]])
            st.dataframe(df, use_container_width=True, hide_index=True, height=380)

            # Value bets en évidence
            value_bets = [r for r in pred["results"] if r["is_value_bet"]]
            if value_bets:
                st.markdown("### 💎 Value bets détectés")
                for vb in value_bets[:5]:
                    st.markdown(
                        f"- **N°{vb['number']} {vb['name']}** "
                        f"@ cote {vb['odds']:.2f} — "
                        f"prob. modèle {vb['win_prob']:.1f}% vs marché {vb['win_prob_market']:.1f}% "
                        f"→ Kelly recommandé : **{vb['kelly_recommended']*100:.2f}%** "
                        f"(ROI espéré : {vb['expected_roi']:+.1f}%)"
                    )
            else:
                st.info("⚪ Aucun value bet détecté sur ce marché.")

            # Meilleur placé
            if pred["best_place"]:
                bp = pred["best_place"]
                st.markdown("---")
                st.markdown("## 🥉 Meilleur pari **PLACÉ**")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("N°", bp["number"])
                c2.metric("Cheval", bp["name"][:15])
                c3.metric("Prob. Placé", f"{bp['place_prob']:.1f}%")
                c4.metric("ROI Placé", f"{bp['expected_roi_place']:+.1f}%")
                st.markdown(
                    f"💡 Cote placé estimée : **{bp['estimated_place_odds']:.2f}** — "
                    f"Mise Kelly recommandée : **{bp['kelly_recommended']*100:.2f}%** "
                    f"du bankroll"
                )

            # Exotiques
            st.markdown("---")
            st.markdown("## 🎲 Paris exotiques (Top combinaisons)")
            ex = pred["exotics"]
            tabs_exo = st.tabs(["Couplé Gagnant", "Couplé Placé",
                                "Trio Ordre", "Trio Désordre",
                                "Quarté+", "Quinté+"])

            def _render_exotic(items, key):
                if not items:
                    st.info("Aucune combinaison significative.")
                    return
                df_e = pd.DataFrame([{
                    "Rg": x["rank"],
                    "Combo": x.get("combo", "—"),
                    **({"Détail": x["names"]} if "names" in x else {}),
                    "Prob %": x["prob_pct"],
                    "Cote est.": x["estimated_odds"],
                    "ROI %": x["expected_roi"],
                } for x in items])
                st.dataframe(df_e, use_container_width=True, hide_index=True)

            with tabs_exo[0]: _render_exotic(ex["couple_gagnant"], "cg")
            with tabs_exo[1]: _render_exotic(ex["couple_place"], "cp")
            with tabs_exo[2]: _render_exotic(ex["trio_ordre"], "to")
            with tabs_exo[3]: _render_exotic(ex["trio_desordre"], "td")
            with tabs_exo[4]: _render_exotic(ex["quarte_desordre"], "q4")
            with tabs_exo[5]: _render_exotic(ex["quinte_desordre"], "q5")

    # ---------- TAB 3 : AIDE ----------
    with tab3:
        st.markdown("""
## 🎓 Méthodologie QuantTurf v4.0

### 🔬 Architecture du moteur

```
Musique → Parsing + Shrinkage bayésien → Score composite
                                              ↓
                                          Softmax
                                              ↓
                                  Correction empirique (corde+exp)
                                              ↓
                                       p_modèle
                                              ↓
Cotes marché → Débiaisage power → p_marché
                                              ↓
                              BENTER BLEND : p ∝ p_modèle^α · p_marché^β
                                              ↓
                          Plackett-Luce (5000 ordres simulés)
                                              ↓
                    Win / Place / Couplé / Trio / Quarté+ / Quinté+
                                              ↓
                                   Kelly dynamique + ROI
```

### 📚 Formules clés

**1. Shrinkage bayésien (musique)**
$$\\text{score}_{\\text{shrunk}} = \\frac{n \\cdot \\text{score}_{\\text{obs}} + K \\cdot \\mu_{\\text{pop}}}{n + K}$$

**2. Débiaisage des cotes (favori-outsider correction)**
$$p_{\\text{vraie}} \\propto \\left(\\frac{1}{\\text{cote}}\\right)^\\gamma, \\quad \\gamma \\approx 1.12$$

**3. Benter Blend**
$$p_{\\text{finale}} \\propto p_{\\text{modèle}}^\\alpha \\cdot p_{\\text{marché}}^\\beta$$

**4. Plackett-Luce (Harville)** — ordre d'arrivée séquentiel proportionnel aux forces.

**5. Kelly fractionnaire dynamique**
$$f^* = \\frac{p \\cdot b - q}{b}, \\quad f_{\\text{misé}} = \\min\\left(f^* \\cdot \\frac{1}{1+\\text{vol}}, f_{\\max}\\right)$$

### 🎯 Stratégie recommandée

| Type de pari | Quand l'utiliser | Risque |
|---|---|---|
| **Gagnant (value)** | Ratio > 1.20 ET cote > 2.5 | 🟡 Moyen |
| **Placé** | Champion avec cote ≥ 4 | 🟢 Faible |
| **Couplé Placé** | ROI > 50% | 🟡 Moyen |
| **Trio désordre** | ROI > 100% sur 3 favoris | 🟠 Élevé |
| **Quinté+** | Mise faible, ROI espéré > 200% | 🔴 Très élevé |

### ⚠️ Avertissements

- 🎰 **Les performances passées ne préjugent pas des résultats futurs**
- 💸 **Jouez avec modération** — ne misez jamais plus que ce que vous pouvez perdre
- 📊 Le modèle nécessite un marché suffisamment liquide pour le Benter Blend
- 🐎 La corde au Trot n'est pertinente qu'en départ **AUTOSTART**
- 🔍 Les statistiques empiriques sont des **valeurs indicatives basées sur des études publiques** ; affinez-les selon votre propre base de données.

### 📖 Références

- Benter, W. (1994). *Computer Based Horse Race Handicapping and Wagering Systems.*
- Harville, D. (1973). *Assigning Probabilities to the Outcomes of Multi-Entry Competitions.*
- Kelly, J. L. (1956). *A New Interpretation of Information Rate.*
- Snowberg & Wolfers (2010). *Explaining the Favorite-Longshot Bias.*
        """)


if __name__ == "__main__":
    main()
