"""
QuantTurf Pro v3.1 — Professional Grade Horse Racing Prediction Engine
=======================================================================
Architecture: Modular | Robust Validation | Performance Optimized | ROI-Focused
Nouveau v3.1 : Saisie des partants via tableau Excel-like (st.data_editor)

Author: QuantTurf Analytics
Version: 3.1.0 (Excel-like Input Release)
Requirements: streamlit>=1.30, numpy>=1.24, pandas>=2.0, plotly>=5.0, scipy>=1.10
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from scipy.stats import zscore, norm, entropy
from itertools import combinations
import re
import time
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Tuple, Any
from functools import lru_cache
from enum import Enum
import json
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# LOGGING & CONFIG
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Centralized configuration management"""
    APP_VERSION: str = "3.1.0"
    APP_NAME: str = "QuantTurf Pro"
    
    # Core parameters
    MC_ITERATIONS: int = 3000
    MARKET_WEIGHT: float = 0.35
    VALUE_THRESHOLD: float = 1.15
    TEMPERATURE: float = 1.5
    NOISE_BASE: float = 0.15
    
    # Advanced parameters
    KELLY_FRACTION: float = 0.25
    MIN_KELLY_ODDS: float = 2.50
    CONFIDENCE_MIN_BET: float = 0.65
    
    # Validation thresholds
    MIN_RUNNERS: int = 2
    MAX_RUNNERS: int = 25
    MIN_MUSIC_LENGTH: int = 1
    MIN_ODDS: float = 1.01
    MAX_ODDS: float = 999.0
    MIN_DISTANCE: int = 800
    MAX_DISTANCE: int = 8000
    MIN_AGE: int = 2
    MAX_AGE: int = 20
    
    RACE_TYPES: List[str] = field(default_factory=lambda: 
        ["Plat", "Attelé", "Monté", "Haies", "Steeple-chase", "Cross-country"])
    
    MUSIC_POSITION_SCORES: Dict[str, float] = field(default_factory=lambda: {
        "1": 10.0, "2": 7.5, "3": 5.5, "4": 4.0, "5": 3.0,
        "6": 2.0, "7": 1.5, "8": 1.0, "9": 0.5, "0": 0.2,
        "D": -2.0, "A": -1.5, "T": -1.5, "R": -1.0, "P": 0.3,
    })
    
    MUSIC_RACE_TYPE_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "a": 1.00, "m": 0.90, "p": 1.00, "h": 0.95,
        "s": 0.90, "c": 0.85, "x": 1.00,
    })
    
    DRAW_IMPACT_BASE: Dict[int, float] = field(default_factory=lambda: {
        1: 0.35, 2: 0.40, 3: 0.35, 4: 0.25, 5: 0.15,
        6: 0.05, 7: -0.05, 8: -0.12, 9: -0.18, 10: -0.24,
        11: -0.30, 12: -0.35, 13: -0.40, 14: -0.44, 15: -0.48,
        16: -0.50, 17: -0.52, 18: -0.54, 19: -0.55, 20: -0.55,
    })

CONFIG = Config()

# =============================================================================
# DATACLASSES & ENUMS
# =============================================================================

class RaceType(str, Enum):
    PLAT = "Plat"
    ATTELE = "Attelé"
    MONTE = "Monté"
    HAIES = "Haies"
    STEEPLE = "Steeple-chase"
    CROSS = "Cross-country"


@dataclass
class HorseInputData:
    number: int
    name: str
    age: int
    sex: str
    odds: float = 0.0
    earnings: int = 0
    driver_win_pct: float = 12.0
    trainer_win_pct: float = 12.0
    music: str = ""
    draw: int = 0
    
    def validate(self) -> List[str]:
        errors = []
        if not (1 <= self.number <= 30):
            errors.append(f"N°{self.number}: numéro hors limites [1-30]")
        if not self.name or len(self.name.strip()) == 0:
            errors.append(f"N°{self.number}: nom obligatoire")
        if not (CONFIG.MIN_AGE <= self.age <= CONFIG.MAX_AGE):
            errors.append(f"N°{self.number}: âge {self.age} hors limites [{CONFIG.MIN_AGE}-{CONFIG.MAX_AGE}]")
        if self.sex not in ["H", "F", "G", "M", "E"]:
            errors.append(f"N°{self.number}: sexe invalide ({self.sex})")
        if self.odds < 0:
            errors.append(f"N°{self.number}: cote négative ({self.odds})")
        if self.earnings < 0:
            errors.append(f"N°{self.number}: gains négatifs ({self.earnings})")
        if not (0 <= self.driver_win_pct <= 100):
            errors.append(f"N°{self.number}: % driver invalide ({self.driver_win_pct})")
        if not (0 <= self.trainer_win_pct <= 100):
            errors.append(f"N°{self.number}: % entraîneur invalide ({self.trainer_win_pct})")
        if self.draw < 0 or self.draw > 30:
            errors.append(f"N°{self.number}: corde invalide ({self.draw})")
        return errors


@dataclass
class RaceInfo:
    race_type: str
    distance: int
    n_runners: int
    discipline: str = ""
    race_level: str = ""
    date: Optional[str] = None
    
    def validate(self) -> List[str]:
        errors = []
        if self.race_type not in CONFIG.RACE_TYPES:
            errors.append(f"Type de course invalide: {self.race_type}")
        if not (CONFIG.MIN_DISTANCE <= self.distance <= CONFIG.MAX_DISTANCE):
            errors.append(f"Distance {self.distance}m hors limites [{CONFIG.MIN_DISTANCE}-{CONFIG.MAX_DISTANCE}]m")
        if not (CONFIG.MIN_RUNNERS <= self.n_runners <= CONFIG.MAX_RUNNERS):
            errors.append(f"Nombre de partants {self.n_runners} invalide [{CONFIG.MIN_RUNNERS}-{CONFIG.MAX_RUNNERS}]")
        return errors


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
    win_streak: int = 0
    place_streak: int = 0
    consistency: float = 0.0


# =============================================================================
# SECTION 1 — MUSIC PARSING
# =============================================================================

@lru_cache(maxsize=256)
def parse_music(music_str: str) -> MusicMetrics:
    if not music_str or music_str.strip() in ("", "-", "INEDIT", "INÉDIT", "N/A", "0"):
        return _debutant_profile()
    
    try:
        clean = music_str.strip().upper()
        clean = re.sub(r"[() ]", "", clean)
        tokens = re.findall(r"([0-9DATRP])([AMPHSC]?)", clean)
        
        if not tokens:
            return _debutant_profile()
        
        raw_scores, numeric_positions, race_types_seen = [], [], []
        
        for pos_char, rtype_char in tokens:
            rtype = rtype_char.lower() if rtype_char else "a"
            pos_score = CONFIG.MUSIC_POSITION_SCORES.get(pos_char, 0.3)
            type_weight = CONFIG.MUSIC_RACE_TYPE_WEIGHTS.get(rtype, 1.0)
            raw_scores.append(pos_score * type_weight)
            
            if pos_char.isdigit():
                numeric_positions.append(int(pos_char) if pos_char != "0" else 10)
            race_types_seen.append(rtype)
        
        n = len(raw_scores)
        raw_scores = np.array(raw_scores)
        
        decay = np.array([np.exp(-0.30 * i) for i in range(n)])
        decay /= decay.sum()
        weighted_score = float(np.dot(raw_scores, decay))
        
        recent_n = min(3, n)
        recent_decay = decay[:recent_n] / decay[:recent_n].sum()
        recent_form = float(np.dot(raw_scores[:recent_n], recent_decay))
        
        if len(numeric_positions) >= 2:
            pos_std = float(np.std(numeric_positions))
            regularity = max(0.0, 1.0 - pos_std / 5.0)
        else:
            pos_std = 0.0
            regularity = 0.50
        
        if n >= 4:
            recent_avg = np.mean(raw_scores[:n // 2])
            old_avg = np.mean(raw_scores[n // 2:])
            trend = (recent_avg - old_avg) / (abs(old_avg) + 1e-9)
        else:
            trend = 0.0
        
        win_count = sum(1 for p in numeric_positions if p == 1)
        podium_count = sum(1 for p in numeric_positions if p <= 3)
        
        win_streak = _calculate_streak(numeric_positions, 1)
        place_streak = _calculate_streak(numeric_positions, 3)
        
        consistency = 1.0 - (pos_std / 10.0 if len(numeric_positions) >= 2 else 0.5)
        consistency = max(0.0, min(1.0, consistency))
        
        return MusicMetrics(
            score=weighted_score,
            regularity=regularity,
            races_count=n,
            avg_position=float(np.mean(numeric_positions)) if numeric_positions else 5.0,
            best_position=int(min(numeric_positions)) if numeric_positions else 10,
            recent_form=recent_form,
            trend=float(trend),
            is_debutant=False,
            win_ratio=win_count / max(n, 1),
            podium_ratio=podium_count / max(n, 1),
            win_streak=win_streak,
            place_streak=place_streak,
            consistency=consistency,
        )
    
    except Exception as e:
        logger.warning(f"Music parsing error for '{music_str}': {str(e)}")
        return _debutant_profile()


def _debutant_profile() -> MusicMetrics:
    return MusicMetrics(
        score=3.0, regularity=0.50, races_count=0,
        avg_position=5.0, best_position=10, recent_form=3.0,
        trend=0.0, is_debutant=True, win_ratio=0.0, podium_ratio=0.0
    )


def _calculate_streak(positions: List[int], threshold: int) -> int:
    if not positions:
        return 0
    streak = 0
    for p in positions[:5]:
        if p <= threshold:
            streak += 1
        else:
            break
    return streak

# =============================================================================
# SECTION 2 — FEATURE ENGINEERING
# =============================================================================

def age_distance_factor(age: int, distance: int, race_type: str) -> float:
    age = max(CONFIG.MIN_AGE, min(age, CONFIG.MAX_AGE))
    distance = max(CONFIG.MIN_DISTANCE, min(distance, CONFIG.MAX_DISTANCE))
    
    if race_type == RaceType.PLAT:
        if age == 2:
            f = 1.0 if distance <= 1600 else 0.65
        elif age == 3:
            f = 1.05
        elif 4 <= age <= 7:
            f = 1.08 + (age - 4) * 0.01
        elif age == 8:
            f = 1.00
        else:
            f = max(0.70, 1.0 - (age - 8) * 0.05)
    elif race_type in (RaceType.ATTELE, RaceType.MONTE):
        if age <= 3:
            f = 0.80
        elif 4 <= age <= 9:
            f = 1.05 + (age - 4) * 0.01
        elif age == 10:
            f = 1.00
        else:
            f = max(0.75, 1.0 - (age - 10) * 0.04)
    else:
        if age <= 4:
            f = 0.85
        elif 5 <= age <= 10:
            f = 1.05 + (age - 5) * 0.005
        elif age == 11:
            f = 1.00
        else:
            f = max(0.72, 1.0 - (age - 11) * 0.04)
    
    if distance > 3000 and age >= 5:
        f *= 1.04
    return float(f)


def draw_factor(draw: int, race_type: str, distance: int) -> float:
    if race_type != RaceType.PLAT or not draw or draw <= 0:
        return 0.0
    draw = min(int(draw), 20)
    base = CONFIG.DRAW_IMPACT_BASE.get(draw, -0.55)
    if distance <= 1400:
        return base * 1.60
    elif distance <= 1800:
        return base * 1.00
    else:
        return base * 0.45


def earnings_factor(earnings: float, races_count: int) -> float:
    if not earnings or earnings <= 0 or races_count <= 0:
        return 0.40
    epr = earnings / max(races_count, 1)
    log_epr = np.log1p(epr)
    return float(min(1.0, log_epr / np.log1p(20000)))


def human_factor(driver_pct: float, trainer_pct: float) -> float:
    d = max(0.001, float(driver_pct or 12.0) / 100.0)
    t = max(0.001, float(trainer_pct or 12.0) / 100.0)
    combined = float(np.sqrt(d * t))
    if d >= 0.25 and t >= 0.20:
        combined *= 1.30
    elif d >= 0.22 or t >= 0.18:
        combined *= 1.15
    elif d >= 0.18 or t >= 0.15:
        combined *= 1.08
    return combined


def market_prob(odds: float, n_runners: int) -> float:
    if not odds or odds <= CONFIG.MIN_ODDS:
        return 1.0 / max(n_runners, 2)
    return 1.0 / float(odds)

# =============================================================================
# SECTION 3 — NORMALIZATION
# =============================================================================

def normalize_features(features_list: List[Dict]) -> List[Dict]:
    if not features_list:
        return features_list
    df = pd.DataFrame(features_list)
    norm_cols = [
        "music_score", "recent_form", "regularity", "trend",
        "win_ratio", "podium_ratio", "earnings_factor",
        "age_dist_factor", "human_factor",
    ]
    for col in norm_cols:
        if col not in df.columns:
            continue
        vals = df[col].values.astype(float)
        std = vals.std()
        if std > 1e-9:
            df[f"{col}_z"] = (vals - vals.mean()) / std
        else:
            df[f"{col}_z"] = 0.0
        mn, mx = vals.min(), vals.max()
        if mx - mn > 1e-9:
            df[f"{col}_norm"] = (vals - mn) / (mx - mn)
        else:
            df[f"{col}_norm"] = 0.5
    return df.to_dict("records")

# =============================================================================
# SECTION 4 — WEIGHTS
# =============================================================================

def get_race_weights(race_type: str) -> Dict[str, float]:
    weights_map = {
        RaceType.PLAT: {
            "music_score": 0.28, "recent_form": 0.18, "regularity": 0.07,
            "trend": 0.04, "win_ratio": 0.05, "podium_ratio": 0.04,
            "earnings_factor": 0.08, "age_dist_factor": 0.07,
            "draw_factor": 0.09, "human_factor": 0.10,
        },
        RaceType.ATTELE: {
            "music_score": 0.30, "recent_form": 0.20, "regularity": 0.09,
            "trend": 0.05, "win_ratio": 0.06, "podium_ratio": 0.04,
            "earnings_factor": 0.08, "age_dist_factor": 0.04,
            "draw_factor": 0.01, "human_factor": 0.13,
        },
        RaceType.MONTE: {
            "music_score": 0.27, "recent_form": 0.18, "regularity": 0.09,
            "trend": 0.05, "win_ratio": 0.06, "podium_ratio": 0.04,
            "earnings_factor": 0.08, "age_dist_factor": 0.04,
            "draw_factor": 0.00, "human_factor": 0.19,
        },
    }
    default_weights = {
        "music_score": 0.26, "recent_form": 0.20, "regularity": 0.11,
        "trend": 0.04, "win_ratio": 0.05, "podium_ratio": 0.05,
        "earnings_factor": 0.08, "age_dist_factor": 0.06,
        "draw_factor": 0.00, "human_factor": 0.15,
    }
    return weights_map.get(race_type, default_weights)

# =============================================================================
# SECTION 5 — COMPOSITE SCORE
# =============================================================================

def composite_score(feat: Dict, weights: Dict) -> float:
    score = (
        weights["music_score"] * feat.get("music_score", 3.0) +
        weights["recent_form"] * feat.get("recent_form", 3.0) +
        weights["regularity"] * feat.get("regularity", 0.5) * 10.0 +
        weights["trend"] * (feat.get("trend", 0.0) + 1.0) * 5.0 +
        weights["win_ratio"] * feat.get("win_ratio", 0.0) * 20.0 +
        weights["podium_ratio"] * feat.get("podium_ratio", 0.0) * 10.0 +
        weights["earnings_factor"] * feat.get("earnings_factor", 0.4) * 8.0 +
        weights["age_dist_factor"] * feat.get("age_dist_factor", 1.0) * 5.0 +
        weights["draw_factor"] * (feat.get("draw_factor", 0.0) + 1.0) * 5.0 +
        weights["human_factor"] * feat.get("human_factor", 0.12) * 18.0
    )
    return max(0.01, score)

# =============================================================================
# SECTION 6 — SOFTMAX & CALIBRATION
# =============================================================================

def softmax(scores: np.ndarray, temperature: float = CONFIG.TEMPERATURE) -> np.ndarray:
    s = np.array(scores, dtype=float) / temperature
    s -= s.max()
    e = np.exp(s)
    return e / e.sum()


def logit_calibration(raw_probs: np.ndarray) -> np.ndarray:
    eps = 1e-9
    logit = np.log((raw_probs + eps) / (1 - raw_probs + eps))
    logit = logit - logit.mean() * 0.1
    calibrated = 1.0 / (1.0 + np.exp(-logit))
    return calibrated / calibrated.sum()


def bayesian_blend(model_probs: np.ndarray, market_probs: np.ndarray,
                   market_weight: float) -> np.ndarray:
    mp = np.array(market_probs, dtype=float)
    if mp.sum() < 1e-9:
        mp = np.ones(len(model_probs)) / len(model_probs)
    else:
        mp /= mp.sum()
    eps = 1e-9
    lo_model = np.log((model_probs + eps) / (1 - model_probs + eps))
    lo_market = np.log((mp + eps) / (1 - mp + eps))
    lo_blend = (1 - market_weight) * lo_model + market_weight * lo_market
    blended = 1.0 / (1.0 + np.exp(-lo_blend))
    return blended / blended.sum()

# =============================================================================
# SECTION 7 — MONTE CARLO
# =============================================================================

def monte_carlo(features_list: List[Dict], weights: Dict, 
                n_iter: int = CONFIG.MC_ITERATIONS,
                market_weight: float = CONFIG.MARKET_WEIGHT) -> Dict:
    n = len(features_list)
    all_probs = np.zeros((n_iter, n))
    win_counts = np.zeros(n)
    
    base_scores = np.array([composite_score(f, weights) for f in features_list])
    
    noise_factors = np.array([
        2.20 if f.get("is_debutant", False) else
        1.60 if f.get("regularity", 0.5) < 0.30 else
        0.70 if f.get("regularity", 0.5) > 0.80 else
        1.00
        for f in features_list
    ])
    
    for it in range(n_iter):
        noises = np.random.normal(0, CONFIG.NOISE_BASE * noise_factors, n)
        noisy = base_scores * np.exp(noises)
        noisy = np.maximum(noisy, 0.001)
        probs = softmax(noisy)
        all_probs[it] = probs
        winner = np.random.choice(n, p=probs)
        win_counts[winner] += 1
    
    simulated_probs = win_counts / n_iter
    mean_probs = all_probs.mean(axis=0)
    std_probs = all_probs.std(axis=0)
    vol_per_horse = std_probs / (mean_probs + 1e-9)
    
    place_counts = np.zeros(n)
    for it in range(n_iter):
        top2 = np.argsort(-all_probs[it])[:2]
        place_counts[top2] += 1
    place_probs = place_counts / n_iter
    
    return {
        "simulated_probs": simulated_probs,
        "mean_probs": mean_probs,
        "std_probs": std_probs,
        "vol_per_horse": vol_per_horse,
        "place_probs": place_probs,
    }

# =============================================================================
# SECTION 8 — KELLY CRITERION
# =============================================================================

def calculate_kelly_bet(prob: float, odds: float, kelly_fraction: float = CONFIG.KELLY_FRACTION) -> Tuple[float, float]:
    if odds <= CONFIG.MIN_KELLY_ODDS or prob < 0.10:
        return 0.0, 0.0
    q = 1.0 - prob
    b = odds - 1.0
    kelly = (prob * b - q) / b
    kelly = max(0.0, kelly)
    fractional_kelly = kelly * kelly_fraction
    return float(kelly), float(fractional_kelly)


def calculate_roi(prob: float, odds: float, bet_amount: float) -> float:
    if bet_amount <= 0 or odds <= 1.0:
        return 0.0
    expected_winnings = bet_amount * odds * prob
    expected_loss = bet_amount * (1 - prob)
    expected_value = expected_winnings - expected_loss
    return (expected_value / bet_amount) * 100.0

# =============================================================================
# SECTION 9 — MAIN ENGINE
# =============================================================================

def run_engine(race_info: Dict, horses: List[Dict],
               mc_iter: int = CONFIG.MC_ITERATIONS,
               market_weight: float = CONFIG.MARKET_WEIGHT,
               value_threshold: float = CONFIG.VALUE_THRESHOLD) -> Dict:
    start_time = time.time()
    try:
        n_runners = len(horses)
        race_info["n_runners"] = n_runners
        
        race_data = RaceInfo(
            race_type=race_info.get("race_type", "Plat"),
            distance=int(race_info.get("distance", 1600)),
            n_runners=n_runners,
            discipline=race_info.get("discipline", ""),
            race_level=race_info.get("race_level", ""),
        )
        race_errors = race_data.validate()
        if race_errors:
            raise ValueError("\n".join(race_errors))
        
        horse_validated = []
        for h in horses:
            horse_data = HorseInputData(**h)
            horse_errors = horse_data.validate()
            if horse_errors:
                raise ValueError(f"Partant N°{h.get('number')}: " + horse_errors[0])
            horse_validated.append(h)
        
        feats = []
        for h in horse_validated:
            music = parse_music(h.get("music", ""))
            feat = {
                "number": h.get("number", 0),
                "name": h.get("name", ""),
                "odds": float(h.get("odds", 0)),
                "music_score": music.score,
                "recent_form": music.recent_form,
                "regularity": music.regularity,
                "trend": music.trend,
                "win_ratio": music.win_ratio,
                "podium_ratio": music.podium_ratio,
                "races_count": music.races_count,
                "is_debutant": music.is_debutant,
                "age_dist_factor": age_distance_factor(
                    h.get("age", 4), race_data.distance, race_data.race_type
                ),
                "draw_factor": draw_factor(h.get("draw", 0), race_data.race_type, race_data.distance),
                "earnings_factor": earnings_factor(h.get("earnings", 0), music.races_count),
                "human_factor": human_factor(h.get("driver_win_pct", 12), h.get("trainer_win_pct", 12)),
                "market_prob": market_prob(h.get("odds", 0), n_runners),
                "driver_win_pct": h.get("driver_win_pct", 12),
                "trainer_win_pct": h.get("trainer_win_pct", 12),
                "earnings": h.get("earnings", 0),
                "age": h.get("age", 4),
                "sex": h.get("sex", ""),
                "draw": h.get("draw", 0),
                "music_consistency": music.consistency,
                "win_streak": music.win_streak,
            }
            feats.append(feat)
        
        feats = normalize_features(feats)
        weights = get_race_weights(race_data.race_type)
        scores = np.array([composite_score(f, weights) for f in feats])
        
        sm_probs = softmax(scores)
        cal_probs = logit_calibration(sm_probs)
        
        raw_mkt = np.array([f["market_prob"] for f in feats])
        if raw_mkt.sum() < 1e-9:
            raw_mkt = np.ones(n_runners) / n_runners
        norm_mkt = raw_mkt / raw_mkt.sum()
        
        has_odds = any(h.get("odds", 0) > CONFIG.MIN_KELLY_ODDS for h in horses)
        if has_odds:
            bayes_probs = bayesian_blend(cal_probs, norm_mkt, market_weight)
        else:
            bayes_probs = cal_probs
        
        mc = monte_carlo(feats, weights, n_iter=mc_iter, market_weight=market_weight)
        
        final_probs = 0.55 * bayes_probs + 0.45 * mc["mean_probs"]
        final_probs /= final_probs.sum()
        prob_z = zscore(final_probs)
        
        results = []
        for i, (feat, horse) in enumerate(zip(feats, horses)):
            ratio = final_probs[i] / (norm_mkt[i] + 1e-9)
            is_value = ratio >= value_threshold and final_probs[i] >= 0.04
            kelly, kelly_frac = calculate_kelly_bet(final_probs[i], horse.get("odds", 2.0))
            roi = calculate_roi(final_probs[i], horse.get("odds", 2.0), 100.0)
            
            result = {
                "rank": 0,
                "number": horse.get("number", i + 1),
                "name": horse.get("name", f"Cheval {i+1}"),
                "odds": float(horse.get("odds", 0)),
                "sex": horse.get("sex", ""),
                "age": horse.get("age", 4),
                "model_prob": round(float(final_probs[i]) * 100, 2),
                "market_prob": round(float(norm_mkt[i]) * 100, 2),
                "place_prob": round(float(mc["place_probs"][i]) * 100, 2),
                "composite_score": round(float(scores[i]), 4),
                "music_score": round(feat.get("music_score", 0.0), 2),
                "recent_form": round(feat.get("recent_form", 0.0), 2),
                "regularity": round(feat.get("regularity", 0.0), 2),
                "trend": round(feat.get("trend", 0.0), 3),
                "win_ratio": round(feat.get("win_ratio", 0.0), 3),
                "podium_ratio": round(feat.get("podium_ratio", 0.0), 3),
                "human_factor": round(feat.get("human_factor", 0.0), 4),
                "earnings_factor": round(feat.get("earnings_factor", 0.0), 3),
                "draw_factor": round(feat.get("draw_factor", 0.0), 3),
                "value_ratio": round(float(ratio), 2),
                "is_value_bet": is_value,
                "is_debutant": feat.get("is_debutant", False),
                "mc_std": round(float(mc["std_probs"][i]) * 100, 2),
                "prob_z": round(float(prob_z[i]), 3),
                "driver_win_pct": feat.get("driver_win_pct", 12),
                "trainer_win_pct": feat.get("trainer_win_pct", 12),
                "earnings": feat.get("earnings", 0),
                "kelly_criterion": round(kelly, 4),
                "kelly_bet_fraction": round(kelly_frac, 4),
                "expected_roi": round(roi, 2),
                "music_consistency": round(feat.get("music_consistency", 0.5), 3),
                "win_streak": feat.get("win_streak", 0),
            }
            results.append(result)
        
        results.sort(key=lambda x: x["model_prob"], reverse=True)
        for i, r in enumerate(results):
            r["rank"] = i + 1
        
        bases = results[:2]
        outsiders = [r for r in results[2:] if r["model_prob"] > 2.5]
        outsiders.sort(key=lambda x: x["value_ratio"], reverse=True)
        outsiders = outsiders[:3]
        
        top6 = [r["number"] for r in results[:min(6, n_runners)]]
        trio_combos = list(combinations(top6, 3))[:10]
        top8 = [r["number"] for r in results[:min(8, n_runners)]]
        quinte_combos = list(combinations(top8, 5))[:10]
        
        sorted_p = sorted([r["model_prob"] for r in results], reverse=True)
        if len(sorted_p) >= 2:
            gap = sorted_p[0] - sorted_p[1]
            conf_idx = min(100.0, round(45.0 + gap * 2.2, 1))
        else:
            conf_idx = 50.0
        
        avg_vol = float(mc["vol_per_horse"].mean())
        vol_idx = min(100.0, round(avg_vol * 55.0, 1))
        
        if has_odds:
            raw_overround = sum(1.0 / h["odds"] for h in horses if h.get("odds", 0) > CONFIG.MIN_ODDS)
            overround_pct = round((raw_overround - 1.0) * 100, 1)
        else:
            overround_pct = None
        
        execution_time = time.time() - start_time
        
        return {
            "results": results,
            "bases": bases,
            "outsiders": outsiders,
            "trio_combos": trio_combos,
            "quinte_combos": quinte_combos,
            "confidence_idx": conf_idx,
            "volatility_idx": vol_idx,
            "overround_pct": overround_pct,
            "weights": weights,
            "mc": mc,
            "has_odds": has_odds,
            "execution_time": round(execution_time, 2),
        }
    except Exception as e:
        logger.error(f"Engine error: {str(e)}")
        raise

# =============================================================================
# SECTION 10 — ANALYSIS
# =============================================================================

def generate_analysis(pred: Dict, race: Dict) -> str:
    results = pred["results"]
    bases = pred["bases"]
    outsiders = pred["outsiders"]
    conf = pred["confidence_idx"]
    vol = pred["volatility_idx"]
    rt = race.get("race_type", "Plat")
    dist = race.get("distance", 1600)
    nr = race.get("n_runners", len(results))
    
    lines = []
    lines.append(f"## 📊 Analyse QuantTurf Pro v{CONFIG.APP_VERSION}\n")
    lines.append(f"**{rt}** — **{dist}m** — **{nr} partants**\n\n")
    lines.append("---\n\n")
    
    if conf > 72:
        conf_txt = "**Hiérarchie claire** — Favori solidement identifié."
    elif conf > 56:
        conf_txt = "**Difficulté intermédiaire** — Plusieurs candidats sérieux."
    else:
        conf_txt = "**Course très ouverte** — Hiérarchie incertaine."
    
    if vol > 62:
        vol_txt = "**Volatilité très élevée** — Forte incertitude stochastique."
    elif vol > 38:
        vol_txt = "**Volatilité modérée** — Aléas possibles mais lisible."
    else:
        vol_txt = "**Volatilité faible** — Course structurellement stable."
    
    lines.append(f"{conf_txt}\n\n{vol_txt}\n\n")
    
    if bases:
        lines.append("### ⭐ Bases\n")
        for b in bases:
            vsign = " 🟢 VALUE" if b["is_value_bet"] else ""
            lines.append(
                f"- **N°{b['number']} — {b['name']}** : {b['model_prob']}% | "
                f"Kelly {b['kelly_bet_fraction']:.1%} | ROI {b['expected_roi']:.1f}%{vsign}\n"
            )
        lines.append("\n")
    
    if outsiders:
        lines.append("### 💎 Outsiders Value\n")
        for o in outsiders:
            lines.append(
                f"- **N°{o['number']} — {o['name']}** : {o['model_prob']}% | "
                f"Ratio {o['value_ratio']:.2f}x | Cote {o['odds']}\n"
            )
        lines.append("\n")
    
    lines.append("### ⚙️ Configuration\n")
    lines.append(
        f"Confiance: **{conf}/100** | Volatilité: **{vol}/100** | "
        f"Exécution: **{pred.get('execution_time', 0)}s**\n"
    )
    return "".join(lines)

# =============================================================================
# SECTION 11 — UI STYLING
# =============================================================================

def apply_css() -> None:
    st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #07071a 0%, #0d1b2a 40%, #12192b 100%); }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1b2a, #07071a); }
[data-testid="metric-container"] { background: rgba(13,27,42,0.85); border: 1px solid rgba(0,255,136,0.18); }
.stButton > button { background: linear-gradient(135deg, #00c896, #00b4d8); color: #000; font-weight: 700; }
h1, h2, h3 { color: #e8e8e8 !important; }

/* Tableau Excel-like */
[data-testid="stDataFrame"], [data-testid="stDataEditor"] {
    border: 1px solid rgba(0,255,136,0.25);
    border-radius: 8px;
    background: rgba(13,27,42,0.55);
}
.excel-hint {
    background: rgba(0,180,216,0.10);
    border-left: 3px solid #00b4d8;
    padding: 10px 14px;
    border-radius: 6px;
    color: #b8d4e8;
    font-size: 0.88em;
    margin: 8px 0 14px 0;
}
</style>
""", unsafe_allow_html=True)


def render_header() -> None:
    st.markdown(f"""
<div style="text-align:center; padding: 22px 0;">
    <h1 style="font-size:2.8em; background: linear-gradient(90deg,#00ff88,#00b4d8);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
        🏇 {CONFIG.APP_NAME} PRO
    </h1>
    <p style="color:#6b7fa3; font-size:0.95em;">
        Moteur Quantitatif Professionnel — V{CONFIG.APP_VERSION} — Saisie Tableur
    </p>
</div>
""", unsafe_allow_html=True)


def sidebar_config() -> Tuple[int, float, float]:
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.markdown("---")
        mc_iter = st.slider("MC Itérations", 500, 5000, CONFIG.MC_ITERATIONS, 250)
        mw = st.slider("Poids Marché", 0.0, 0.60, CONFIG.MARKET_WEIGHT, 0.05)
        vt = st.slider("Seuil Value", 1.05, 1.60, CONFIG.VALUE_THRESHOLD, 0.05)
        st.markdown("---")
        st.markdown(
            "### 📊 Outils Pro\n"
            "- Kelly Criterion\n"
            "- ROI Expected\n"
            "- Tableau Excel-like\n"
            "- Import / Export CSV"
        )
    return mc_iter, mw, vt

# =============================================================================
# SECTION 12 — TABLEAU EXCEL-LIKE (NOUVEAU)
# =============================================================================

DEFAULT_N_ROWS = 14

def build_empty_dataframe(n_rows: int = DEFAULT_N_ROWS) -> pd.DataFrame:
    """Construit un DataFrame vide avec les colonnes du tableau de saisie."""
    return pd.DataFrame({
        "N°": list(range(1, n_rows + 1)),
        "Nom": [f"Cheval {i+1}" for i in range(n_rows)],
        "Cote": [0.0] * n_rows,
        "Musique": [""] * n_rows,
        "% Driver": [12.0] * n_rows,
        "% Entraîneur": [12.0] * n_rows,
        "Sexe": ["H"] * n_rows,
        "Âge": [4] * n_rows,
        "Gains €": [0] * n_rows,
        "Corde": [0] * n_rows,
    })


def get_column_config() -> Dict:
    """Configuration type-safe des colonnes du data_editor (style Excel)."""
    return {
        "N°": st.column_config.NumberColumn(
            "N°",
            help="Numéro du partant (1-30)",
            min_value=1, max_value=30, step=1, format="%d", width="small",
            required=True,
        ),
        "Nom": st.column_config.TextColumn(
            "Nom",
            help="Nom du cheval",
            max_chars=40, width="medium", required=True,
        ),
        "Cote": st.column_config.NumberColumn(
            "Cote",
            help="Cote PMU (laisser 0 si inconnue)",
            min_value=0.0, max_value=999.0, step=0.1, format="%.1f", width="small",
        ),
        "Musique": st.column_config.TextColumn(
            "Musique",
            help="Ex: 6p11p5p6 — chiffres + lettres (a/m/p/h/s/c)",
            max_chars=40, width="medium",
        ),
        "% Driver": st.column_config.NumberColumn(
            "% Driver",
            help="% de victoires du driver/jockey",
            min_value=0.0, max_value=100.0, step=0.5, format="%.1f%%", width="small",
        ),
        "% Entraîneur": st.column_config.NumberColumn(
            "% Entraîneur",
            help="% de victoires de l'entraîneur",
            min_value=0.0, max_value=100.0, step=0.5, format="%.1f%%", width="small",
        ),
        "Sexe": st.column_config.SelectboxColumn(
            "Sexe",
            help="H=Hongre, F=Femelle, M=Mâle, G=Châtré, E=Entier",
            options=["H", "F", "M", "G", "E"], width="small", required=True,
        ),
        "Âge": st.column_config.NumberColumn(
            "Âge",
            help="Âge du cheval en années",
            min_value=2, max_value=20, step=1, format="%d", width="small",
            required=True,
        ),
        "Gains €": st.column_config.NumberColumn(
            "Gains €",
            help="Gains carrière en €",
            min_value=0, max_value=9_999_999, step=1000, format="%d €", width="medium",
        ),
        "Corde": st.column_config.NumberColumn(
            "Corde",
            help="Numéro de corde (plat seulement, 0 = N/A)",
            min_value=0, max_value=30, step=1, format="%d", width="small",
        ),
    }


def dataframe_to_horses(df: pd.DataFrame) -> List[Dict]:
    """Convertit le DataFrame édité en liste de partants prêts pour l'engine."""
    horses = []
    for _, row in df.iterrows():
        # Ignore les lignes manifestement vides (pas de nom OU N° null)
        name = str(row.get("Nom", "")).strip()
        num = row.get("N°", None)
        if not name or pd.isna(num):
            continue
        try:
            horses.append({
                "number": int(num),
                "name": name,
                "age": int(row.get("Âge", 4) or 4),
                "sex": str(row.get("Sexe", "H") or "H"),
                "odds": float(row.get("Cote", 0) or 0),
                "earnings": int(row.get("Gains €", 0) or 0),
                "driver_win_pct": float(row.get("% Driver", 12) or 12),
                "trainer_win_pct": float(row.get("% Entraîneur", 12) or 12),
                "music": str(row.get("Musique", "") or "").strip(),
                "draw": int(row.get("Corde", 0) or 0),
            })
        except (ValueError, TypeError) as e:
            logger.warning(f"Ligne ignorée (conversion impossible): {e}")
            continue
    return horses


def horses_to_dataframe(horses: List[Dict]) -> pd.DataFrame:
    """Inverse: utile pour charger un exemple ou un CSV importé."""
    rows = []
    for h in horses:
        rows.append({
            "N°": h.get("number", 1),
            "Nom": h.get("name", ""),
            "Cote": h.get("odds", 0.0),
            "Musique": h.get("music", ""),
            "% Driver": h.get("driver_win_pct", 12.0),
            "% Entraîneur": h.get("trainer_win_pct", 12.0),
            "Sexe": h.get("sex", "H"),
            "Âge": h.get("age", 4),
            "Gains €": h.get("earnings", 0),
            "Corde": h.get("draw", 0),
        })
    return pd.DataFrame(rows)


def get_example_dataframe() -> pd.DataFrame:
    """DataFrame d'exemple correspondant à la capture d'écran fournie."""
    data = [
        (1,  "Packing Fighter",  8.8,  "6p11p5p6", 3,  9),
        (2,  "Charity Garden",   42.0, "11p11p5p", 8,  12),
        (3,  "Fluorescent",      100.0,"14p12p14", 10, 4),
        (4,  "Vermilion",        4.3,  "2p8p10p6", 13, 15),
        (5,  "Absolute Hero",    3.7,  "1p4p7p2p", 7,  19),
        (6,  "State Security",   48.0, "11p10p",   10, 7),
        (7,  "Super Gold",       18.0, "1p6p14p6", 7,  9),
        (8,  "Grand Turismo",    12.0, "4p13p13p", 10, 5),
        (9,  "Star Elegance",    22.0, "6p8p2p14", 5,  13),
        (10, "Star Brose",       13.0, "6p3p2p5p", 10, 6),
        (11, "Kyrs Treasure",    43.0, "12p13p12", 8,  5),
        (12, "Kingly Den",       15.0, "7p2p2p6p", 7,  3),
        (13, "Fortune King",     15.0, "5p11p11p", 6,  2),
        (14, "Romantic",         7.5,  "5p3p2p2p", 6,  5),
    ]
    rows = []
    for num, name, cote, mus, drv, trn in data:
        rows.append({
            "N°": num, "Nom": name, "Cote": cote, "Musique": mus,
            "% Driver": float(drv), "% Entraîneur": float(trn),
            "Sexe": "H", "Âge": 5, "Gains €": 0, "Corde": 0,
        })
    return pd.DataFrame(rows)

# =============================================================================
# SECTION 13 — STREAMLIT APP
# =============================================================================

def main() -> None:
    st.set_page_config(
        page_title=f"🏇 {CONFIG.APP_NAME}",
        page_icon="🏇",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_css()
    render_header()
    mc_iter, market_w, value_t = sidebar_config()
    
    # Init state
    if "horses_df" not in st.session_state:
        st.session_state["horses_df"] = build_empty_dataframe(DEFAULT_N_ROWS)
    
    tab1, tab2 = st.tabs(["📥 Données", "📊 Résultats"])
    
    # =================== TAB 1 — DONNÉES ===================
    with tab1:
        # --- Infos de course ---
        st.markdown("## 🏁 Informations de course")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            race_type = st.selectbox("Type", CONFIG.RACE_TYPES)
        with c2:
            distance = st.number_input("Distance (m)", 800, 7200, 1600, 100)
        with c3:
            discipline = st.text_input("Prix")
        with c4:
            level = st.text_input("Niveau")
        
        st.markdown("---")
        
        # --- Tableau Excel-like ---
        st.markdown("## 🐎 Partants — Saisie Tableur")
        st.markdown(
            '<div class="excel-hint">'
            "💡 <b>Mode Excel</b> : éditez directement les cellules. "
            "Ajoutez/supprimez des lignes avec les boutons du tableau. "
            "Copier-coller depuis Excel/Google Sheets supporté. "
            "Les lignes sans <b>Nom</b> seront ignorées."
            "</div>",
            unsafe_allow_html=True,
        )
        
        # Barre d'actions rapides
        a1, a2, a3, a4 = st.columns([1, 1, 1, 2])
        with a1:
            if st.button("➕ +5 lignes", use_container_width=True):
                current = st.session_state["horses_df"]
                n_current = len(current)
                extra = build_empty_dataframe(5)
                extra["N°"] = range(n_current + 1, n_current + 6)
                extra["Nom"] = [f"Cheval {n_current + i + 1}" for i in range(5)]
                st.session_state["horses_df"] = pd.concat(
                    [current, extra], ignore_index=True
                )
                st.rerun()
        with a2:
            if st.button("🗑️ Vider", use_container_width=True):
                st.session_state["horses_df"] = build_empty_dataframe(DEFAULT_N_ROWS)
                st.rerun()
        with a3:
            if st.button("📋 Exemple", use_container_width=True, help="Charge l'exemple de la capture d'écran"):
                st.session_state["horses_df"] = get_example_dataframe()
                st.rerun()
        with a4:
            uploaded = st.file_uploader(
                "📂 Importer CSV", type=["csv"], label_visibility="collapsed"
            )
            if uploaded is not None:
                try:
                    df_imp = pd.read_csv(uploaded)
                    # On garde uniquement les colonnes connues, on ajoute celles qui manquent
                    expected = build_empty_dataframe(1).columns.tolist()
                    for col in expected:
                        if col not in df_imp.columns:
                            df_imp[col] = build_empty_dataframe(len(df_imp))[col]
                    st.session_state["horses_df"] = df_imp[expected]
                    st.success(f"✅ {len(df_imp)} ligne(s) importée(s).")
                except Exception as e:
                    st.error(f"Import KO : {e}")
        
        # Éditeur de tableau (cœur du nouveau design)
        edited_df = st.data_editor(
            st.session_state["horses_df"],
            column_config=get_column_config(),
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            key="horses_editor",
            height=min(560, 45 + 35 * (len(st.session_state["horses_df"]) + 1)),
        )
        st.session_state["horses_df"] = edited_df
        
        # Stats rapides + export CSV
        horses_preview = dataframe_to_horses(edited_df)
        s1, s2, s3 = st.columns([1, 1, 2])
        with s1:
            st.metric("Partants valides", len(horses_preview))
        with s2:
            with_odds = sum(1 for h in horses_preview if h["odds"] > 0)
            st.metric("Avec cote", with_odds)
        with s3:
            csv_buffer = edited_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "💾 Exporter CSV",
                data=csv_buffer,
                file_name=f"partants_{datetime.now():%Y%m%d_%H%M}.csv",
                mime="text/csv",
                use_container_width=True,
            )
        
        st.markdown("---")
        
        # --- Lancement de l'analyse ---
        if st.button("🚀 ANALYSER LA COURSE", use_container_width=True, type="primary"):
            horses_input = dataframe_to_horses(edited_df)
            if len(horses_input) < 2:
                st.error("❌ Minimum 2 partants valides (avec Nom rempli) requis.")
                return
            if len(horses_input) > CONFIG.MAX_RUNNERS:
                st.error(f"❌ Maximum {CONFIG.MAX_RUNNERS} partants autorisés.")
                return
            
            with st.spinner(f"Analyse de {len(horses_input)} partants..."):
                try:
                    pred = run_engine(
                        {"race_type": race_type, "distance": int(distance),
                         "discipline": discipline, "race_level": level},
                        horses_input,
                        mc_iter=mc_iter, market_weight=market_w, value_threshold=value_t
                    )
                    st.session_state["prediction"] = pred
                    st.session_state["race_info"] = {
                        "race_type": race_type, "distance": distance,
                        "n_runners": len(horses_input),
                    }
                    st.success(f"✅ Analyse terminée en {pred.get('execution_time', 0)}s — Voir l'onglet 📊 Résultats")
                except Exception as e:
                    st.error(f"❌ Erreur : {str(e)}")
    
    # =================== TAB 2 — RÉSULTATS ===================
    with tab2:
        if "prediction" not in st.session_state:
            st.info("👈 Lancez l'analyse depuis l'onglet **📥 Données**.")
        else:
            pred = st.session_state["prediction"]
            
            st.markdown("## 📊 KPIs")
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Confiance", f"{pred['confidence_idx']}/100")
            with k2:
                st.metric("Volatilité", f"{pred['volatility_idx']}/100")
            with k3:
                st.metric("Partants", len(pred["results"]))
            with k4:
                vb = sum(1 for r in pred["results"] if r["is_value_bet"])
                st.metric("Value Bets", vb)
            
            st.markdown("---\n## 🏆 Classement")
            res_df = []
            for r in pred["results"]:
                res_df.append({
                    "Rg": r["rank"],
                    "N°": r["number"],
                    "Nom": r["name"],
                    "Cote": r["odds"],
                    "Modèle %": f"{r['model_prob']:.2f}",
                    "Marché %": f"{r['market_prob']:.2f}",
                    "Place %": f"{r['place_prob']:.2f}",
                    "Kelly %": f"{r['kelly_bet_fraction']*100:.2f}",
                    "ROI %": f"{r['expected_roi']:.1f}",
                    "Ratio": f"{r['value_ratio']:.2f}x",
                    "Value": "🟢" if r["is_value_bet"] else "⚪",
                })
            st.dataframe(pd.DataFrame(res_df), use_container_width=True, hide_index=True)
            
            st.markdown("---\n## 💡 Analyse")
            st.markdown(generate_analysis(pred, st.session_state.get("race_info", {})))


if __name__ == "__main__":
    main()
