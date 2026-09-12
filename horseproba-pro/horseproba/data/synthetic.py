"""
Génération d'un historique de courses synthétique mais réaliste.

Pourquoi des données synthétiques ?
-----------------------------------
- Permettre à l'application de fonctionner hors-ligne et de démontrer le
  pipeline complet (features -> modèle -> backtest) sans dépendre d'une source
  web fragile.
- Vérifier que le modèle « retrouve » des effets connus : la « vérité terrain »
  est ici un modèle de Plackett-Luce dont on connaît les coefficients.

Mécanisme
---------
Pour chaque course on tire n partants avec des attributs corrélés (un cheval
ayant beaucoup gagné a une bonne musique et une cote basse). La force latente
s_i est une combinaison linéaire des attributs + bruit. L'arrivée est obtenue
par échantillonnage de Plackett-Luce : on tire le gagnant avec probabilité
softmax(s), puis le 2e parmi les restants, etc. Les cotes du marché sont une
version bruitée des vraies probabilités, avec un prélèvement de ~15 % (pari mutuel).
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import List

import numpy as np
import pandas as pd

_HORSE_PREFIX = [
    "Royal", "Golden", "Silver", "Black", "Blue", "Storm", "Wild", "Lucky", "Fast", "Bold",
    "Night", "Sun", "Star", "Iron", "Velvet", "Crystal", "Diamond", "Shadow", "Fire", "Ocean",
]
_HORSE_SUFFIX = [
    "Dream", "Prince", "Queen", "Arrow", "Wind", "Spirit", "Legend", "Rocket", "Dancer", "Flash",
    "Glory", "Runner", "Knight", "Comet", "Thunder", "Whisper", "Ember", "Tide", "Falcon", "Echo",
]
_JOCKEYS = [
    "M. Guyon", "C. Soumillon", "P-C. Boudot", "M. Barzalona", "A. Lemaitre", "S. Pasquier",
    "T. Bachelot", "C. Demuro", "E. Hardouin", "M. Velon", "A. Pouchin", "J-B. Eyquem",
    "F. Veron", "T. Piccone", "H. Journiac", "D. Boche",
]
_TRAINERS = [
    "A. Fabre", "J-C. Rouget", "F. Head", "C. Laffon-Parias", "F-H. Graffard", "M. Delzangles",
    "H-A. Pantall", "P. Bary", "N. Clément", "S. Wattel", "Y. Barberot", "M. Boutin",
]
_TRACKS = ["ParisLongchamp", "Chantilly", "Deauville", "Saint-Cloud", "Lyon-Parilly", "Vincennes", "Cagnes-sur-Mer"]
_GOINGS = ["bon", "bon souple", "souple", "très souple", "lourd", "PSF"]

# Coefficients « vrais » de la force latente (échelle logit)
TRUE_COEFS = {
    "quality": 1.6,      # qualité intrinsèque (inobservée directement)
    "form": 0.9,         # forme récente
    "jockey": 0.5,       # talent du jockey
    "trainer": 0.4,      # entraîneur
    "draw": -0.25,       # corde extérieure défavorable (plat)
    "weight": -0.06,     # par kg au-dessus de la moyenne
    "fresh": 0.3,        # fraîcheur
}


def _plackett_luce_sample(rng: np.random.Generator, strength: np.ndarray) -> np.ndarray:
    """Tire un ordre d'arrivée complet selon Plackett-Luce ; renvoie les places (1..n)."""
    n = len(strength)
    remaining = list(range(n))
    positions = np.zeros(n, dtype=int)
    for place in range(1, n + 1):
        s = strength[remaining]
        p = np.exp(s - s.max())
        p /= p.sum()
        pick = rng.choice(len(remaining), p=p)
        positions[remaining[pick]] = place
        remaining.pop(pick)
    return positions


def _musique_from_history(rng: np.random.Generator, quality: float, form: float, n: int = 6) -> str:
    """Génère une musique cohérente avec la qualité/forme (les bons chevaux ont de bons chiffres)."""
    tokens: List[str] = []
    for k in range(n):
        latent = quality + form * (0.8**k) + rng.normal(0, 1.0)
        if rng.random() < 0.05:
            tokens.append(rng.choice(["Dp", "Tp", "Ap"]))
            continue
        # plus latent est élevé, plus la place est petite
        place = int(np.clip(round(6 - 2.2 * latent + rng.normal(0, 1.2)), 1, 12))
        tokens.append(f"{place if place <= 9 else 0}p")
    return " ".join(tokens)


def generate_history(
    n_races: int = 300,
    seed: int = 42,
    start_date: date = date(2024, 1, 6),
    min_runners: int = 7,
    max_runners: int = 16,
) -> pd.DataFrame:
    """
    Génère `n_races` courses terminées (avec `finish_position`) au schéma canonique.
    Déterministe pour une graine donnée.
    """
    rng = np.random.default_rng(seed)

    # Talents persistants des jockeys / entraîneurs (échelle logit)
    jockey_skill = dict(zip(_JOCKEYS, rng.normal(0, 0.6, len(_JOCKEYS))))
    trainer_skill = dict(zip(_TRAINERS, rng.normal(0, 0.5, len(_TRAINERS))))

    rows = []
    for r in range(n_races):
        n = int(rng.integers(min_runners, max_runners + 1))
        race_date = start_date + timedelta(days=int(r * 365 / max(n_races, 1)))
        meeting = int(rng.integers(1, 5))
        race_no = int(rng.integers(1, 9))
        race_id = f"{race_date.isoformat()}_R{meeting}_C{race_no}"
        track = rng.choice(_TRACKS)
        discipline = "trot" if track == "Vincennes" else "plat"
        distance = int(rng.choice([1200, 1400, 1600, 1800, 2000, 2100, 2400, 2700, 3000]))
        going = rng.choice(_GOINGS)

        quality = rng.normal(0, 1, n)
        form = rng.normal(0, 0.7, n)
        jockeys = rng.choice(_JOCKEYS, n, replace=False)
        trainers = rng.choice(_TRAINERS, n)
        jsk = np.array([jockey_skill[j] for j in jockeys])
        tsk = np.array([trainer_skill[t] for t in trainers])
        draw = rng.permutation(n) + 1
        draw_c = (draw - draw.mean()) / max(draw.std(), 1)
        weight = np.round(rng.normal(57, 2.0, n) + 1.5 * quality, 1)  # handicap : les bons portent plus
        weight_c = weight - weight.mean()
        days = rng.choice([7, 10, 14, 18, 21, 25, 30, 35, 45, 60, 90, 150, 240], n)
        fresh = np.exp(-((np.log1p(days) - np.log1p(28)) ** 2) / (2 * 0.9**2))
        fresh_c = fresh - fresh.mean()

        strength = (
            TRUE_COEFS["quality"] * quality
            + TRUE_COEFS["form"] * form
            + TRUE_COEFS["jockey"] * jsk
            + TRUE_COEFS["trainer"] * tsk
            + TRUE_COEFS["draw"] * draw_c * (discipline == "plat")
            + TRUE_COEFS["weight"] * weight_c
            + TRUE_COEFS["fresh"] * fresh_c
        )
        positions = _plackett_luce_sample(rng, strength)

        # Marché : probabilités vraies + bruit + prélèvement 15 %
        true_p = np.exp(strength - strength.max())
        true_p /= true_p.sum()
        noisy = np.exp(np.log(true_p) + rng.normal(0, 0.45, n))
        noisy /= noisy.sum()
        odds = np.round(np.clip(1.0 / (noisy * 1.15), 1.1, 150), 1)

        starts = rng.integers(3, 40, n)
        win_rate = 1 / (1 + np.exp(-(quality - 1.2)))
        wins = np.minimum(starts, rng.binomial(starts, np.clip(win_rate, 0.02, 0.6)))
        places = np.minimum(starts, wins + rng.binomial(starts - wins, 0.3))
        earnings = np.round(starts * (2500 + 9000 * np.clip(quality + 1, 0, None)) * rng.uniform(0.7, 1.3, n), 0)

        for i in range(n):
            rows.append(
                {
                    "race_id": race_id,
                    "race_date": race_date.isoformat(),
                    "track": track,
                    "discipline": discipline,
                    "distance_m": distance,
                    "going": going,
                    "horse": f"{rng.choice(_HORSE_PREFIX)} {rng.choice(_HORSE_SUFFIX)} {r % 97 + i}",
                    "jockey": jockeys[i],
                    "trainer": trainers[i],
                    "draw": int(draw[i]),
                    "weight_kg": float(weight[i]),
                    "age": int(rng.integers(3, 8)),
                    "days_since_last_run": int(days[i]),
                    "musique": _musique_from_history(rng, quality[i], form[i]),
                    "career_starts": int(starts[i]),
                    "career_wins": int(wins[i]),
                    "career_places": int(places[i]),
                    "earnings": float(earnings[i]),
                    "odds": float(odds[i]),
                    "finish_position": int(positions[i]),
                }
            )
    return pd.DataFrame(rows)
