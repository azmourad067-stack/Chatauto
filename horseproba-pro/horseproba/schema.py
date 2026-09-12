"""
Schéma de données canonique utilisé dans toute l'application.

Chaque ligne d'un DataFrame « partants » (runners) décrit UN cheval dans UNE course.
Les colonnes sont volontairement simples afin qu'un utilisateur puisse produire un
CSV à la main ou depuis un tableur.

Colonnes obligatoires (minimum vital pour un pronostic) :
    race_id        identifiant unique de la course (texte)          ex: "2024-06-15_R1_C3"
    horse          nom du cheval                                     ex: "Torquator Tasso"

Colonnes fortement recommandées (améliorent nettement le modèle) :
    odds           cote décimale (gain pour 1 misé, ex: 4.5). Peut être la cote probable
                   ou la cote finale. Utilisée comme "sagesse du marché".
    musique        forme récente au format PMU/Turf : "1p3p2p0pDp" (chiffres = place,
                   0 = non placé, D = disqualifié, T = tombé, A = arrêté, lettre = discipline)
    jockey, trainer
    draw           numéro de corde / place à la corde (plat)
    weight_kg      poids porté (plat, handicaps)
    age            âge du cheval
    days_since_last_run   jours depuis la dernière course
    career_starts, career_wins, career_places   statistiques de carrière
    earnings       gains en carrière (euros)

Colonnes de contexte (au niveau de la course, répétées sur chaque ligne) :
    race_date      date ISO "AAAA-MM-JJ"
    track          hippodrome
    discipline     "plat" | "trot" | "obstacle"
    distance_m     distance en mètres
    going          état du terrain ("bon", "souple", "lourd", "PSF", ...)

Colonnes cible (uniquement pour l'historique servant à l'entraînement / backtest) :
    finish_position   place à l'arrivée (1 = gagnant ; 0 ou NaN = non classé/disqualifié)
"""

from __future__ import annotations

REQUIRED_COLUMNS = ["race_id", "horse"]

RECOMMENDED_COLUMNS = [
    "odds",
    "musique",
    "jockey",
    "trainer",
    "draw",
    "weight_kg",
    "age",
    "days_since_last_run",
    "career_starts",
    "career_wins",
    "career_places",
    "earnings",
]

CONTEXT_COLUMNS = ["race_date", "track", "discipline", "distance_m", "going"]

TARGET_COLUMN = "finish_position"

ALL_COLUMNS = REQUIRED_COLUMNS + RECOMMENDED_COLUMNS + CONTEXT_COLUMNS + [TARGET_COLUMN]

NUMERIC_COLUMNS = [
    "odds",
    "odds_probable",
    "odds_direct",
    "draw",
    "weight_kg",
    "age",
    "days_since_last_run",
    "career_starts",
    "career_wins",
    "career_places",
    "earnings",
    "distance_m",
    "finish_position",
]

DISCIPLINES = ["plat", "trot", "obstacle"]
