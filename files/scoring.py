"""Logique de calcul des scores intrinsèque et de valeur.

Principe : les paramètres sont normalisés RELATIVEMENT au peloton de la
course du jour (et non sur une échelle absolue fixe), car le pronostic
hippique est un problème de classement relatif, pas de score dans l'absolu.
"""

import pandas as pd

from models import COL_FORME, COL_POIDS, COL_DISTANCE, COL_TERRAIN, COL_JOCKEY, COL_COTE

MAPPING_APTITUDE = {"Favorable": 10, "Neutre": 5, "Défavorable": 0}
MAPPING_JOCKEY = {"Fort": 10, "Moyen": 5, "Faible": 0}

# Poids par défaut du score intrinsèque (modifiables dans l'interface).
POIDS_DEFAUT = {
    "forme": 0.30,
    "distance": 0.20,
    "terrain": 0.20,
    "poids": 0.15,
    "jockey": 0.15,
}


def _normaliser_min_max(serie: pd.Series, inverser: bool = False) -> pd.Series:
    """Normalise une série numérique entre 0 et 10, relativement au peloton du jour.

    inverser=True est utilisé pour le poids porté : porter moins de poids
    que les autres partants est un avantage, donc une note plus haute.
    """
    minimum, maximum = serie.min(), serie.max()
    if maximum == minimum:
        # Tous les chevaux sont à égalité sur ce critère : note neutre pour tous.
        return pd.Series([5.0] * len(serie), index=serie.index)
    normalise = (serie - minimum) / (maximum - minimum) * 10
    if inverser:
        normalise = 10 - normalise
    return normalise


def calculer_score_intrinseque(df: pd.DataFrame, poids: dict = None) -> pd.DataFrame:
    """Ajoute les notes normalisées par critère et le score intrinsèque pondéré.

    Le score intrinsèque n'utilise PAS la cote : il représente l'estimation
    de la valeur du cheval indépendamment de l'opinion du marché.
    """
    poids = poids or POIDS_DEFAUT
    resultat = df.copy()

    resultat["note_forme"] = _normaliser_min_max(df[COL_FORME])
    resultat["note_poids"] = _normaliser_min_max(df[COL_POIDS], inverser=True)
    resultat["note_distance"] = df[COL_DISTANCE].map(MAPPING_APTITUDE)
    resultat["note_terrain"] = df[COL_TERRAIN].map(MAPPING_APTITUDE)
    resultat["note_jockey"] = df[COL_JOCKEY].map(MAPPING_JOCKEY)

    resultat["score_intrinseque"] = (
        resultat["note_forme"] * poids["forme"]
        + resultat["note_distance"] * poids["distance"]
        + resultat["note_terrain"] * poids["terrain"]
        + resultat["note_poids"] * poids["poids"]
        + resultat["note_jockey"] * poids["jockey"]
    )
    return resultat


def calculer_score_valeur(df: pd.DataFrame) -> pd.DataFrame:
    """Compare la probabilité implicite du modèle à celle du marché (cote).

    Nécessite que calculer_score_intrinseque ait déjà été appelé (colonne
    'score_intrinseque' présente).
    """
    resultat = df.copy()

    total_score = resultat["score_intrinseque"].sum()
    resultat["proba_modele"] = (
        resultat["score_intrinseque"] / total_score if total_score > 0 else 0.0
    )

    # Probabilité implicite brute par la cote, puis retrait du surround
    # (l'overround / la marge du bookmaker) pour que les probabilités
    # du peloton somment bien à 100%.
    proba_marche_brute = 1 / resultat[COL_COTE]
    overround = proba_marche_brute.sum()
    resultat["proba_marche"] = proba_marche_brute / overround

    resultat["ecart_valeur"] = resultat["proba_modele"] - resultat["proba_marche"]
    return resultat
