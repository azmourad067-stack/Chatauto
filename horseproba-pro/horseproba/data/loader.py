"""
Chargement et validation de fichiers CSV au schéma `horseproba.schema`.

Toutes les fonctions sont défensives : elles ne lèvent que `DataValidationError`
(avec un message lisible pour l'utilisateur final) et jamais d'exception brute.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import pandas as pd

from ..schema import NUMERIC_COLUMNS, REQUIRED_COLUMNS, RECOMMENDED_COLUMNS

# Alias fréquents (fichiers export Excel, anglais/français) -> nom canonique
COLUMN_ALIASES = {
    "cheval": "horse",
    "nom": "horse",
    "horse_name": "horse",
    "cote": "odds",
    "cotes": "odds",
    "odd": "odds",
    "forme": "musique",
    "form": "musique",
    "driver": "jockey",
    "entraineur": "trainer",
    "entraîneur": "trainer",
    "corde": "draw",
    "numero": "draw",
    "poids": "weight_kg",
    "weight": "weight_kg",
    "distance": "distance_m",
    "terrain": "going",
    "hippodrome": "track",
    "date": "race_date",
    "course": "race_id",
    "place": "finish_position",
    "arrivee": "finish_position",
    "arrivée": "finish_position",
    "position": "finish_position",
    "gains": "earnings",
    "jours_depuis_derniere_course": "days_since_last_run",
}


class DataValidationError(ValueError):
    """Erreur de validation avec message destiné à l'utilisateur."""


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    new_cols = []
    for c in df.columns:
        key = str(c).strip().lower().replace(" ", "_")
        new_cols.append(COLUMN_ALIASES.get(key, key))
    df.columns = new_cols
    return df


def coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Force les types numériques (les valeurs invalides deviennent NaN) et nettoie le texte."""
    df = df.copy()
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            # accepte "4,5" comme 4.5
            if df[col].dtype == object:
                df[col] = df[col].astype(str).str.replace(",", ".", regex=False).str.strip()
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["horse", "jockey", "trainer", "track", "going", "discipline", "race_id", "musique"]:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
    if "discipline" in df.columns:
        df["discipline"] = df["discipline"].str.lower()
    return df


def validate_runners(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Valide un DataFrame de partants. Renvoie (df_nettoyé, avertissements).
    Lève DataValidationError si les colonnes obligatoires manquent ou si le
    fichier est vide.
    """
    if df is None or df.empty:
        raise DataValidationError("Le fichier est vide : aucune ligne à analyser.")

    df = _normalize_columns(df)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataValidationError(
            "Colonnes obligatoires manquantes : "
            + ", ".join(missing)
            + f". Colonnes détectées : {', '.join(map(str, df.columns))}."
        )

    df = coerce_types(df)
    warnings: List[str] = []

    # Lignes sans nom de cheval
    before = len(df)
    df = df[df["horse"].notna() & (df["horse"].astype(str).str.len() > 0)]
    if len(df) < before:
        warnings.append(f"{before - len(df)} ligne(s) sans nom de cheval ignorée(s).")

    # Doublons cheval/course
    dup = df.duplicated(subset=["race_id", "horse"], keep="first")
    if dup.any():
        warnings.append(f"{int(dup.sum())} doublon(s) cheval/course supprimé(s).")
        df = df[~dup]

    # Courses avec un seul partant : non exploitables
    counts = df.groupby("race_id")["horse"].transform("count")
    solo = counts < 2
    if solo.any():
        warnings.append(
            f"{df.loc[solo, 'race_id'].nunique()} course(s) avec un seul partant ignorée(s)."
        )
        df = df[~solo]

    if df.empty:
        raise DataValidationError("Aucune course exploitable après nettoyage (au moins 2 partants requis).")

    absent = [c for c in RECOMMENDED_COLUMNS if c not in df.columns]
    if absent:
        warnings.append(
            "Colonnes recommandées absentes (le modèle utilisera des valeurs neutres) : "
            + ", ".join(absent)
        )

    if "odds" in df.columns:
        bad = (df["odds"] <= 1.0) & df["odds"].notna()
        if bad.any():
            warnings.append(f"{int(bad.sum())} cote(s) ≤ 1 ignorée(s) (cote décimale attendue, ex: 4.5).")
            df.loc[bad, "odds"] = np.nan

    return df.reset_index(drop=True), warnings


def load_runners_csv(source: Union[str, Path, bytes, io.IOBase]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Lit un CSV (chemin, bytes ou fichier téléversé Streamlit) en détectant le
    séparateur (`,` ou `;`) et l'encodage (UTF-8 puis Latin-1).
    """
    raw: bytes
    try:
        if isinstance(source, (str, Path)):
            raw = Path(source).read_bytes()
        elif isinstance(source, bytes):
            raw = source
        else:
            raw = source.read()
            if isinstance(raw, str):
                raw = raw.encode("utf-8")
    except OSError as exc:
        raise DataValidationError(f"Impossible de lire le fichier : {exc}") from exc

    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            text = raw.decode(encoding)
            df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
            return validate_runners(df)
        except DataValidationError:
            raise
        except Exception as exc:  # noqa: BLE001 — on veut un message propre
            last_error = exc
    raise DataValidationError(f"Format CSV non reconnu : {last_error}")


def sample_dataset_path() -> Path:
    """Chemin vers le jeu d'exemple embarqué dans le dépôt."""
    return Path(__file__).resolve().parents[2] / "data" / "sample_races.csv"


def load_sample_history() -> pd.DataFrame:
    """Charge le jeu d'exemple (historique avec résultats) ; DataFrame vide en cas d'échec."""
    try:
        df, _ = load_runners_csv(sample_dataset_path())
        return df
    except DataValidationError:
        return pd.DataFrame()
