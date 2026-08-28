"""Modèles et constantes pour l'outil de pronostic hippique."""

import pandas as pd

COL_NOM = "Nom du cheval"
COL_FORME = "Forme (podiums /5 dernières courses)"
COL_POIDS = "Poids porté (kg)"
COL_DISTANCE = "Aptitude distance"
COL_TERRAIN = "Aptitude terrain"
COL_JOCKEY = "Jockey / Entraîneur"
COL_COTE = "Cote probable"

OPTIONS_APTITUDE = ["Favorable", "Neutre", "Défavorable"]
OPTIONS_JOCKEY = ["Fort", "Moyen", "Faible"]

COLONNES = [
    COL_NOM,
    COL_FORME,
    COL_POIDS,
    COL_DISTANCE,
    COL_TERRAIN,
    COL_JOCKEY,
    COL_COTE,
]


def dataframe_vide(nb_chevaux: int) -> pd.DataFrame:
    """Crée un DataFrame pré-rempli de valeurs par défaut pour n chevaux.

    Sert de base au tableau éditable (st.data_editor) dans l'app Streamlit.
    """
    data = {
        COL_NOM: [f"Cheval {i + 1}" for i in range(nb_chevaux)],
        COL_FORME: [2 for _ in range(nb_chevaux)],
        COL_POIDS: [58.0 for _ in range(nb_chevaux)],
        COL_DISTANCE: ["Neutre" for _ in range(nb_chevaux)],
        COL_TERRAIN: ["Neutre" for _ in range(nb_chevaux)],
        COL_JOCKEY: ["Moyen" for _ in range(nb_chevaux)],
        COL_COTE: [10.0 for _ in range(nb_chevaux)],
    }
    return pd.DataFrame(data)
