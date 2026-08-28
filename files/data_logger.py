"""Gestion de l'export CSV des pronostics pour archivage manuel par l'utilisateur.

Streamlit Community Cloud ne garantit pas de stockage persistant : on ne
tente donc pas d'écrire un fichier serveur. À la place, chaque analyse peut
être téléchargée en CSV par l'utilisateur, avec une colonne à compléter
manuellement après la course. En accumulant ces fichiers au fil du temps,
l'utilisateur se constitue une base historique qui pourra, plus tard, servir
à calibrer les poids du score intrinsèque de façon empirique.
"""

import pandas as pd
from datetime import datetime


def preparer_export(df: pd.DataFrame, infos_course: dict) -> pd.DataFrame:
    """Ajoute les métadonnées de la course au tableau de résultats avant export."""
    export = df.copy()
    export.insert(0, "Date d'analyse", datetime.now().strftime("%Y-%m-%d %H:%M"))
    export.insert(1, "Distance course (m)", infos_course.get("distance"))
    export.insert(2, "Terrain du jour", infos_course.get("terrain"))
    export.insert(3, "Résultat réel (à compléter)", "")
    return export


def dataframe_vers_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convertit un DataFrame en bytes CSV téléchargeables.

    Séparateur ';' et décimales ',' pour une ouverture directe correcte
    dans Excel FR ; encodage utf-8-sig pour les accents.
    """
    return df.to_csv(index=False, sep=";", decimal=",").encode("utf-8-sig")
