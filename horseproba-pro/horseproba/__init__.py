"""
horseproba — moteur de pronostics hippiques probabilistes.

Organisation du paquet :
- horseproba.data     : chargement (CSV / web), validation, données de démonstration
- horseproba.features : ingénierie des variables (forme, cotes, jockey, terrain...)
- horseproba.model    : modèle probabiliste (logit conditionnel / Bradley-Terry généralisé)
- horseproba.evaluate : métriques de qualité des probabilités (log-loss, Brier, calibration)
- horseproba.ui       : composants Streamlit réutilisables
"""

__version__ = "1.0.0"
