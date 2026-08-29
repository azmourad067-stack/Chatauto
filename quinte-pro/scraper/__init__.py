"""Module de recherche/collecte web en temps reel (donnees hippiques).

Deux couches, a ne pas confondre avec le modele de prediction :
- adaptateurs de sources (open-pmu-api, moteur de recherche, sites officiels)
- moteur de fusion qui normalise, met en cache et gere les pannes.
"""
from scraper.engine import collect_race_data  # noqa: F401
