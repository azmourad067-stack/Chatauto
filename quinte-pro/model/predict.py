"""Inference : transforme une course + partants en classement des 5 premiers.

Combinaison scraping + modele :
  1. le scraper enrichit chaque partant (donnees temps reel) ;
  2. build_features_df construit les features, medianes d'entrainement si
     donnees manquantes ;
  3. le regresseur fournit le score de rang (plus petit = mieux) -> ORDRE,
     le classifieur fournit la probabilite de top-5 -> AFFICHAGE.
  Les probabilites sont des sorties de modeles, PAS des garanties.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.features import FEATURES, build_features_df  # noqa: E402

MODEL_PATH = ROOT / "artifacts" / "model.joblib"
METRICS_PATH = ROOT / "artifacts" / "metrics.json"


def load_model(path: Path = MODEL_PATH) -> dict:
    import joblib

    bundle = joblib.load(path)
    if bundle["feature_names"] != FEATURES:
        raise RuntimeError("Schema de features du modele != schema du code (reentrainez).")
    return bundle


def load_metrics(path: Path = METRICS_PATH) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {}


def predict_top5(race: dict, partants: list[dict], bundle: dict | None = None) -> list[dict]:
    """Retourne le classement complet trie + top 5 enrichi.

    race     : {hippodrome, discipline, distance, date?}
    partants : liste de dicts enrichis par engine.collect_race_data (ou saisie brute)
    """
    if bundle is None:
        bundle = load_model()

    df = build_features_df(race, partants, bundle["medians"])
    score_reg = bundle["reg"].predict(df)                      # plus petit = mieux
    proba_top5 = bundle["clf"].predict_proba(df)[:, 1]         # P(top5)

    # probabilite normalisee (heuristique softmax sur -score) pour l'affichage
    temp = 1.0
    z = np.exp(-score_reg / temp)
    proba_norm = z / z.sum()

    order = np.argsort(score_reg)
    ranked = []
    for i in order:
        p = partants[i]
        ranked.append({
            "rang": int(np.where(order == i)[0][0]) + 1,
            "num": int(p.get("num") or i + 1),
            "cheval": p.get("horse", "?"),
            "driver": p.get("driver", ""),
            "score": round(float(score_reg[i]), 4),
            "proba_top5": round(float(proba_top5[i]), 4),
            "proba_norm": round(float(proba_norm[i]), 4),
            "sentiment": round(float(p.get("web_sentiment") or 0.0), 2),
            "sources": p.get("data_sources", []),
        })

    top5 = ranked[:5]
    return {"classement_complet": ranked, "top5": top5}
