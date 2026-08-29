"""Pipeline d'entrainement, SEPARE de l'application principale.

Usage :  python model/train.py [--real chemin.csv] [--races N] [--seed S]

Par defaut : genere le corpus synthetique (dataset.py) et entraine
deux modeles complementsaires :
  - reg : HistGradientBoostingRegressor sur la place normalisee (score de rang)
  - clf : HistGradientBoostingClassifier sur "top 5 oui/non" (probabilite)
Produit artifacts/model.joblib + artifacts/metrics.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.dataset import generate_synthetic_races, load_real_data  # noqa: E402
from model.features import FEATURES  # noqa: E402

ARTIFACTS = ROOT / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)
MODEL_PATH = ARTIFACTS / "model.joblib"
METRICS_PATH = ARTIFACTS / "metrics.json"


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def train(data: pd.DataFrame, seed: int = 42) -> dict:
    """Entraine les modeles sur un DataFrame au schema FEATURES."""
    X = data[FEATURES]
    y_place = data["place_norm"].to_numpy()
    y_top5 = data["top5"].to_numpy()

    X_tr, X_te, yp_tr, yp_te, y5_tr, y5_te = train_test_split(
        X, y_place, y_top5, test_size=0.2, random_state=seed
    )

    reg = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_depth=6,
        l2_regularization=1.0, random_state=seed,
    )
    reg.fit(X_tr, yp_tr)

    clf = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.08, max_depth=5,
        l2_regularization=1.0, random_state=seed,
    )
    clf.fit(X_tr, y5_tr)

    pred_place = reg.predict(X_te)
    prob_top5 = clf.predict_proba(X_te)[:, 1]

    rmse = float(np.sqrt(np.mean((pred_place - yp_te) ** 2)))
    rho = spearman(pred_place, yp_te)
    auc = float(roc_auc_score(y5_te, prob_top5))

    medians = {c: float(X[c].median()) for c in FEATURES}

    metrics = {
        "rmse_place_norm": round(rmse, 4),
        "spearman_place": round(rho, 4),
        "auc_top5": round(auc, 4),
        "baseline_auc": 0.5,
        "n_lignes": int(len(X)),
        "n_courses": int(data["race_id"].nunique()),
        "n_partants_moy": round(float(data.groupby("race_id").size().mean()), 1),
        "train_date": datetime.now(timezone.utc).isoformat(),
        "source": "dataset_synthetique" if "place_norm" in data.columns and "horse_name" in data.columns else "personnalise",
    }

    model_bundle = {
        "reg": reg,
        "clf": clf,
        "feature_names": FEATURES,
        "medians": medians,
        "metrics": metrics,
        "schema_version": 1,
    }
    import joblib

    joblib.dump(model_bundle, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrainement du modele de pronostic")
    parser.add_argument("--real", type=str, default=None,
                        help="Chemin d'un vrai historique CSV (schema FEATURES)")
    parser.add_argument("--races", type=int, default=3000, help="Nb de courses synthetiques")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.real:
        data = load_real_data(args.real)
        print(f"[train] Chargement du jeu reel: {len(data)} partants")
    else:
        print(f"[train] Generation du dataset synthetique ({args.races} courses)...")
        data = generate_synthetic_races(n_races=args.races, seed=args.seed)
        print(f"[train] {len(data)} lignes / {data['race_id'].nunique()} courses")

    metrics = train(data, seed=args.seed)
    print("[train] Metriques (jeu de test 20%):")
    print(f"  RMSE place normalisee : {metrics['rmse_place_norm']}")
    print(f"  Spearman (place)      : {metrics['spearman_place']}   (0 = hasard, 1 = parfait)")
    print(f"  AUC top-5             : {metrics['auc_top5']}         (0.5 = hasard)")
    print(f"[train] Modele sauvegarde: {MODEL_PATH} / {METRICS_PATH}")


if __name__ == "__main__":
    main()
