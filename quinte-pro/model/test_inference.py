"""Smoke test : verifie que le pipeline modele (charge + predit) tourne.

Usage : python model/test_inference.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.predict import load_metrics, load_model, predict_top5  # noqa: E402


def build_fake_race() -> tuple[dict, list[dict]]:
    race = {"hippodrome": "Vincennes", "discipline": "trot_attelle", "distance": 2700}
    noms = [
        ("JASMIN DU PONT", "M. ABRIVARD", 480000),
        ("HOKKAIDO JIEL", "F. NIVARD", 610000),
        ("IDEAL DE HOUELLE", "A. BARRIER", 210000),
        ("GOOD GAME", "E. RAFFIN", 390000),
        ("FLASH DE COSSE", "D. THOMAIN", 150000),
        ("HAPPY VALLEY", "G. GELORMINI", 95000),
        ("INDIANA JONES", "B. GOOP", 76000),
        ("KEEP GOING", "P. VERCRUYSSE", 120000),
        ("JAGUAR DU RIB", "C. MEGISSIER", 45000),
        ("GALANT DE L'ITON", "L. ABRIVARD", 30000),
        ("HERMES DU RIB", "J.-M. BAZIRE", 88000),
        ("IVANHOE JISCE", "F. OUVRIE", 250000),
    ]
    partants = []
    for i, (cheval, driver, gains) in enumerate(noms, start=1):
        partants.append({
            "num": i, "horse": cheval, "driver": driver, "age": 6 + (i % 4),
            "gains": gains,
            "musique": "1p 2p 3p 1p (25) 4p" if i <= 4 else "8p 9p 0a 7p (25) 5p",
            "distance_pref": 2650.0 + 50 * (i % 5),
            "web_sentiment": 0.4 if i <= 4 else -0.2,
            "data_sources": ["saisie utilisateur", "recherche web"],
            "horse_win_rate": 0.22 - 0.015 * i,
            "driver_win_rate": 0.18 - 0.012 * i,
        })
    return race, partants


def main() -> None:
    bundle = load_model()
    metrics = load_metrics()
    race, partants = build_fake_race()
    out = predict_top5(race, partants, bundle)

    print("=== Modeles charges ===")
    print(f"  metriques entrainement: RMSE={metrics.get('rmse_place_norm')}, "
          f"Spearman={metrics.get('spearman_place')}, AUC_top5={metrics.get('auc_top5')}")
    print(f"  features attendues: {len(bundle['feature_names'])}")
    print("\n=== Pronostic (course factice, 12 partants) ===")
    for row in out["top5"]:
        print(f"  {row['rang']}. {row['cheval']:<22} (driver: {row['driver']:<16}) "
              f"proba_top5={row['proba_top5']:.1%}  score={row['score']}")
    assert len(out["top5"]) == 5, "Le top 5 doit contenir 5 chevaux"
    assert len(out["classement_complet"]) == len(partants), "Le classement doit couvrir tous les partants"
    print("\n[test_inference] OK : pipeline entrainement -> inference fonctionnel.")


if __name__ == "__main__":
    main()
