#!/usr/bin/env python
"""
Enregistre les pronostics du modèle de PRODUCTION pour les courses d'une
journée (par défaut aujourd'hui), AVANT qu'elles ne soient courues.

Rôle dans la boucle d'auto-apprentissage
=========================================
`scripts/train_model.py` évalue le modèle par BACKTEST (rejoue le passé) :
c'est utile mais ce n'est pas la même chose que vérifier ses performances
RÉELLES sur des pronostics publiés à l'avance. Ce script ferme cette boucle :

    1. AVANT les courses (ce script) : on prédit et on enregistre
       (race_id, cheval, p_win, cote au moment de la prédiction, version du
       modèle utilisé) dans `data/predictions_log.csv`.
    2. APRÈS les courses (`scripts/resolve_predictions.py`, une fois que
       `scripts/build_history.py` a collecté le résultat) : on rapproche ces
       prédictions des résultats réels (`is_winner`).
    3. Le résultat cumulé est visible dans l'onglet « Auto-apprentissage » de
       l'application (`evaluate.summarize_live_predictions`) : une courbe de
       performance RÉELLE, pas seulement rétrospective.

Ce script est prévu pour tourner CHAQUE JOUR (voir la GitHub Action
`.github/workflows/learning_loop.yml`), typiquement le matin, avant les
premières courses. Il est volontairement TOLÉRANT AUX PANNES : une course qui
échoue à se charger est journalisée et ignorée, le script continue.

Déduplication
-------------
Chaque (race_id, horse) n'est enregistré QU'UNE SEULE FOIS : si le script est
relancé le même jour (ex. reprise après coupure), les prédictions déjà
journalisées ne sont PAS écrasées. On veut comparer les résultats au PREMIER
pronostic publié, pas au dernier recalcul juste avant le départ (qui aurait un
avantage déloyal : des cotes plus fraîches, potentiellement plus informatives).

Usage
-----
    python scripts/log_predictions.py                      # aujourd'hui
    python scripts/log_predictions.py --date 2024-09-15     # date explicite
    python scripts/log_predictions.py --discipline plat
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from horseproba import registry  # noqa: E402
from horseproba.data import pmu  # noqa: E402
from horseproba.model import flag_outsiders  # noqa: E402

LOG = logging.getLogger("log_predictions")

LOG_COLUMNS = [
    "race_id", "horse", "race_date", "track", "discipline", "predicted_at",
    "model_saved_at", "odds", "p_win", "p_place", "market_p", "value",
    "is_outsider_pick", "is_winner", "resolved_at",
]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Journalise les pronostics du modèle de production pour une journée.")
    p.add_argument("--date", type=str, default=None, help="Date AAAA-MM-JJ (défaut : aujourd'hui).")
    p.add_argument("--discipline", choices=["plat", "trot", "obstacle", "all"], default="all")
    p.add_argument("--out", type=Path, default=Path("data/predictions_log.csv"))
    p.add_argument("--registry", type=Path, default=registry.DEFAULT_REGISTRY_DIR)
    p.add_argument("--pause", type=float, default=0.7)
    p.add_argument("--user-agent", type=str, default="HorseProbaApp/1.0 (prediction logger; contact: unset)")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def _load_existing(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=LOG_COLUMNS)
    try:
        return pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Journal existant illisible (%s) : reconstruction depuis rien.", exc)
        return pd.DataFrame(columns=LOG_COLUMNS)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S",
    )

    day = datetime.strptime(args.date, "%Y-%m-%d").date() if args.date else date.today()

    model = registry.load_model(registry.production_dir(args.registry))
    if model is None:
        LOG.error(
            "Aucun modèle de production trouvé dans %s. Lancez d'abord scripts/train_model.py "
            "(un modèle 'à froid', avec les coefficients a priori, n'est volontairement pas journalisé : "
            "il n'a rien appris et fausserait le suivi de performance).",
            registry.production_dir(args.registry),
        )
        return 1
    model_saved_at = (registry.load_metadata(registry.production_dir(args.registry)) or {}).get("saved_at", "")

    try:
        refs = pmu.fetch_program(day, args.user_agent)
    except pmu.PMUFetchError as exc:
        LOG.error("Programme indisponible pour %s : %s", day, exc)
        return 1
    if args.discipline != "all":
        refs = [r for r in refs if r.discipline == args.discipline]
    LOG.info("%s : %d course(s) au programme.", day, len(refs))

    existing = _load_existing(args.out)
    known = set(zip(existing.get("race_id", []), existing.get("horse", [])))
    now = datetime.now(timezone.utc).isoformat()
    new_rows: List[dict] = []

    for ref in refs:
        try:
            runners = pmu.fetch_runners(ref, args.user_agent)
        except pmu.PMUFetchError as exc:
            LOG.warning("  ✘ %s ignorée (partants indisponibles) : %s", ref.race_id, exc)
            continue
        except Exception as exc:  # noqa: BLE001
            LOG.error("  ✘ %s erreur inattendue : %s", ref.race_id, exc)
            continue

        to_log = runners[~runners.apply(lambda r: (r["race_id"], r["horse"]) in known, axis=1)]
        if to_log.empty:
            LOG.debug("%s : déjà journalisée, ignorée.", ref.race_id)
            time.sleep(max(args.pause, 0.0))
            continue
        try:
            pred = model.predict(runners)
        except Exception as exc:  # noqa: BLE001
            LOG.error("  ✘ %s : échec de la prédiction (%s).", ref.race_id, exc, exc_info=True)
            continue
        pred["is_outsider_pick"] = flag_outsiders(pred)
        pred = pred[pred["horse"].isin(to_log["horse"])]

        for _, row in pred.iterrows():
            new_rows.append({
                "race_id": row["race_id"], "horse": row["horse"], "race_date": ref.day.isoformat(),
                "track": ref.track, "discipline": ref.discipline, "predicted_at": now,
                "model_saved_at": model_saved_at, "odds": row.get("odds"), "p_win": row.get("p_win"),
                "p_place": row.get("p_place"), "market_p": row.get("market_p"), "value": row.get("value"),
                "is_outsider_pick": bool(row.get("is_outsider_pick", False)),
                "is_winner": pd.NA, "resolved_at": pd.NA,
            })
        LOG.info("  ✔ %s — %d partants journalisés.", ref.race_id, len(pred))
        time.sleep(max(args.pause, 0.0))

    if not new_rows:
        LOG.info("Rien de nouveau à journaliser.")
        return 0

    out = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True) if not existing.empty else pd.DataFrame(new_rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    LOG.info("Journal mis à jour : %d nouvelles lignes (%d au total) -> %s", len(new_rows), len(out), args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
