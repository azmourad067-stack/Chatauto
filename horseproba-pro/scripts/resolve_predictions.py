#!/usr/bin/env python
"""
Rapproche le journal des pronostics (`data/predictions_log.csv`, produit par
`scripts/log_predictions.py`) des RÉSULTATS RÉELS une fois connus (présents
dans `data/history_pmu.csv` après passage de `scripts/build_history.py`).

C'est la deuxième moitié de la boucle d'auto-apprentissage : sans cette
étape, on aurait des pronostics enregistrés mais jamais confrontés à la
réalité, ce qui rend impossible toute mesure honnête de progression.

Chaque ligne du journal dont `is_winner` est encore vide est recherchée dans
l'historique par (race_id, horse). Si un résultat existe, `is_winner` est
renseigné (1 si arrivée 1ʳᵉ, 0 sinon) et `resolved_at` horodaté. Les lignes
déjà résolues ne sont jamais modifiées (traçabilité : on ne réécrit pas le
passé). Les lignes sans résultat disponible (course pas encore courue, ou pas
encore collectée par `build_history.py`) restent en attente — pas une erreur.

Usage
-----
    python scripts/resolve_predictions.py
    python scripts/resolve_predictions.py --predictions data/predictions_log.csv --history data/history_pmu.csv
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

LOG = logging.getLogger("resolve_predictions")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rapproche les pronostics journalisés des résultats réels connus.")
    p.add_argument("--predictions", type=Path, default=Path("data/predictions_log.csv"))
    p.add_argument("--history", type=Path, default=Path("data/history_pmu.csv"))
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def resolve(predictions: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Fonction pure (testable sans I/O) : renvoie une COPIE mise à jour de `predictions`."""
    out = predictions.copy()
    if out.empty:
        return out
    if "is_winner" not in out.columns:
        out["is_winner"] = pd.NA
    if "resolved_at" not in out.columns:
        out["resolved_at"] = pd.NA

    pending_mask = out["is_winner"].isna()
    if not pending_mask.any():
        return out

    hist = history.copy()
    hist["finish_position"] = pd.to_numeric(hist.get("finish_position"), errors="coerce")
    hist = hist.dropna(subset=["finish_position"])
    results = hist.drop_duplicates(subset=["race_id", "horse"]).set_index(["race_id", "horse"])["finish_position"]

    idx = pd.MultiIndex.from_frame(out.loc[pending_mask, ["race_id", "horse"]])
    matched = results.reindex(idx)
    found = matched.notna().to_numpy()

    now = datetime.now(timezone.utc).isoformat()
    pending_idx = out.index[pending_mask]
    resolved_idx = pending_idx[found]
    if len(resolved_idx) == 0:
        return out

    out.loc[resolved_idx, "is_winner"] = (matched[found].to_numpy() == 1).astype(int)
    out.loc[resolved_idx, "resolved_at"] = now
    return out


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S",
    )

    if not args.predictions.exists():
        LOG.info("Aucun journal de pronostics (%s) : rien à résoudre.", args.predictions)
        return 0
    if not args.history.exists():
        LOG.warning("Historique introuvable (%s) : impossible de résoudre quoi que ce soit pour l'instant.", args.history)
        return 0

    try:
        predictions = pd.read_csv(args.predictions)
    except Exception as exc:  # noqa: BLE001
        LOG.error("Journal de pronostics illisible (%s).", exc)
        return 1
    history = pd.read_csv(args.history)

    n_pending_before = int(predictions.get("is_winner", pd.Series(dtype=float)).isna().sum())
    out = resolve(predictions, history)
    n_pending_after = int(out["is_winner"].isna().sum())
    n_resolved = n_pending_before - n_pending_after

    if n_resolved > 0:
        out.to_csv(args.predictions, index=False)
        LOG.info("%d pronostic(s) résolu(s). %d encore en attente de résultat.", n_resolved, n_pending_after)
    else:
        LOG.info("Aucun nouveau résultat disponible pour l'instant (%d en attente).", n_pending_after)
    return 0


if __name__ == "__main__":
    sys.exit(main())
