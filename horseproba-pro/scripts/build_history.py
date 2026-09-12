#!/usr/bin/env python
"""
Constitution d'un historique RÉEL de courses terminées à partir de la source
web publique PMU, au format CSV attendu par l'application (schéma
`horseproba/schema.py`, colonne `finish_position` renseignée).

Usage
-----
    # 30 derniers jours (hors aujourd'hui), toutes disciplines, vers data/history_pmu.csv
    python scripts/build_history.py --days 30

    # Période explicite, plat uniquement, fichier de sortie personnalisé
    python scripts/build_history.py --start 2024-09-01 --end 2024-09-30 \
        --discipline plat --out data/history_plat_sept.csv

    # Mise à jour incrémentale : ne re-télécharge pas les courses déjà présentes
    python scripts/build_history.py --days 7 --out data/history_pmu.csv

Le fichier produit se charge ensuite dans l'application via la barre latérale :
« Historique d'entraînement → Mon fichier CSV (résultats) ».

Usage responsable
-----------------
- La source est NON OFFICIELLE (endpoints JSON du site pmu.fr) : structure susceptible
  de changer, aucun engagement de disponibilité. Le script tolère les échecs course
  par course et journalise ce qui a été ignoré.
- Débit volontairement limité (~1,5 requête/s au maximum, voir `--pause`). Une journée
  complète ≈ 60-100 courses ≈ 2-4 minutes. Ne lancez pas de collectes massives en parallèle.
- Renseignez `--user-agent` avec un contact pour être identifiable.
- Vérifiez les conditions d'utilisation du site avant un usage intensif ou commercial.

Reprise / incrémental
---------------------
Si le fichier de sortie existe, les `race_id` déjà présents sont ignorés et les
nouvelles lignes sont ajoutées. Une sauvegarde intermédiaire est écrite après
chaque journée, de sorte qu'une interruption (Ctrl-C, coupure réseau) ne perd
que la journée en cours.

Enrichissement post-collecte
----------------------------
`days_since_last_run` n'est pas fourni par la source ; il est RECALCULÉ ici à
partir des courses précédentes du même cheval présentes dans l'historique
(première sortie connue -> NaN, valeur neutre pour le modèle).
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd

# Permet d'exécuter le script depuis la racine du dépôt sans installation du paquet
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from horseproba.data import pmu  # noqa: E402
from horseproba.schema import ALL_COLUMNS  # noqa: E402

LOG = logging.getLogger("build_history")

OUTPUT_COLUMNS = [c for c in ALL_COLUMNS] + ["odds_probable", "odds_direct", "num_pmu", "incident"]


# --------------------------------------------------------------------------- #
# Arguments
# --------------------------------------------------------------------------- #
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Construit un historique CSV de courses PMU terminées.")
    period = p.add_mutually_exclusive_group()
    period.add_argument("--days", type=int, help="Nombre de jours en arrière à partir d'hier (ex. 30).")
    period.add_argument("--start", type=str, help="Date de début AAAA-MM-JJ (avec --end).")
    p.add_argument("--end", type=str, help="Date de fin AAAA-MM-JJ (incluse). Défaut : hier.")
    p.add_argument("--discipline", choices=["plat", "trot", "obstacle", "all"], default="all", help="Filtrer une discipline.")
    p.add_argument("--out", type=Path, default=Path("data/history_pmu.csv"), help="Fichier CSV de sortie.")
    p.add_argument("--pause", type=float, default=0.7, help="Pause (s) entre deux courses, en plus du rate-limit interne.")
    p.add_argument("--max-races", type=int, default=None, help="Arrêter après N courses collectées (tests).")
    p.add_argument("--user-agent", type=str, default="HorseProbaApp/1.0 (history builder; contact: unset)", help="User-Agent HTTP identifiable.")
    p.add_argument("--no-resume", action="store_true", help="Ignorer le fichier existant et repartir de zéro.")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    yesterday = date.today() - timedelta(days=1)
    if args.days is not None:
        args.end_date = yesterday
        args.start_date = yesterday - timedelta(days=max(args.days, 1) - 1)
    elif args.start:
        args.start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
        args.end_date = datetime.strptime(args.end, "%Y-%m-%d").date() if args.end else yesterday
    else:
        args.end_date = yesterday
        args.start_date = yesterday - timedelta(days=13)  # 14 jours par défaut
    if args.start_date > args.end_date:
        p.error("La date de début doit précéder la date de fin.")
    if args.end_date >= date.today():
        LOG.warning("Les courses du jour ne sont pas toutes terminées : la fin est ramenée à hier.")
        args.end_date = yesterday
    return args


# --------------------------------------------------------------------------- #
# Collecte
# --------------------------------------------------------------------------- #
def load_existing(path: Path, resume: bool) -> pd.DataFrame:
    if not resume or not path.exists():
        return pd.DataFrame(columns=OUTPUT_COLUMNS)
    try:
        df = pd.read_csv(path)
        LOG.info("Fichier existant : %d lignes, %d courses (mode incrémental).", len(df), df["race_id"].nunique())
        return df
    except Exception as exc:  # noqa: BLE001
        LOG.warning("Fichier existant illisible (%s) : il sera écrasé.", exc)
        return pd.DataFrame(columns=OUTPUT_COLUMNS)


def save(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [c for c in OUTPUT_COLUMNS if c in df.columns] + [c for c in df.columns if c not in OUTPUT_COLUMNS]
    df[cols].to_csv(path, index=False)


def collect_day(day: date, discipline: str, known: set, pause: float, user_agent: str, budget: Optional[int]) -> List[pd.DataFrame]:
    """Collecte toutes les courses terminées d'une journée. Ne lève jamais : journalise et continue."""
    frames: List[pd.DataFrame] = []
    try:
        refs = pmu.fetch_program(day, user_agent)
    except pmu.PMUFetchError as exc:
        LOG.warning("%s : programme indisponible (%s).", day, exc)
        return frames
    except Exception as exc:  # noqa: BLE001
        LOG.error("%s : erreur inattendue sur le programme (%s).", day, exc)
        return frames

    if discipline != "all":
        refs = [r for r in refs if r.discipline == discipline]
    LOG.info("%s : %d course(s) au programme.", day, len(refs))

    for ref in refs:
        if budget is not None and len(frames) >= budget:
            break
        if ref.race_id in known:
            LOG.debug("%s déjà présent, ignoré.", ref.race_id)
            continue
        try:
            df = pmu.fetch_finished_race(ref, user_agent)
            frames.append(df)
            LOG.info("  ✔ %s — %d partants, gagnant : %s", ref.race_id, len(df), df.loc[df["finish_position"] == 1, "horse"].iloc[0])
        except pmu.PMUFetchError as exc:
            LOG.warning("  ✘ %s ignorée : %s", ref.race_id, exc)
        except Exception as exc:  # noqa: BLE001
            LOG.error("  ✘ %s erreur inattendue : %s", ref.race_id, exc)
        time.sleep(max(pause, 0.0))
    return frames


# --------------------------------------------------------------------------- #
# Enrichissement
# --------------------------------------------------------------------------- #
def recompute_days_since_last_run(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recalcule `days_since_last_run` à partir des sorties du même cheval dans
    l'historique (la source ne le fournit pas). Reste NaN pour la première
    apparition connue — le modèle traite NaN comme neutre.
    """
    out = df.copy()
    out["_d"] = pd.to_datetime(out["race_date"], errors="coerce")
    out["_h"] = out["horse"].astype(str).str.strip().str.lower()
    out = out.sort_values(["_h", "_d"])
    prev = out.groupby("_h")["_d"].shift(1)
    computed = (out["_d"] - prev).dt.days
    out["days_since_last_run"] = out["days_since_last_run"].where(out["days_since_last_run"].notna(), computed)
    return out.drop(columns=["_d", "_h"]).sort_values(["race_date", "race_id", "finish_position"]).reset_index(drop=True)


def sanity_report(df: pd.DataFrame) -> None:
    n_races = df["race_id"].nunique()
    winners = df[df["finish_position"] == 1].groupby("race_id").size()
    LOG.info("─" * 60)
    LOG.info("Historique : %d courses, %d partants.", n_races, len(df))
    LOG.info("Courses avec exactement 1 gagnant : %d / %d", int((winners == 1).sum()), n_races)
    if "odds" in df.columns:
        LOG.info("Cotes renseignées : %.0f %%", 100 * df["odds"].notna().mean())
    if "musique" in df.columns:
        LOG.info("Musique renseignée : %.0f %%", 100 * (df["musique"].fillna("").astype(str).str.len() > 0).mean())
    LOG.info("Disciplines : %s", df["discipline"].value_counts().to_dict())
    LOG.info("Période : %s → %s", df["race_date"].min(), df["race_date"].max())


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s", datefmt="%H:%M:%S")

    LOG.info("Collecte PMU du %s au %s (%s) → %s", args.start_date, args.end_date, args.discipline, args.out)
    existing = load_existing(args.out, resume=not args.no_resume)
    known = set(existing["race_id"].astype(str)) if not existing.empty else set()

    collected = 0
    day = args.start_date
    try:
        while day <= args.end_date:
            budget = None if args.max_races is None else max(args.max_races - collected, 0)
            if budget == 0:
                break
            frames = collect_day(day, args.discipline, known, args.pause, args.user_agent, budget)
            if frames:
                new = pd.concat(frames, ignore_index=True)
                existing = pd.concat([existing, new], ignore_index=True)
                known.update(new["race_id"].astype(str))
                collected += new["race_id"].nunique()
                save(existing, args.out)  # sauvegarde intermédiaire
                LOG.info("%s : +%d course(s) enregistrée(s) (total collecté : %d).", day, new["race_id"].nunique(), collected)
            day += timedelta(days=1)
    except KeyboardInterrupt:
        LOG.warning("Interruption : les journées complètes déjà collectées sont sauvegardées.")

    if existing.empty:
        LOG.error("Aucune course collectée. Vérifiez la connexion, la période, ou la disponibilité de la source.")
        return 1

    existing = recompute_days_since_last_run(existing)
    save(existing, args.out)
    sanity_report(existing)
    LOG.info("Fichier écrit : %s", args.out.resolve())
    LOG.info("Chargez-le dans l'application : barre latérale → « Mon fichier CSV (résultats) ».")
    return 0


if __name__ == "__main__":
    sys.exit(main())
