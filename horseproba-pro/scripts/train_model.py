#!/usr/bin/env python
"""
Ré-entraîne le modèle de pronostic sur l'historique disponible et, si les
résultats le justifient, PROMEUT le nouveau modèle en production.

C'est le cœur de la boucle d'AUTO-APPRENTISSAGE : ce script est conçu pour
être exécuté régulièrement (manuellement, ou automatiquement via la GitHub
Action `.github/workflows/learning_loop.yml`, après chaque collecte de
nouveaux résultats par `scripts/build_history.py`).

Pourquoi un garde-fou de promotion (« challenger vs champion ») ?
===================================================================
Ré-entraîner régulièrement ne suffit pas : rien ne garantit qu'un nouveau
modèle, entraîné sur un historique légèrement différent, soit MEILLEUR que
celui actuellement en production (variance d'échantillonnage, mauvaise passe
temporaire...). Écraser aveuglément la production à chaque exécution ferait
de l'auto-apprentissage une marche aléatoire, pas un progrès garanti.

Protocole retenu
-----------------
1. Le CANDIDAT est évalué par un BACKTEST WALK-FORWARD complet
   (`evaluate.backtest`, plusieurs blocs hors-échantillon successifs sur
   TOUT l'historique disponible aujourd'hui) → `candidate_logloss`.
   Le walk-forward est préféré à un simple split train/holdout : plusieurs
   blocs hors-échantillon réduisent la variance de l'estimation par rapport
   à un seul découpage, particulièrement important avec un historique encore
   modeste.
2. Le modèle de PRODUCTION actuel n'est PAS ré-évalué sur les nouvelles
   données (cela biaiserait la comparaison en sa faveur s'il a déjà été
   entraîné dessus lors d'une exécution précédente — piège classique de
   « fuite » qu'on évite ici délibérément). On compare plutôt au log-loss
   backtest qui avait été ENREGISTRÉ au moment de SA PROPRE promotion
   (`meta.json["metrics"]["backtest_logloss"]`), sur les données disponibles
   à l'époque.
3. PROMOTION si :
     a. le candidat bat le repère uniforme (`candidate_logloss < logloss_uniform`)
        — condition minimale de bon sens ;
     ET b. pas de production existante, OU le candidat n'est pas
        significativement pire que le score enregistré de la production
        (`candidate_logloss <= score_production × (1 + tolérance)`).
   Sinon, la production actuelle est CONSERVÉE et l'échec est journalisé
   (pas une exception : un refus de promotion est un résultat normal du
   garde-fou, pas une erreur du pipeline).
4. Si promu, le modèle sauvegardé est ré-entraîné une dernière fois sur TOUT
   l'historique (le backtest ne sert qu'à la DÉCISION, pas à produire le
   modèle final : on ne veut pas priver la production de ses courses les
   plus récentes).
5. Optimisation : si l'historique n'a PAS grandi depuis la dernière
   exécution (même nombre de courses), le script ne fait rien (`--force`
   pour outrepasser) — utile pour un cron quotidien qui peut tomber un jour
   sans nouvelles courses terminées.
6. Dans tous les cas (promotion, refus, ou historique inchangé), une ligne
   est ajoutée à `models/training_log.csv` : c'est ce journal qui permet de
   visualiser la progression du modèle dans le temps (onglet
   « Auto-apprentissage » de l'application).

Usage
-----
    python scripts/train_model.py
    python scripts/train_model.py --history data/history_pmu.csv --gbm-weight 0.35
    python scripts/train_model.py --force            # ré-entraîne et promeut même sans nouvelles données
    python scripts/train_model.py --tolerance 0.08   # garde-fou plus permissif
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

from horseproba import evaluate, registry  # noqa: E402
from horseproba.ensemble import EnsembleModel  # noqa: E402

LOG = logging.getLogger("train_model")

MIN_RACES_FOR_BACKTEST = 20


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Ré-entraîne et (si justifié) promeut le modèle de production.")
    p.add_argument("--history", type=Path, default=Path("data/history_pmu.csv"), help="Historique CSV d'entrée.")
    p.add_argument("--registry", type=Path, default=registry.DEFAULT_REGISTRY_DIR, help="Répertoire du registre de modèles.")
    p.add_argument("--gbm-weight", type=float, default=0.35, help="Poids du GBM dans l'ensemble (0 = logit seul).")
    p.add_argument("--l2", type=float, default=1.0, help="Régularisation L2 du logit conditionnel.")
    p.add_argument("--n-folds", type=int, default=4, help="Nombre de blocs du backtest walk-forward de validation.")
    p.add_argument("--tolerance", type=float, default=0.05, help="Tolérance relative du garde-fou (0.05 = 5%%).")
    p.add_argument("--force", action="store_true", help="Ré-entraîne/promeut même sans nouvelles données, ignore l'échec du garde-fou.")
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def _fresh_model(gbm_weight: float, l2: float) -> EnsembleModel:
    from horseproba.model import ConditionalLogitModel
    from horseproba.model_gbm import GradientBoostingRanker

    return EnsembleModel(logit=ConditionalLogitModel(l2=l2), gbm=GradientBoostingRanker(), gbm_weight=gbm_weight)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.history.exists():
        LOG.error("Historique introuvable : %s (lancez d'abord scripts/build_history.py)", args.history)
        return 1

    history = pd.read_csv(args.history)
    history["finish_position"] = pd.to_numeric(history.get("finish_position"), errors="coerce")
    history = history.dropna(subset=["finish_position"])
    n_races_total = int(history["race_id"].nunique())
    LOG.info("Historique chargé : %d courses, %d lignes.", n_races_total, len(history))

    prod_dir = registry.production_dir(args.registry)
    existing_meta = registry.load_metadata(prod_dir)

    if not args.force and existing_meta is not None and (existing_meta.get("metrics") or {}).get("n_races_total") == n_races_total:
        LOG.info("Historique inchangé depuis le dernier entraînement (%d courses) : rien à faire.", n_races_total)
        registry.append_training_log(args.registry, {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "n_races_total": n_races_total,
            "promoted": False,
            "reason": "historique inchangé depuis le dernier entraînement (aucune nouvelle course)",
        })
        return 0

    timestamp = datetime.now(timezone.utc).isoformat()
    row: dict = {"timestamp": timestamp, "n_races_total": n_races_total,
                 "gbm_weight_configured": args.gbm_weight, "l2": args.l2}

    try:
        candidate_logloss = uniform_logloss = market_logloss = None
        if n_races_total >= MIN_RACES_FOR_BACKTEST:
            report = evaluate.backtest(
                history, n_folds=args.n_folds,
                model_factory=lambda: _fresh_model(args.gbm_weight, args.l2),
            )
            candidate_logloss = report.logloss_model
            uniform_logloss = report.logloss_uniform
            market_logloss = report.logloss_market
            LOG.info(
                "Backtest candidat : log-loss=%.4f (uniforme=%.4f, marché=%s) sur %d courses.",
                candidate_logloss, uniform_logloss,
                f"{market_logloss:.4f}" if market_logloss is not None else "n/d", report.n_races,
            )
        else:
            LOG.warning("Historique trop court (< %d courses) pour un backtest fiable.", MIN_RACES_FOR_BACKTEST)

        row.update({
            "candidate_logloss_backtest": candidate_logloss,
            "uniform_logloss_backtest": uniform_logloss,
            "market_logloss_backtest": market_logloss,
        })

        production_score = None
        if existing_meta is not None:
            production_score = (existing_meta.get("metrics") or {}).get("backtest_logloss")
        row["production_logloss_reference"] = production_score
        if args.force:
            promote, reason = True, "forcé (--force)"
        elif candidate_logloss is None:
            promote, reason = (existing_meta is None), "backtest impossible (historique trop court) ; promotion par défaut si aucune production n'existe encore"
        elif candidate_logloss >= uniform_logloss:
            promote, reason = False, "le candidat ne bat pas le repère uniforme au backtest"
        elif production_score is None:
            promote, reason = True, "aucune production existante à battre, et le candidat bat l'uniforme"
        elif candidate_logloss <= production_score * (1 + args.tolerance):
            promote, reason = True, f"candidat ≤ référence production × (1+{args.tolerance:.0%})"
        else:
            promote, reason = False, "le candidat est significativement pire que la référence de la production actuelle"

        LOG.info("Décision de promotion : %s (%s)", "OUI" if promote else "NON", reason)
        row["promoted"] = promote
        row["reason"] = reason

        if promote:
            final_model = _fresh_model(args.gbm_weight, args.l2)
            fit_res = final_model.fit(history)
            metrics = {
                "backtest_logloss": candidate_logloss,
                "backtest_logloss_uniform": uniform_logloss,
                "backtest_logloss_market": market_logloss,
                "promotion_reason": reason,
                "n_races_total": n_races_total,
            }
            registry.save_model(final_model, prod_dir, metrics=metrics)
            row["logit_pseudo_r2_full"] = fit_res.logit_pseudo_r2
            row["gbm_fitted_full"] = fit_res.gbm_fitted
            LOG.info("Nouveau modèle promu en production (%s).", prod_dir)
        else:
            LOG.info("Production conservée telle quelle (%s).", prod_dir)

        registry.append_training_log(args.registry, row)
        return 0

    except Exception as exc:  # noqa: BLE001 — pipeline planifié : ne jamais lever, toujours journaliser
        LOG.error("Échec de l'entraînement : %s", exc, exc_info=True)
        row["promoted"] = False
        row["reason"] = f"erreur : {exc}"
        try:
            registry.append_training_log(args.registry, row)
        except Exception:
            LOG.error("Impossible de journaliser l'échec.", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
