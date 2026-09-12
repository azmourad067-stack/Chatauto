"""
Registre de modèles : persistance versionnée du modèle de PRODUCTION et
journal des entraînements successifs.

Rôle dans la boucle d'auto-apprentissage
=========================================
Avant cette évolution, l'application ré-entraînait un modèle À CHAQUE session
Streamlit, sur tout l'historique disponible : correct mais volatil (pas de
mémoire d'une session à l'autre, pas de trace de progression) et redondant
(ré-entraînement inutile si l'historique n'a pas changé).

Ce module introduit un RÉPERTOIRE DE MODÈLE PERSISTÉ (`models/production/`,
committé dans le dépôt comme `data/history_pmu.csv` l'est déjà) :

    models/
      production/
        logit.json     — coefficients du logit conditionnel (JSON, lisible)
        gbm.joblib      — modèle de gradient boosting entraîné (binaire, absent
                          si l'historique était trop court lors du dernier
                          entraînement retenu)
        meta.json       — métadonnées : date d'entraînement, nb de courses,
                          poids de l'ensemble, métriques de validation
      training_log.csv  — UNE LIGNE PAR TENTATIVE D'ENTRAÎNEMENT (promue ou
                           non), permettant de tracer l'évolution de la
                           qualité du modèle au fil des jours (onglet
                           « Auto-apprentissage » de l'app)

`scripts/train_model.py` est le seul responsable de l'ÉCRITURE dans ce
registre (avec une logique de garde-fou avant promotion, voir ce script).
L'application Streamlit et les autres scripts ne font que LIRE via
`load_model` / `load_metadata` / `load_training_log`.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

from .ensemble import EnsembleModel
from .model import ConditionalLogitModel
from .model_gbm import GradientBoostingRanker

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_DIR = Path("models")
PRODUCTION_DIRNAME = "production"
TRAINING_LOG_FILENAME = "training_log.csv"


def production_dir(registry_dir: Path = DEFAULT_REGISTRY_DIR) -> Path:
    return Path(registry_dir) / PRODUCTION_DIRNAME


def save_model(model: EnsembleModel, directory: Path, metrics: Optional[dict] = None) -> None:
    """
    Sauvegarde un `EnsembleModel` dans `directory` (typiquement
    `models/production` ou un répertoire horodaté pour archivage).

    Le GBM (objet scikit-learn) ne peut pas être sérialisé en JSON : il est
    persisté séparément via `joblib` (déjà une dépendance de scikit-learn).
    S'il n'a pas pu être entraîné (historique insuffisant), aucun fichier
    `gbm.joblib` n'est écrit et `load_model` reconstruira un GBM vide.
    """
    import joblib  # import local : dépendance lourde, inutile pour la simple lecture

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "logit.json").write_text(model.logit.to_json(), encoding="utf-8")

    gbm_path = directory / "gbm.joblib"
    if model.gbm.is_fitted:
        joblib.dump(model.gbm, gbm_path)
    elif gbm_path.exists():
        gbm_path.unlink()

    meta = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "gbm_weight": model.gbm_weight,
        "metrics": metrics or {},
        **model.metadata(),
    }
    (directory / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    logger.info("Modèle sauvegardé dans %s (gbm_fitted=%s)", directory, model.gbm.is_fitted)


def load_model(directory: Path) -> Optional[EnsembleModel]:
    """
    Charge un `EnsembleModel` depuis `directory`. Renvoie `None` si aucun
    modèle n'y est encore persisté (première exécution, ex. avant le tout
    premier passage de `scripts/train_model.py`) — l'appelant doit alors
    prévoir un repli (coefficients a priori, ou entraînement à la volée).
    """
    import joblib  # import local, voir save_model

    directory = Path(directory)
    logit_path = directory / "logit.json"
    if not logit_path.exists():
        return None
    try:
        logit = ConditionalLogitModel.from_json(logit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, KeyError):
        logger.error("logit.json illisible dans %s, modèle ignoré", directory, exc_info=True)
        return None

    gbm_path = directory / "gbm.joblib"
    gbm: GradientBoostingRanker
    if gbm_path.exists():
        try:
            gbm = joblib.load(gbm_path)
        except Exception:
            logger.error("gbm.joblib illisible dans %s, repli sur logit seul", directory, exc_info=True)
            gbm = GradientBoostingRanker()
    else:
        gbm = GradientBoostingRanker()

    weight = 0.35
    meta = load_metadata(directory)
    if meta is not None:
        weight = meta.get("gbm_weight", weight)

    return EnsembleModel(logit=logit, gbm=gbm, gbm_weight=weight)


def load_metadata(directory: Path) -> Optional[dict]:
    """Métadonnées du modèle persisté (date d'entraînement, métriques...), ou `None`."""
    meta_path = Path(directory) / "meta.json"
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("meta.json illisible dans %s", directory, exc_info=True)
        return None


def append_training_log(registry_dir: Path, row: dict) -> None:
    """
    Ajoute une ligne au journal d'entraînement (`training_log.csv`), qu'il y
    ait eu promotion ou non. C'est CE fichier qui permet de tracer la courbe
    de progression du modèle dans l'onglet « Auto-apprentissage » : même une
    tentative refusée (garde-fou déclenché) est une information utile.
    """
    registry_dir = Path(registry_dir)
    registry_dir.mkdir(parents=True, exist_ok=True)
    path = registry_dir / TRAINING_LOG_FILENAME
    df_row = pd.DataFrame([row])
    if path.exists():
        try:
            existing = pd.read_csv(path)
            out = pd.concat([existing, df_row], ignore_index=True)
        except (OSError, pd.errors.ParserError):
            logger.error("training_log.csv illisible, écrasement avec la seule nouvelle ligne", exc_info=True)
            out = df_row
    else:
        out = df_row
    out.to_csv(path, index=False)
    logger.info("Journal d'entraînement mis à jour (%d lignes au total)", len(out))


def load_training_log(registry_dir: Path = DEFAULT_REGISTRY_DIR) -> pd.DataFrame:
    path = Path(registry_dir) / TRAINING_LOG_FILENAME
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (OSError, pd.errors.ParserError):
        logger.warning("training_log.csv illisible", exc_info=True)
        return pd.DataFrame()
