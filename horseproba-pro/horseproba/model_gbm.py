"""
Modèle alternatif : GRADIENT BOOSTING sur arbres + normalisation intra-course.

Pourquoi un second modèle, en complément du logit conditionnel (`model.py`) ?
==============================================================================
Le logit conditionnel est un modèle ADDITIF LINÉAIRE : la force d'un cheval est
une combinaison pondérée FIXE de ses variables (mêmes coefficients pour tous les
chevaux, toutes les courses). C'est robuste, interprétable, et raisonnable avec
peu de données — mais cela ne peut pas capturer d'INTERACTIONS non-linéaires,
par exemple :
    « tel driver performe nettement mieux SEULEMENT quand le terrain est lourd
      ET la distance dépasse 2700 m »
    « un cheval en petite forme récente (musique moyenne) mais avec une grosse
      fraîcheur ET un entraîneur en grande forme redevient compétitif, alors
      qu'aucun de ces 3 facteurs pris isolément ne le indiquerait »
Un modèle d'arbres de décision boostés (HistGradientBoostingClassifier,
scikit-learn — déjà une dépendance du projet) apprend ce type d'interactions
directement depuis les données. C'est précisément ce qui permet de repérer des
OUTSIDERS À POTENTIEL que le marché (et un modèle purement additif) sous-évalue :
un profil de variables individuellement "moyennes" mais dont la COMBINAISON
précise a historiquement bien fonctionné.

Approche retenue : classification binaire + normalisation Plackett-Luce
========================================================================
Un vrai « ranker » listwise (type LambdaMART) demanderait une dépendance
supplémentaire (LightGBM/XGBoost avec objectif `rank:*`) que ce projet évite
pour rester léger (déploiement Streamlit Community Cloud, `requirements.txt`
minimal). On utilise donc une approche standard, documentée depuis longtemps
dans la littérature sur les paris hippiques (ex. Bill Benter, années 1990,
utilisait des réseaux de neurones exactement de cette façon) :

    1. Chaque partant devient une observation indépendante,
       cible y = 1 si gagnant, 0 sinon (déséquilibre corrigé par
       `class_weight="balanced"`).
    2. `HistGradientBoostingClassifier` estime un score s(x) ∈ [0,1] —
       PAS directement interprétable comme une probabilité de course, car
       l'indépendance entre partants d'une même course est violée par
       construction (on sait qu'exactement un gagnant existe par course).
    3. On reconvertit ce score en force latente via logit(s) = log(s/(1−s)),
       puis on applique un SOFTMAX INTRA-COURSE — exactement le même mécanisme
       que le modèle logit conditionnel (`ConditionalLogitModel._softmax`).
       Ce n'est plus la probabilité brute du classifieur qui compte, mais son
       ORDRE et son ÉCART RELATIF au sein de la course, ce qui corrige à la
       fois le déséquilibre de classes et la violation d'indépendance.

Limites assumées (documentées, pas cachées)
============================================
- Moins interprétable coefficient par coefficient : on fournit à la place des
  IMPORTANCES DE VARIABLES globales (`feature_importances_`), moins précises
  qu'une contribution par cheval.
- Plus sensible au sur-apprentissage avec peu de données : profondeur et
  nombre d'arbres volontairement limités (`max_depth=4`, `max_iter=150`), et
  seuil minimum de courses/victoires avant d'accepter d'entraîner un modèle
  (sinon on retombe silencieusement sur `None`, et l'ensemble n'utilise que le
  logit — voir `ensemble.py`).
- Pas de garantie de meilleure performance que le logit seul sur peu de
  données : c'est justement pour cela qu'on les COMBINE (`ensemble.py`) plutôt
  que de remplacer l'un par l'autre, et que `scripts/train_model.py` compare
  les deux par backtest avant toute promotion en production.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from .features import (
    FEATURE_COLUMNS,
    _norm_key,
    build_features,
    compute_entity_rates,
    compute_grouped_rates,
    distance_bucket,
    going_bucket,
    within_race_standardize,
)
from .model import plackett_luce_topk

# Nombre minimum de courses ET de victoires distinctes avant d'accepter
# d'entraîner un GBM : en-dessous, le risque de sur-apprentissage dépasse le
# bénéfice potentiel, et un modèle non entraîné (None) est plus honnête qu'un
# modèle instable présenté comme fiable.
MIN_RACES_FOR_GBM = 40
MIN_WINS_FOR_GBM = 40


@dataclass
class GBMFitResult:
    n_races: int
    n_runners: int
    fitted: bool
    feature_importances: Dict[str, float] = field(default_factory=dict)
    message: str = ""


@dataclass
class GradientBoostingRanker:
    """Wrapper autour de HistGradientBoostingClassifier respectant l'interface
    fit(history) / predict(runners) de `ConditionalLogitModel`, pour pouvoir
    être utilisé de façon interchangeable (et combiné) dans `ensemble.py`."""

    features: List[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    max_depth: int = 4
    max_iter: int = 150
    learning_rate: float = 0.08
    l2_regularization: float = 1.0
    random_state: int = 7

    jockey_rates: Dict[str, float] = field(default_factory=dict)
    trainer_rates: Dict[str, float] = field(default_factory=dict)
    combo_rates: Dict[Tuple[str, str], float] = field(default_factory=dict)
    going_rates: Dict[Tuple[str, str], float] = field(default_factory=dict)
    distance_rates: Dict[Tuple[str, str], float] = field(default_factory=dict)
    track_rates: Dict[Tuple[str, str], float] = field(default_factory=dict)

    fit_result: Optional[GBMFitResult] = None
    _clf: Optional[HistGradientBoostingClassifier] = field(default=None, repr=False)

    # ------------------------------------------------------------------ #
    def _design(self, runners: pd.DataFrame):
        feats = build_features(
            runners,
            jockey_rates=self.jockey_rates,
            trainer_rates=self.trainer_rates,
            combo_rates=self.combo_rates,
            going_rates=self.going_rates,
            distance_rates=self.distance_rates,
            track_rates=self.track_rates,
        )
        feats = within_race_standardize(feats, self.features)
        X = feats[self.features].to_numpy(dtype=float)
        groups = [np.asarray(idx) for _, idx in feats.groupby("race_id", sort=False).indices.items()]
        return feats, X, groups

    @property
    def is_fitted(self) -> bool:
        return self._clf is not None

    # ------------------------------------------------------------------ #
    def fit(self, history: pd.DataFrame) -> GBMFitResult:
        hist = history.copy()
        hist["finish_position"] = pd.to_numeric(hist["finish_position"], errors="coerce")
        self.jockey_rates = compute_entity_rates(hist, "jockey")
        self.trainer_rates = compute_entity_rates(hist, "trainer")
        if "jockey" in hist.columns and "horse" in hist.columns:
            self.combo_rates = compute_grouped_rates(
                hist, lambda h: list(zip(h["jockey"].map(_norm_key), h["horse"].map(_norm_key))), k=30.0
            )
        if "going" in hist.columns:
            self.going_rates = compute_grouped_rates(
                hist, lambda h: list(zip(h["horse"].map(_norm_key), h["going"].map(going_bucket))), k=25.0
            )
        if "distance_m" in hist.columns:
            self.distance_rates = compute_grouped_rates(
                hist, lambda h: list(zip(h["horse"].map(_norm_key), h["distance_m"].map(distance_bucket))), k=25.0
            )
        if "track" in hist.columns:
            self.track_rates = compute_grouped_rates(
                hist, lambda h: list(zip(h["horse"].map(_norm_key), h["track"].map(_norm_key))), k=25.0
            )

        feats, X, groups = self._design(hist)
        y = (feats["finish_position"] == 1).astype(int).to_numpy()
        valid_groups = [g for g in groups if y[g].sum() == 1]
        n_runners = int(sum(len(g) for g in valid_groups))
        n_wins = len(valid_groups)

        if n_wins < MIN_RACES_FOR_GBM or n_wins < MIN_WINS_FOR_GBM:
            self._clf = None
            self.fit_result = GBMFitResult(
                n_races=n_wins,
                n_runners=n_runners,
                fitted=False,
                message=(
                    f"Historique insuffisant pour le GBM (< {MIN_RACES_FOR_GBM} courses exploitables) : "
                    "modèle non entraîné, l'ensemble retombera intégralement sur le logit conditionnel."
                ),
            )
            return self.fit_result

        idx = np.concatenate(valid_groups)
        Xtr, ytr = X[idx], y[idx]
        clf = HistGradientBoostingClassifier(
            max_depth=self.max_depth,
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            l2_regularization=self.l2_regularization,
            random_state=self.random_state,
            class_weight="balanced",
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=15,
        )
        clf.fit(Xtr, ytr)
        self._clf = clf

        importances = self._permutation_like_importance(Xtr, ytr)
        self.fit_result = GBMFitResult(
            n_races=n_wins,
            n_runners=n_runners,
            fitted=True,
            feature_importances=importances,
            message=f"GBM entraîné sur {n_wins} courses ({n_runners} partants).",
        )
        return self.fit_result

    def _permutation_like_importance(self, X: np.ndarray, y: np.ndarray, n_repeats: int = 3) -> Dict[str, float]:
        """
        Importance de variable par permutation (simplifiée) : on mesure la
        dégradation de la log-vraisemblance du classifieur quand on mélange
        aléatoirement une colonne. Moins coûteux que sklearn.inspection pour
        rester rapide dans une boucle de ré-entraînement quotidienne.
        """
        if self._clf is None or len(X) == 0:
            return {}
        rng = np.random.default_rng(self.random_state)
        base_p = self._clf.predict_proba(X)[:, 1].clip(1e-9, 1 - 1e-9)
        base_ll = float(np.mean(y * np.log(base_p) + (1 - y) * np.log(1 - base_p)))
        out: Dict[str, float] = {}
        for k, f in enumerate(self.features):
            drops = []
            for _ in range(n_repeats):
                Xp = X.copy()
                rng.shuffle(Xp[:, k])
                p = self._clf.predict_proba(Xp)[:, 1].clip(1e-9, 1 - 1e-9)
                ll = float(np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
                drops.append(base_ll - ll)  # positif = la variable est utile
            out[f] = float(np.mean(drops))
        return out

    # ------------------------------------------------------------------ #
    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    def predict(self, runners: pd.DataFrame, n_places: int = 3, mc_samples: int = 20000, seed: int = 7) -> pd.DataFrame:
        """Même contrat de sortie que `ConditionalLogitModel.predict` (colonnes
        strength/p_win/p_place/fair_odds/market_p/value/rank)."""
        feats, X, groups = self._design(runners)
        out = feats.copy()

        if self._clf is None:
            # Modèle non entraîné : repli neutre (force nulle -> équiprobable
            # au sein de chaque course). `ensemble.py` gère ce cas en pesant
            # entièrement sur le logit dans ce scénario.
            out["strength"] = 0.0
        else:
            raw = self._clf.predict_proba(X)[:, 1].clip(1e-6, 1 - 1e-6)
            out["strength"] = np.log(raw / (1 - raw))

        out["p_win"] = 0.0
        out["p_place"] = 0.0
        rng = np.random.default_rng(seed)
        for g in groups:
            s = out["strength"].to_numpy()[g]
            p = self._softmax(s)
            out.iloc[g, out.columns.get_loc("p_win")] = p
            out.iloc[g, out.columns.get_loc("p_place")] = plackett_luce_topk(s, min(n_places, len(g)), rng, mc_samples)

        out["fair_odds"] = 1.0 / out["p_win"].clip(lower=1e-6)
        if "odds" in out.columns:
            odds = pd.to_numeric(out["odds"], errors="coerce")
            inv = (1.0 / odds.where(odds > 1)).fillna(0.0)
            tot = inv.groupby(out["race_id"]).transform("sum").replace(0, np.nan)
            out["market_p"] = (inv / tot).fillna(np.nan)
        else:
            out["market_p"] = np.nan
        out["value"] = out["p_win"] - out["market_p"]
        out["rank"] = out.groupby("race_id")["p_win"].rank(ascending=False, method="first").astype(int)
        return out.sort_values(["race_id", "rank"]).reset_index(drop=True)
