"""
Modèle d'ENSEMBLE : combine le logit conditionnel (`model.py`) et le gradient
boosting (`model_gbm.py`) en une seule probabilité par cheval.

Pourquoi un ensemble plutôt qu'un seul modèle ?
================================================
Les deux modèles ont des biais différents et complémentaires :
    - le logit conditionnel est stable, bien calibré, et fiable même avec peu
      de données (repli sur les coefficients a priori) — mais additif ;
    - le GBM capture des interactions non-linéaires mais peut sur-apprendre,
      surtout avec peu de données (d'où le seuil `MIN_RACES_FOR_GBM`).

Combiner deux modèles dont les erreurs ne sont pas parfaitement corrélées
réduit la variance globale de l'estimation — c'est le principe de base de
tout ensemble learning. On utilise un « opinion pooling » log-linéaire
(moyenne géométrique pondérée des probabilités, renormalisée) plutôt qu'une
simple moyenne arithmétique : c'est la façon mathématiquement correcte de
combiner deux distributions de probabilité (Genest & Zidek, 1986) — une
moyenne arithmétique de probabilités a tendance à "aplatir" les distributions
piquées, alors que la moyenne géométrique préserve mieux la confiance des deux
modèles quand ils sont d'accord.

    p_ensemble(i) ∝ p_logit(i)^(1 − w) · p_gbm(i)^w         (par course)

`w` (`gbm_weight`) est volontairement modeste par défaut (0.35) : le GBM est
le modèle le plus récent/le moins éprouvé de l'application, on lui laisse une
influence réelle sans qu'il domine le pronostic. `scripts/train_model.py`
peut ajuster `w` par validation (grille simple) si le backtest le justifie.

Si le GBM n'a pas pu être entraîné (historique insuffisant), l'ensemble
retombe intégralement sur le logit (`w` effectif = 0) : le comportement reste
identique à la version précédente du script quand peu de données sont
disponibles.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from .model import ConditionalLogitModel, plackett_luce_topk
from .model_gbm import GradientBoostingRanker


@dataclass
class EnsembleFitResult:
    logit_n_races: int
    logit_pseudo_r2: float
    gbm_fitted: bool
    gbm_n_races: int
    message: str = ""


@dataclass
class EnsembleModel:
    logit: ConditionalLogitModel = field(default_factory=ConditionalLogitModel)
    gbm: GradientBoostingRanker = field(default_factory=GradientBoostingRanker)
    gbm_weight: float = 0.35
    fit_result: Optional[EnsembleFitResult] = None

    # ------------------------------------------------------------------ #
    def fit(self, history: pd.DataFrame) -> EnsembleFitResult:
        logit_res = self.logit.fit(history)
        gbm_res = self.gbm.fit(history)
        self.fit_result = EnsembleFitResult(
            logit_n_races=logit_res.n_races,
            logit_pseudo_r2=logit_res.pseudo_r2,
            gbm_fitted=gbm_res.fitted,
            gbm_n_races=gbm_res.n_races,
            message=gbm_res.message,
        )
        return self.fit_result

    # ------------------------------------------------------------------ #
    def predict(self, runners: pd.DataFrame, n_places: int = 3, mc_samples: int = 20000, seed: int = 7) -> pd.DataFrame:
        logit_pred = self.logit.predict(runners, n_places=n_places, mc_samples=mc_samples, seed=seed)
        effective_w = self.gbm_weight if self.gbm.is_fitted else 0.0

        if effective_w <= 0.0:
            out = logit_pred.copy()
            out["p_win_logit"] = out["p_win"]
            out["p_win_gbm"] = np.nan
            out["ensemble_weight_gbm"] = 0.0
            return out

        gbm_pred = self.gbm.predict(runners, n_places=n_places, mc_samples=mc_samples, seed=seed)
        # les deux DataFrames sont triés par (race_id, rank) propre à CHAQUE
        # modèle : on rejoint sur (race_id, horse) pour rester correct.
        key_cols = ["race_id", "horse"] if "horse" in runners.columns else ["race_id"]
        left = logit_pred.set_index(key_cols)
        right = gbm_pred.set_index(key_cols)[["p_win"]].rename(columns={"p_win": "p_win_gbm"})
        merged = left.join(right, how="left").reset_index()
        merged["p_win_gbm"] = merged["p_win_gbm"].fillna(merged["p_win"])  # sécurité si jointure incomplète
        merged = merged.rename(columns={"p_win": "p_win_logit"})

        log_p = (1 - effective_w) * np.log(merged["p_win_logit"].clip(lower=1e-9)) + effective_w * np.log(
            merged["p_win_gbm"].clip(lower=1e-9)
        )
        merged["strength"] = log_p  # force latente combinée (échelle logit)
        merged["ensemble_weight_gbm"] = effective_w

        p_win = np.zeros(len(merged))
        p_place = np.zeros(len(merged))
        rng = np.random.default_rng(seed)
        for _, idx in merged.groupby("race_id", sort=False).indices.items():
            idx = np.asarray(idx)
            s = merged["strength"].to_numpy()[idx]
            e = np.exp(s - s.max())
            p_win[idx] = e / e.sum()
            p_place[idx] = plackett_luce_topk(s, min(n_places, len(idx)), rng, mc_samples)
        merged["p_win"] = p_win
        merged["p_place"] = p_place
        merged["fair_odds"] = 1.0 / merged["p_win"].clip(lower=1e-6)
        merged["value"] = merged["p_win"] - merged["market_p"]
        merged["rank"] = merged.groupby("race_id")["p_win"].rank(ascending=False, method="first").astype(int)
        return merged.sort_values(["race_id", "rank"]).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    def explain(self, predicted: pd.DataFrame, horse_index: int, top_k: int = 4) -> List[str]:
        """
        Délègue au logit conditionnel (seul modèle offrant des contributions
        variable-par-variable interprétables) et ajoute une note sur le poids
        du GBM dans la décision finale.
        """
        lines = self.logit.explain(predicted, horse_index, top_k=top_k)
        w = float(predicted.iloc[horse_index].get("ensemble_weight_gbm", 0.0))
        if w > 0:
            p_l = predicted.iloc[horse_index].get("p_win_logit")
            p_g = predicted.iloc[horse_index].get("p_win_gbm")
            if p_l is not None and p_g is not None:
                lines.append(
                    f"Modèle boosting (poids {w:.0%}) : p_win={p_g:.1%} vs p_win logit={p_l:.1%} "
                    f"— {'confirme' if abs(p_g - p_l) < 0.03 else 'nuance'} le diagnostic ci-dessus."
                )
        return lines

    # ------------------------------------------------------------------ #
    def metadata(self) -> dict:
        """Résumé sérialisable en JSON pour le registre de modèles (sans les
        poids binaires du GBM, gérés séparément par `registry.py`)."""
        fr = self.fit_result
        return {
            "gbm_weight_configured": self.gbm_weight,
            "gbm_weight_effective": self.gbm_weight if self.gbm.is_fitted else 0.0,
            "logit_n_races": fr.logit_n_races if fr else None,
            "logit_pseudo_r2": fr.logit_pseudo_r2 if fr else None,
            "gbm_fitted": fr.gbm_fitted if fr else False,
            "gbm_n_races": fr.gbm_n_races if fr else None,
            "logit_coefs": self.logit.coefs,
            "gbm_feature_importances": self.gbm.fit_result.feature_importances if self.gbm.fit_result else {},
        }
