"""
Évaluation de la qualité des probabilités et backtest temporel.

Pourquoi évaluer des PROBABILITÉS et pas seulement le « taux de gagnants trouvés » ?
----------------------------------------------------------------------------------
Le but du modèle n'est pas de désigner un gagnant (impossible de façon fiable)
mais d'estimer correctement des probabilités. Les métriques adaptées sont :

- Log-loss (score logarithmique) sur le gagnant : −(1/N) Σ log p(gagnant).
  Comparé au log-loss du marché et du modèle uniforme (log n).
- Brier score sur la victoire : moyenne de (p_i − y_i)².
- Calibration : parmi les chevaux estimés à ~20 %, environ 20 % gagnent-ils ?
- Taux de réussite du favori du modèle (hit rate top-1) et top-3.
- ROI simulé d'une stratégie naïve « miser 1 € sur chaque cheval où
  p_modèle > p_marché + marge » — fourni À TITRE ILLUSTRATIF uniquement : le
  passé ne garantit rien et les cotes finales diffèrent des cotes probables.

Le backtest est TEMPOREL (walk-forward) : on entraîne sur les courses passées,
on prédit les suivantes, on avance. Cela évite la fuite d'information (utiliser
le futur pour prédire le passé), erreur classique qui gonfle artificiellement
les performances.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .model import ConditionalLogitModel


@dataclass
class EvalReport:
    n_races: int
    logloss_model: float
    logloss_market: Optional[float]
    logloss_uniform: float
    brier_model: float
    brier_market: Optional[float]
    top1_hit_rate: float
    top3_hit_rate: float  # gagnant réel parmi les 3 premiers du modèle
    calibration: pd.DataFrame = field(default_factory=pd.DataFrame)
    roi_value_bets: Optional[float] = None
    n_value_bets: int = 0
    predictions: pd.DataFrame = field(default_factory=pd.DataFrame)

    def as_table(self) -> pd.DataFrame:
        rows = [
            ("Courses évaluées", f"{self.n_races}"),
            ("Log-loss modèle (↓ mieux)", f"{self.logloss_model:.4f}"),
            ("Log-loss marché", "n/d" if self.logloss_market is None else f"{self.logloss_market:.4f}"),
            ("Log-loss uniforme (1/n)", f"{self.logloss_uniform:.4f}"),
            ("Brier modèle (↓ mieux)", f"{self.brier_model:.4f}"),
            ("Brier marché", "n/d" if self.brier_market is None else f"{self.brier_market:.4f}"),
            ("Favori du modèle gagnant", f"{100 * self.top1_hit_rate:.1f} %"),
            ("Gagnant dans le top 3 du modèle", f"{100 * self.top3_hit_rate:.1f} %"),
            ("ROI simulé « value bets » (illustratif)", "n/d" if self.roi_value_bets is None else f"{100 * self.roi_value_bets:+.1f} % sur {self.n_value_bets} mises"),
        ]
        return pd.DataFrame(rows, columns=["Métrique", "Valeur"])


def _race_metrics(pred: pd.DataFrame) -> EvalReport:
    """Calcule les métriques sur un DataFrame de prédictions contenant finish_position."""
    pred = pred.copy()
    pred["is_win"] = (pd.to_numeric(pred["finish_position"], errors="coerce") == 1).astype(float)
    groups = pred.groupby("race_id", sort=False)
    valid = groups["is_win"].transform("sum") == 1
    pred = pred[valid]
    if pred.empty:
        return EvalReport(0, np.nan, None, np.nan, np.nan, None, np.nan, np.nan)

    winners = pred[pred["is_win"] == 1]
    n = len(winners)
    ll_model = -np.log(winners["p_win"].clip(lower=1e-9)).mean()
    sizes = pred.groupby("race_id")["horse"].transform("count")
    ll_unif = np.log(sizes[pred["is_win"] == 1]).mean()

    has_market = pred["market_p"].notna().all() if "market_p" in pred else False
    ll_market = -np.log(winners["market_p"].clip(lower=1e-9)).mean() if has_market else None

    brier_model = ((pred["p_win"] - pred["is_win"]) ** 2).mean()
    brier_market = ((pred["market_p"] - pred["is_win"]) ** 2).mean() if has_market else None

    top1 = (winners["rank"] == 1).mean()
    top3 = (winners["rank"] <= 3).mean()

    # Calibration par déciles de probabilité prédite
    bins = pd.cut(pred["p_win"], bins=[0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 1.0], include_lowest=True)
    calib = pred.groupby(bins, observed=True).agg(
        predite=("p_win", "mean"), observee=("is_win", "mean"), effectif=("is_win", "size")
    ).reset_index().rename(columns={"p_win": "tranche"})
    calib["tranche"] = calib["tranche"].astype(str)

    # ROI illustratif : miser 1 € si p_modèle > p_marché × 1.15 (marge 15 %)
    roi, n_bets = None, 0
    if has_market and "odds" in pred:
        odds = pd.to_numeric(pred["odds"], errors="coerce")
        bets = pred[(pred["p_win"] > pred["market_p"] * 1.15) & odds.notna()]
        n_bets = len(bets)
        if n_bets:
            returns = (bets["is_win"] * odds.loc[bets.index]).sum()
            roi = float((returns - n_bets) / n_bets)

    return EvalReport(
        n_races=n,
        logloss_model=float(ll_model),
        logloss_market=None if ll_market is None else float(ll_market),
        logloss_uniform=float(ll_unif),
        brier_model=float(brier_model),
        brier_market=None if brier_market is None else float(brier_market),
        top1_hit_rate=float(top1),
        top3_hit_rate=float(top3),
        calibration=calib,
        roi_value_bets=roi,
        n_value_bets=n_bets,
        predictions=pred,
    )


def _ordered_race_ids(history: pd.DataFrame) -> List[str]:
    """Identifiants de course triés chronologiquement (par date si connue,
    sinon dans l'ordre d'apparition — comportement partagé par `backtest` et
    `chronological_split` pour rester cohérents)."""
    hist = history.copy()
    if "race_date" in hist.columns:
        hist["_d"] = pd.to_datetime(hist["race_date"], errors="coerce")
        order = hist.groupby("race_id")["_d"].min().sort_values(kind="stable")
    else:
        order = pd.Series(range(hist["race_id"].nunique()), index=hist["race_id"].unique())
    return list(order.index)


def chronological_split(
    history: pd.DataFrame, holdout_frac: float = 0.15, min_holdout: int = 15, max_holdout: int = 200
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Scinde l'historique en (train, holdout) en respectant la CHRONOLOGIE : le
    holdout contient toujours les courses les PLUS RÉCENTES. C'est essentiel
    pour comparer honnêtement un modèle candidat à un modèle de production
    (voir `scripts/train_model.py`) — un split aléatoire permettrait au
    candidat de "voir" indirectement des informations de la même période que
    le modèle de production a utilisées pour s'entraîner, ce qui fausserait
    la comparaison en sa faveur.
    """
    race_ids = _ordered_race_ids(history)
    n = len(race_ids)
    n_holdout = int(np.clip(round(n * holdout_frac), min_holdout, max_holdout))
    n_holdout = min(n_holdout, max(n - 5, 0))  # garder au moins 5 courses pour l'entraînement
    if n_holdout <= 0:
        return history.copy(), history.iloc[0:0].copy()
    train_ids, holdout_ids = race_ids[:-n_holdout], race_ids[-n_holdout:]
    train = history[history["race_id"].isin(train_ids)].copy()
    holdout = history[history["race_id"].isin(holdout_ids)].copy()
    return train, holdout


def backtest(
    history: pd.DataFrame,
    l2: float = 1.0,
    initial_train_frac: float = 0.5,
    n_folds: int = 4,
    model_factory: Optional[Callable[[], object]] = None,
) -> EvalReport:
    """
    Backtest walk-forward :
    - trie les courses par date (ou par ordre d'apparition si pas de date) ;
    - entraîne sur les premières `initial_train_frac` courses ;
    - prédit le bloc suivant, ré-entraîne en incluant ce bloc, etc. (`n_folds` blocs).

    `model_factory` : fonction sans argument renvoyant une INSTANCE FRAÎCHE du
    modèle à évaluer à chaque fold (par défaut `ConditionalLogitModel(l2=l2)`).
    Permet de réutiliser exactement le même protocole de backtest pour le GBM
    ou l'ensemble (`horseproba.ensemble.EnsembleModel`), qui suivent la même
    interface `fit(history)` / `predict(runners)`.

    Renvoie un EvalReport agrégé sur toutes les prédictions hors-échantillon.
    """
    if model_factory is None:
        model_factory = lambda: ConditionalLogitModel(l2=l2)  # noqa: E731

    race_ids = _ordered_race_ids(history)
    n = len(race_ids)
    if n < 10:
        raise ValueError("Au moins 10 courses terminées sont nécessaires pour un backtest.")

    start = max(5, int(n * initial_train_frac))
    fold_edges = np.linspace(start, n, n_folds + 1).astype(int)
    preds: List[pd.DataFrame] = []
    for a, b in zip(fold_edges[:-1], fold_edges[1:]):
        if b <= a:
            continue
        train_ids, test_ids = race_ids[:a], race_ids[a:b]
        train = history[history["race_id"].isin(train_ids)]
        test = history[history["race_id"].isin(test_ids)]
        model = model_factory()
        model.fit(train)
        preds.append(model.predict(test))
    if not preds:
        raise ValueError("Backtest impossible : pas assez de courses pour former des blocs de test.")
    return _race_metrics(pd.concat(preds, ignore_index=True))


def evaluate_predictions(pred: pd.DataFrame) -> EvalReport:
    """Évalue des prédictions déjà calculées (in-sample ou externes) comportant finish_position."""
    return _race_metrics(pred)


def summarize_live_predictions(log: pd.DataFrame) -> pd.DataFrame:
    """
    Résume un JOURNAL DE PRÉDICTIONS RÉSOLUES (produit par
    `scripts/log_predictions.py` puis `scripts/resolve_predictions.py`,
    colonnes attendues : predicted_at, race_id, horse, p_win, is_winner) en
    statistiques QUOTIDIENNES : nombre de partants évalués, log-loss moyen,
    Brier moyen, taux de réussite du favori du modèle.

    Contrairement à `backtest` (qui ré-entraîne sur des repli historiques),
    ceci mesure la performance RÉELLE des pronostics tels qu'ils ont été
    publiés au fil du temps — c'est la boucle de retour visible de
    l'auto-apprentissage (onglet « Auto-apprentissage » de l'application).
    """
    required = {"predicted_at", "race_id", "horse", "p_win", "is_winner"}
    if log is None or log.empty or not required.issubset(log.columns):
        return pd.DataFrame()
    df = log.dropna(subset=["is_winner", "p_win"]).copy()
    if df.empty:
        return pd.DataFrame()
    df["predicted_at"] = pd.to_datetime(df["predicted_at"], errors="coerce")
    df = df.dropna(subset=["predicted_at"])
    if df.empty:
        return pd.DataFrame()
    df["date"] = df["predicted_at"].dt.date
    p = df["p_win"].clip(1e-9, 1 - 1e-9)
    y = df["is_winner"].astype(float)
    df["logloss"] = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    df["brier"] = (p - y) ** 2
    df["model_rank"] = df.groupby("race_id")["p_win"].rank(ascending=False, method="first")

    daily = df.groupby("date").agg(
        n_partants=("horse", "size"),
        n_courses=("race_id", "nunique"),
        logloss_moyen=("logloss", "mean"),
        brier_moyen=("brier", "mean"),
    ).reset_index()

    winners = df[df["is_winner"] == 1]
    if not winners.empty:
        hit = winners.groupby("date")["model_rank"].apply(lambda r: float((r == 1).mean()))
        hit = hit.rename("favori_modele_gagnant").reset_index()
        daily = daily.merge(hit, on="date", how="left")
    else:
        daily["favori_modele_gagnant"] = np.nan
    return daily.sort_values("date").reset_index(drop=True)
