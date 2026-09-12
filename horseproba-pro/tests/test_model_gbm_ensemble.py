"""
Tests du modèle Gradient Boosting (`model_gbm.py`) et de l'ensemble
logit+GBM (`ensemble.py`).
"""

import numpy as np
import pytest

from horseproba.data.synthetic import generate_history
from horseproba.ensemble import EnsembleModel
from horseproba.model import ConditionalLogitModel
from horseproba.model_gbm import MIN_RACES_FOR_GBM, GradientBoostingRanker


@pytest.fixture(scope="module")
def small_history():
    """Historique trop court pour le GBM (< MIN_RACES_FOR_GBM courses gagnantes)."""
    return generate_history(n_races=15, seed=1)


@pytest.fixture(scope="module")
def large_history():
    """Historique suffisant pour entraîner le GBM."""
    return generate_history(n_races=180, seed=5)


# ----------------------------------------------------------------------------- GBM
def test_gbm_refuses_to_fit_on_small_history(small_history):
    gbm = GradientBoostingRanker(random_state=0)
    result = gbm.fit(small_history)
    assert result.fitted is False
    assert not gbm.is_fitted
    assert "insuffisant" in result.message.lower()


def test_gbm_predict_is_neutral_when_unfitted(small_history):
    gbm = GradientBoostingRanker(random_state=0)
    gbm.fit(small_history)
    race_id = small_history["race_id"].iloc[0]
    runners = small_history[small_history["race_id"] == race_id].drop(columns=["finish_position"])
    pred = gbm.predict(runners)
    # Force nulle pour tous -> probabilités égales au sein de la course.
    assert pred["p_win"].nunique() == 1
    assert pred["p_win"].sum() == pytest.approx(1.0)


def test_gbm_fits_on_sufficient_history_and_sums_to_one(large_history):
    gbm = GradientBoostingRanker(random_state=0)
    result = gbm.fit(large_history)
    assert result.n_races >= MIN_RACES_FOR_GBM
    assert result.fitted
    assert gbm.is_fitted
    assert result.feature_importances  # au moins quelques variables évaluées

    pred = gbm.predict(large_history.drop(columns=["finish_position"]))
    sums = pred.groupby("race_id")["p_win"].sum()
    assert np.allclose(sums, 1.0, atol=1e-6)
    assert not pred[["p_win", "p_place"]].isna().any().any()


# ----------------------------------------------------------------------------- Ensemble
def test_ensemble_falls_back_to_pure_logit_when_gbm_unfit(small_history):
    ens = EnsembleModel(logit=ConditionalLogitModel(l2=1.0), gbm=GradientBoostingRanker(random_state=0), gbm_weight=0.5)
    fit_res = ens.fit(small_history)
    assert fit_res.gbm_fitted is False

    race_id = small_history["race_id"].iloc[0]
    runners = small_history[small_history["race_id"] == race_id].drop(columns=["finish_position"])
    logit_only = ConditionalLogitModel(l2=1.0)
    logit_only.fit(small_history)

    pred_ens = ens.predict(runners, seed=7)
    pred_logit = logit_only.predict(runners, seed=7)[["horse", "p_win"]].rename(columns={"p_win": "p_win_logit_only"})
    assert (pred_ens["ensemble_weight_gbm"] == 0.0).all()
    merged = pred_ens[["horse", "p_win"]].merge(pred_logit, on="horse")
    assert np.allclose(merged["p_win"], merged["p_win_logit_only"], atol=1e-9)


def test_ensemble_combines_logit_and_gbm_when_both_fit(large_history):
    ens = EnsembleModel(logit=ConditionalLogitModel(l2=1.0), gbm=GradientBoostingRanker(random_state=0), gbm_weight=0.4)
    fit_res = ens.fit(large_history)
    assert fit_res.gbm_fitted

    runners = large_history.drop(columns=["finish_position"])
    pred = ens.predict(runners)
    assert (pred["ensemble_weight_gbm"] == 0.4).all()
    assert not pred["p_win_gbm"].isna().all()
    sums = pred.groupby("race_id")["p_win"].sum()
    assert np.allclose(sums, 1.0, atol=1e-6)
    # value = écart au marché, doit rester dans une plage raisonnable
    assert pred["value"].abs().max() <= 1.0


def test_ensemble_metadata_and_explain(large_history):
    ens = EnsembleModel(logit=ConditionalLogitModel(l2=1.0), gbm=GradientBoostingRanker(random_state=0), gbm_weight=0.4)
    ens.fit(large_history)
    meta = ens.metadata()
    assert meta["gbm_fitted"] is True
    assert meta["gbm_weight_effective"] == pytest.approx(0.4)
    assert "log_odds_implied" in meta["logit_coefs"]

    race_id = large_history["race_id"].iloc[0]
    runners = large_history[large_history["race_id"] == race_id].drop(columns=["finish_position"])
    pred = ens.predict(runners)
    lines = ens.explain(pred, 0)
    assert any("boosting" in line.lower() for line in lines)
