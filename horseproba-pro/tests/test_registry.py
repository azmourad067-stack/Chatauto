"""Tests du registre de modèles (`horseproba/registry.py`)."""

import numpy as np
import pandas as pd

from horseproba import registry
from horseproba.data.synthetic import generate_history
from horseproba.ensemble import EnsembleModel
from horseproba.model import ConditionalLogitModel
from horseproba.model_gbm import GradientBoostingRanker


def test_load_model_returns_none_when_absent(tmp_path):
    assert registry.load_model(tmp_path / "nowhere") is None
    assert registry.load_metadata(tmp_path / "nowhere") is None


def test_save_and_load_roundtrip_without_gbm(tmp_path):
    history = generate_history(n_races=15, seed=2)  # trop court pour le GBM
    model = EnsembleModel(logit=ConditionalLogitModel(l2=1.0), gbm=GradientBoostingRanker(), gbm_weight=0.35)
    model.fit(history)
    assert not model.gbm.is_fitted

    directory = tmp_path / "prod"
    registry.save_model(model, directory, metrics={"backtest_logloss": 1.23})
    assert (directory / "logit.json").exists()
    assert not (directory / "gbm.joblib").exists()  # rien à sauvegarder

    loaded = registry.load_model(directory)
    assert loaded is not None
    assert not loaded.gbm.is_fitted

    runners = history.drop(columns=["finish_position"])
    race_id = history["race_id"].iloc[0]
    r = runners[runners["race_id"] == race_id]
    pred_before = model.predict(r)
    pred_after = loaded.predict(r)
    assert np.allclose(pred_before["p_win"], pred_after["p_win"], atol=1e-9)

    meta = registry.load_metadata(directory)
    assert meta["metrics"]["backtest_logloss"] == 1.23


def test_save_and_load_roundtrip_with_gbm(tmp_path):
    history = generate_history(n_races=180, seed=9)
    model = EnsembleModel(logit=ConditionalLogitModel(l2=1.0), gbm=GradientBoostingRanker(random_state=0), gbm_weight=0.4)
    model.fit(history)
    assert model.gbm.is_fitted

    directory = tmp_path / "prod"
    registry.save_model(model, directory)
    assert (directory / "gbm.joblib").exists()

    loaded = registry.load_model(directory)
    assert loaded.gbm.is_fitted
    assert loaded.gbm_weight == 0.4

    runners = history.drop(columns=["finish_position"])
    pred_before = model.predict(runners)
    pred_after = loaded.predict(runners)
    assert np.allclose(pred_before["p_win"], pred_after["p_win"], atol=1e-6)


def test_save_overwrites_stale_gbm_file_when_no_longer_fitted(tmp_path):
    """Un modèle promu SANS GBM ne doit pas laisser traîner un ancien gbm.joblib."""
    history_large = generate_history(n_races=180, seed=9)
    history_small = generate_history(n_races=15, seed=2)
    directory = tmp_path / "prod"

    m1 = EnsembleModel(gbm=GradientBoostingRanker(random_state=0), gbm_weight=0.4)
    m1.fit(history_large)
    registry.save_model(m1, directory)
    assert (directory / "gbm.joblib").exists()

    m2 = EnsembleModel(gbm=GradientBoostingRanker(random_state=0), gbm_weight=0.4)
    m2.fit(history_small)
    registry.save_model(m2, directory)
    assert not (directory / "gbm.joblib").exists()


def test_training_log_append_and_load(tmp_path):
    registry.append_training_log(tmp_path, {"timestamp": "t1", "promoted": True})
    registry.append_training_log(tmp_path, {"timestamp": "t2", "promoted": False})
    log = registry.load_training_log(tmp_path)
    assert len(log) == 2
    assert list(log["promoted"]) == [True, False]


def test_load_training_log_missing_file_returns_empty(tmp_path):
    log = registry.load_training_log(tmp_path / "nope")
    assert isinstance(log, pd.DataFrame)
    assert log.empty
