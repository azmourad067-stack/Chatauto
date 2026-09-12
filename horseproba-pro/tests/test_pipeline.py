"""
Tests unitaires du pipeline (exécuter : `pytest -q`).

Ils ne nécessitent ni réseau ni Streamlit.
"""

import numpy as np
import pandas as pd
import pytest

from horseproba.data.loader import DataValidationError, load_runners_csv, sample_dataset_path, validate_runners
from horseproba.data.synthetic import generate_history
from horseproba.evaluate import backtest
from horseproba.features import build_features, implied_probabilities, parse_musique
from horseproba.model import ConditionalLogitModel, plackett_luce_topk


# ----------------------------------------------------------------------------- features
def test_parse_musique_basic():
    f = parse_musique("1p 3p Da 2p 0p")
    assert f.n_runs == 5
    assert f.wins == 1
    assert f.places == 3
    assert f.incidents == 1
    assert 0 < f.score < 1


def test_parse_musique_empty_is_neutral():
    assert parse_musique(None).score == pytest.approx(0.35)
    assert parse_musique("").score == pytest.approx(0.35)
    assert parse_musique("(23)").score == pytest.approx(0.35)


def test_implied_probabilities_normalize():
    p = implied_probabilities(pd.Series([2.0, 4.0, np.nan, 0.5]))
    assert p.sum() == pytest.approx(1.0)
    assert p.iloc[0] > p.iloc[1]


def test_build_features_without_optional_columns():
    df = pd.DataFrame({"race_id": ["r"] * 3, "horse": ["a", "b", "c"]})
    out = build_features(df)
    assert not out.isna().any().any()
    assert out["log_odds_implied"].nunique() == 1  # sans cotes : neutre


# ----------------------------------------------------------------------------- loader
def test_load_sample_csv():
    df, warns = load_runners_csv(sample_dataset_path())
    assert df["race_id"].nunique() == 5
    assert "finish_position" in df.columns


def test_validate_rejects_missing_columns():
    with pytest.raises(DataValidationError):
        validate_runners(pd.DataFrame({"foo": [1, 2]}))


def test_validate_aliases_and_solo_races():
    df = pd.DataFrame({"course": ["a", "a", "b"], "cheval": ["x", "y", "z"], "cote": ["3,5", "4", "2"]})
    out, warns = validate_runners(df)
    assert list(out.columns[:2]) == ["race_id", "horse"]
    assert out["race_id"].unique().tolist() == ["a"]
    assert out["odds"].iloc[0] == pytest.approx(3.5)


# ----------------------------------------------------------------------------- model
def test_predict_probabilities_sum_to_one():
    hist = generate_history(n_races=20, seed=1)
    model = ConditionalLogitModel()
    pred = model.predict(hist)
    sums = pred.groupby("race_id")["p_win"].sum()
    assert np.allclose(sums, 1.0)
    assert ((pred["p_place"] >= pred["p_win"] - 1e-9)).all()
    assert (pred["p_place"] <= 1.0 + 1e-9).all()


def test_plackett_luce_topk_exact_vs_mc():
    s = np.array([1.0, 0.5, 0.0, -0.5, -1.0])
    exact = plackett_luce_topk(s, 3)
    mc = plackett_luce_topk(s, 3, np.random.default_rng(0), mc_samples=40000)
    assert exact.sum() == pytest.approx(3.0)
    assert np.allclose(exact, mc, atol=0.02)


def test_fit_recovers_signal_on_synthetic():
    hist = generate_history(n_races=250, seed=3)
    model = ConditionalLogitModel(l2=0.5)
    res = model.fit(hist)
    assert res.converged
    assert res.pseudo_r2 > 0.15
    # la cote et la forme doivent ressortir positives
    assert model.coefs["log_odds_implied"] > 0
    assert model.coefs["form_score"] > 0


def test_fit_with_too_few_races_keeps_prior():
    hist = generate_history(n_races=3, seed=5)
    model = ConditionalLogitModel()
    res = model.fit(hist)
    assert not res.converged
    assert model.coefs == model.prior


def test_model_roundtrip_json():
    m = ConditionalLogitModel()
    m2 = ConditionalLogitModel.from_json(m.to_json())
    assert m2.coefs == m.coefs


# ----------------------------------------------------------------------------- evaluate
def test_backtest_beats_uniform():
    hist = generate_history(n_races=200, seed=11)
    report = backtest(hist, l2=1.0, n_folds=3)
    assert report.n_races > 50
    assert report.logloss_model < report.logloss_uniform
    assert 0 <= report.top1_hit_rate <= 1


def test_backtest_requires_enough_races():
    with pytest.raises(ValueError):
        backtest(generate_history(n_races=5, seed=2))
