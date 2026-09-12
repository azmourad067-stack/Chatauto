"""
Tests de la boucle de journalisation des pronostics (`scripts/log_predictions.py`
et `scripts/resolve_predictions.py`), sans accès réseau ni modèle réel entraîné
(le modèle et la source PMU sont simulés).
"""

from datetime import date

import pandas as pd
import pytest

from horseproba.data import pmu
from scripts import log_predictions, resolve_predictions


class _FakeModel:
    """Modèle factice : force = 1/odds, juste assez pour produire des probabilités
    plausibles et des colonnes cohérentes avec le contrat de `.predict()`."""

    def predict(self, runners: pd.DataFrame, n_places: int = 3, **kwargs) -> pd.DataFrame:
        out = runners.copy()
        inv = 1.0 / pd.to_numeric(out["odds"], errors="coerce").clip(lower=1.01)
        out["p_win"] = inv / inv.groupby(out["race_id"]).transform("sum")
        out["p_place"] = (out["p_win"] * 2.2).clip(upper=0.95)
        out["market_p"] = out["p_win"]
        out["value"] = 0.0
        out["fair_odds"] = 1.0 / out["p_win"]
        out["rank"] = out["p_win"].rank(ascending=False, method="first").astype(int)
        return out


@pytest.fixture
def fake_pmu_program(monkeypatch):
    ref = pmu.RaceRef(day=date(2024, 6, 15), reunion=1, course=1, label="Prix Test", track="CHANTILLY", discipline="plat", distance_m=1600)
    runners = pd.DataFrame({
        "race_id": [ref.race_id] * 3,
        "horse": ["Alpha", "Beta", "Gamma"],
        "odds": [2.0, 5.0, 15.0],
    })

    monkeypatch.setattr(pmu, "fetch_program", lambda day, ua="x": [ref])
    monkeypatch.setattr(pmu, "fetch_runners", lambda r, ua="x": runners.copy())
    return ref, runners


def test_log_predictions_writes_rows_and_dedupes(tmp_path, monkeypatch, fake_pmu_program):
    ref, runners = fake_pmu_program
    out_path = tmp_path / "predictions_log.csv"

    from horseproba import registry
    monkeypatch.setattr(registry, "load_model", lambda d: _FakeModel())
    monkeypatch.setattr(registry, "load_metadata", lambda d: {"saved_at": "2024-06-14T00:00:00+00:00"})

    rc = log_predictions.main(["--date", "2024-06-15", "--out", str(out_path), "--pause", "0"])
    assert rc == 0
    assert out_path.exists()
    df = pd.read_csv(out_path)
    assert len(df) == 3
    assert set(df["horse"]) == {"Alpha", "Beta", "Gamma"}
    assert df["is_winner"].isna().all()

    # Deuxième exécution le même jour : ne doit RIEN dupliquer (idempotence).
    rc2 = log_predictions.main(["--date", "2024-06-15", "--out", str(out_path), "--pause", "0"])
    assert rc2 == 0
    df2 = pd.read_csv(out_path)
    assert len(df2) == 3


def test_log_predictions_without_production_model_fails_cleanly(tmp_path, monkeypatch, fake_pmu_program):
    from horseproba import registry
    monkeypatch.setattr(registry, "load_model", lambda d: None)
    rc = log_predictions.main(["--date", "2024-06-15", "--out", str(tmp_path / "p.csv"), "--pause", "0"])
    assert rc == 1
    assert not (tmp_path / "p.csv").exists()


def test_resolve_predictions_fills_known_results_only():
    preds = pd.DataFrame({
        "race_id": ["r1", "r1", "r2"],
        "horse": ["Alpha", "Beta", "Gamma"],
        "is_winner": [pd.NA, pd.NA, pd.NA],
        "resolved_at": [pd.NA, pd.NA, pd.NA],
    })
    history = pd.DataFrame({
        "race_id": ["r1", "r1"],
        "horse": ["Alpha", "Beta"],
        "finish_position": [2, 1],
    })
    out = resolve_predictions.resolve(preds, history)
    assert out.loc[out["horse"] == "Alpha", "is_winner"].iloc[0] == 0
    assert out.loc[out["horse"] == "Beta", "is_winner"].iloc[0] == 1
    assert pd.isna(out.loc[out["horse"] == "Gamma", "is_winner"].iloc[0])
    assert out.loc[out["horse"] == "Alpha", "resolved_at"].notna().iloc[0]


def test_resolve_predictions_never_overwrites_already_resolved():
    preds = pd.DataFrame({
        "race_id": ["r1"], "horse": ["Alpha"], "is_winner": [1], "resolved_at": ["2024-01-01T00:00:00"],
    })
    history = pd.DataFrame({"race_id": ["r1"], "horse": ["Alpha"], "finish_position": [5]})  # résultat "différent"
    out = resolve_predictions.resolve(preds, history)
    # La ligne était déjà résolue : elle ne doit pas être touchée, même si l'historique diffère.
    assert out.loc[0, "is_winner"] == 1
    assert out.loc[0, "resolved_at"] == "2024-01-01T00:00:00"


def test_resolve_predictions_main_noop_without_files(tmp_path):
    rc = resolve_predictions.main([
        "--predictions", str(tmp_path / "missing.csv"),
        "--history", str(tmp_path / "missing_hist.csv"),
    ])
    assert rc == 0
