"""
Tests du script de ré-entraînement / promotion (`scripts/train_model.py`).

Utilise des historiques synthétiques modestes pour rester rapide ; on ne
vérifie pas des valeurs de log-loss précises (non déterministes selon la
graine et les blocs de backtest) mais le COMPORTEMENT du garde-fou :
promotion à la première exécution, idempotence si rien ne change, et
`--force` qui outrepasse le garde-fou.
"""

import pandas as pd

from horseproba import registry
from horseproba.data.synthetic import generate_history
from scripts import train_model


def _write_history(path, n_races, seed):
    generate_history(n_races=n_races, seed=seed).to_csv(path, index=False)


def test_first_run_promotes_when_no_production_exists(tmp_path):
    history_path = tmp_path / "history.csv"
    _write_history(history_path, 60, seed=1)
    reg_dir = tmp_path / "models"

    rc = train_model.main(["--history", str(history_path), "--registry", str(reg_dir), "--n-folds", "2"])
    assert rc == 0

    prod = registry.production_dir(reg_dir)
    assert (prod / "logit.json").exists()
    meta = registry.load_metadata(prod)
    assert meta is not None
    assert meta["metrics"]["n_races_total"] == 60

    log = registry.load_training_log(reg_dir)
    assert len(log) == 1
    assert bool(log.iloc[0]["promoted"]) is True


def test_second_run_on_unchanged_history_is_a_noop(tmp_path):
    history_path = tmp_path / "history.csv"
    _write_history(history_path, 60, seed=1)
    reg_dir = tmp_path / "models"

    train_model.main(["--history", str(history_path), "--registry", str(reg_dir), "--n-folds", "2"])
    first_meta = registry.load_metadata(registry.production_dir(reg_dir))

    rc = train_model.main(["--history", str(history_path), "--registry", str(reg_dir), "--n-folds", "2"])
    assert rc == 0
    second_meta = registry.load_metadata(registry.production_dir(reg_dir))
    assert first_meta["saved_at"] == second_meta["saved_at"]  # pas ré-écrit

    log = registry.load_training_log(reg_dir)
    assert len(log) == 2
    assert "inchangé" in log.iloc[1]["reason"]
    assert bool(log.iloc[1]["promoted"]) is False


def test_force_flag_always_promotes(tmp_path):
    history_path = tmp_path / "history.csv"
    _write_history(history_path, 60, seed=1)
    reg_dir = tmp_path / "models"

    train_model.main(["--history", str(history_path), "--registry", str(reg_dir), "--n-folds", "2"])
    first_meta = registry.load_metadata(registry.production_dir(reg_dir))

    rc = train_model.main(["--history", str(history_path), "--registry", str(reg_dir), "--n-folds", "2", "--force"])
    assert rc == 0
    second_meta = registry.load_metadata(registry.production_dir(reg_dir))
    assert second_meta["saved_at"] != first_meta["saved_at"]  # ré-entraîné malgré l'historique inchangé

    log = registry.load_training_log(reg_dir)
    assert bool(log.iloc[-1]["promoted"]) is True
    assert "forcé" in log.iloc[-1]["reason"]


def test_growing_history_triggers_new_training_attempt(tmp_path):
    history_path = tmp_path / "history.csv"
    _write_history(history_path, 60, seed=1)
    reg_dir = tmp_path / "models"
    train_model.main(["--history", str(history_path), "--registry", str(reg_dir), "--n-folds", "2"])

    generate_history(n_races=90, seed=1).to_csv(history_path, index=False)
    rc = train_model.main(["--history", str(history_path), "--registry", str(reg_dir), "--n-folds", "2"])
    assert rc == 0

    log = registry.load_training_log(reg_dir)
    assert len(log) == 2
    assert log.iloc[1]["n_races_total"] == 90
    assert "inchangé" not in log.iloc[1]["reason"]


def test_missing_history_file_returns_error(tmp_path):
    rc = train_model.main(["--history", str(tmp_path / "nope.csv"), "--registry", str(tmp_path / "models")])
    assert rc == 1


def test_main_never_raises_on_corrupt_history(tmp_path):
    bad_path = tmp_path / "bad.csv"
    pd.DataFrame({"race_id": ["r1"], "horse": ["A"]}).to_csv(bad_path, index=False)  # pas de finish_position
    reg_dir = tmp_path / "models"
    rc = train_model.main(["--history", str(bad_path), "--registry", str(reg_dir)])
    # Historique sans course exploitable : le script journalise et sort proprement, sans lever.
    assert rc in (0, 1)
    log = registry.load_training_log(reg_dir)
    assert len(log) == 1
