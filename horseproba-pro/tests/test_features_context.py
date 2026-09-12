"""
Tests des nouvelles variables contextuelles (aptitude terrain/distance/piste,
duo jockey-cheval, mouvement de cote) et de la détection d'outsiders.
"""

import numpy as np
import pandas as pd

from horseproba.features import build_features, compute_grouped_rates, distance_bucket, going_bucket
from horseproba.model import flag_outsiders


def _history_row(race_id, horse, jockey, going, distance_m, track, pos):
    return {
        "race_id": race_id, "horse": horse, "jockey": jockey, "going": going,
        "distance_m": distance_m, "track": track, "finish_position": pos,
    }


def test_going_bucket_ordering_and_priority():
    assert going_bucket("Très souple") == "tres_souple"
    assert going_bucket("souple") == "souple"
    assert going_bucket("Bon souple") == "souple"  # contient "souple" -> classé avec les terrains souples
    assert going_bucket("Bon") == "bon"
    assert going_bucket("Lourd") == "lourd"
    assert going_bucket(None) == "inconnu"


def test_distance_bucket_thresholds():
    assert distance_bucket(1200) == "sprint"
    assert distance_bucket(1600) == "mile"
    assert distance_bucket(2400) == "demi_fond"
    assert distance_bucket(3000) == "fond"
    assert distance_bucket(None) == "inconnu"


def test_compute_grouped_rates_shrinks_small_samples():
    """1 victoire sur 1 sortie ne doit PAS donner un taux proche de 1 (sur-ajustement)."""
    history = pd.DataFrame([
        _history_row("r1", "Solo", "J", "lourd", 2000, "Vincennes", 1),
    ])
    rates = compute_grouped_rates(
        history, lambda h: list(zip(h["horse"].str.lower(), h["going"].map(going_bucket))), prior_rate=0.10, k=25.0
    )
    rate = rates[("solo", "lourd")]
    assert 0.10 < rate < 0.20  # tiré vers le prior (0.10), loin de 1.0


def test_going_aptitude_distinguishes_mudlover_from_neutral_horse():
    rows = []
    # "MudLover" gagne systématiquement sur terrain lourd (5 sorties), jamais sur bon terrain.
    for i in range(5):
        rows.append(_history_row(f"m{i}", "MudLover", "J1", "lourd", 2000, "Vincennes", 1))
    for i in range(3):
        rows.append(_history_row(f"m2{i}", "MudLover", "J1", "bon", 2000, "Vincennes", 5))
    # "Neutral" a un palmarès moyen, peu importe le terrain.
    for i in range(5):
        rows.append(_history_row(f"n{i}", "Neutral", "J2", "lourd" if i % 2 else "bon", 2000, "Vincennes", 3))
    history = pd.DataFrame(rows)

    runners = pd.DataFrame({
        "race_id": ["today", "today"],
        "horse": ["MudLover", "Neutral"],
        "going": ["lourd", "lourd"],
        "distance_m": [2000, 2000],
        "track": ["Vincennes", "Vincennes"],
        "jockey": ["J1", "J2"],
    })
    feats = build_features(runners, history=history)
    assert feats.loc[feats["horse"] == "MudLover", "going_aptitude"].iloc[0] > feats.loc[feats["horse"] == "Neutral", "going_aptitude"].iloc[0]


def test_jockey_horse_synergy_rewards_repeated_winning_combo():
    rows = []
    for i in range(6):
        rows.append(_history_row(f"a{i}", "Ace", "GoodDuo", "bon", 1600, "Chantilly", 1))
    for i in range(6):
        rows.append(_history_row(f"b{i}", "Ace", "OtherJockey", "bon", 1600, "Chantilly", 6))
    history = pd.DataFrame(rows)
    runners = pd.DataFrame({
        "race_id": ["today", "today"], "horse": ["Ace", "Ace"], "jockey": ["GoodDuo", "OtherJockey"],
        "going": ["bon", "bon"], "distance_m": [1600, 1600], "track": ["Chantilly", "Chantilly"],
    })
    feats = build_features(runners, history=history)
    good = feats.loc[feats["jockey"] == "GoodDuo", "jockey_horse_synergy"].iloc[0]
    other = feats.loc[feats["jockey"] == "OtherJockey", "jockey_horse_synergy"].iloc[0]
    assert good > other


def test_market_drift_sign_and_neutrality():
    runners = pd.DataFrame({
        "race_id": ["r", "r", "r"],
        "horse": ["Backed", "Drifted", "Unknown"],
        "odds_probable": [10.0, 4.0, np.nan],
        "odds_direct": [5.0, 8.0, np.nan],   # Backed : cote raccourcie ; Drifted : cote allongée
        "odds": [5.0, 8.0, 6.0],
    })
    feats = build_features(runners)
    backed = feats.loc[feats["horse"] == "Backed", "market_drift"].iloc[0]
    drifted = feats.loc[feats["horse"] == "Drifted", "market_drift"].iloc[0]
    unknown = feats.loc[feats["horse"] == "Unknown", "market_drift"].iloc[0]
    assert backed > 0  # rentré : signal positif
    assert drifted < 0  # sorti : signal négatif
    assert unknown == 0.0  # cotes manquantes -> neutre


def test_build_features_still_neutral_without_context_columns():
    """Sans going/distance_m/track/odds_probable, les nouvelles variables restent neutres (rétrocompatibilité)."""
    df = pd.DataFrame({"race_id": ["r"] * 3, "horse": ["a", "b", "c"]})
    out = build_features(df)
    assert not out.isna().any().any()
    assert out["market_drift"].eq(0.0).all()


# ----------------------------------------------------------------------------- flag_outsiders
def _pred_row(horse, odds, p_win, market_p):
    return {"horse": horse, "odds": odds, "p_win": p_win, "market_p": market_p, "value": p_win - market_p}


def test_flag_outsiders_clear_cases():
    pred = pd.DataFrame([
        _pred_row("Favori", 2.0, 0.45, 0.50),       # cote basse -> jamais outsider
        _pred_row("VraiOutsider", 12.0, 0.10, 0.06),  # cote haute + écart net + p_win suffisant -> outsider
        _pred_row("CoteHauteMaisAgree", 15.0, 0.05, 0.06),  # cote haute mais PAS d'écart positif -> pas outsider
        _pred_row("CoteHauteFaibleChance", 20.0, 0.02, 0.04),  # écart faible/négatif + p_win trop bas -> pas outsider
    ])
    flags = flag_outsiders(pred)
    assert flags.tolist() == [False, True, False, False]


def test_flag_outsiders_empty_input():
    empty = pd.DataFrame(columns=["horse", "odds", "p_win", "market_p", "value"])
    flags = flag_outsiders(empty)
    assert len(flags) == 0
