"""
Tests du script de constitution d'historique et du parsing PMU, sans accès réseau
(les appels HTTP sont remplacés par des réponses JSON simulées).
"""

from datetime import date

import pandas as pd
import pytest

from horseproba.data import pmu
from scripts import build_history


@pytest.fixture
def fake_pmu(monkeypatch):
    """Simule les 3 endpoints PMU pour une journée avec 2 courses (1 terminée, 1 annulée)."""

    def fake_get_json(url, user_agent="x"):
        if url.endswith("/programme/15062024"):
            return {
                "programme": {
                    "reunions": [
                        {
                            "numOfficiel": 1,
                            "hippodrome": {"libelleCourt": "CHANTILLY"},
                            "courses": [
                                {"numOrdre": 1, "libelle": "Prix Test", "specialite": "PLAT", "distance": 1600, "heureDepart": 1718449200000},
                                {"numOrdre": 2, "libelle": "Prix Annulé", "specialite": "ATTELE", "distance": 2700},
                            ],
                        }
                    ]
                }
            }
        if url.endswith("/R1/C1/participants"):
            return {
                "participants": [
                    {"nom": "ALPHA", "numPmu": 1, "jockey": "J. Un", "entraineur": "T. Un", "musique": "1p2p", "age": 4,
                     "dernierRapportDirect": {"rapport": 2.5}, "gainsParticipant": {"gainsCarriere": 1000000},
                     "nombreCourses": 5, "nombreVictoires": 2, "nombrePlaces": 3, "ordreArrivee": 2},
                    {"nom": "BETA", "numPmu": 2, "jockey": "J. Deux", "entraineur": "T. Deux", "musique": "3p1p", "age": 5,
                     "dernierRapportReference": {"rapport": 4.0}, "ordreArrivee": 1},
                    {"nom": "GAMMA", "numPmu": 3, "statut": "NON_PARTANT"},
                    {"nom": "DELTA", "numPmu": 4, "incident": "DISQUALIFIE"},
                ]
            }
        if url.endswith("/R1/C1"):
            return {"penetrometre": {"intitule": "Bon souple"}, "ordreArrivee": [[2], [1]], "statut": "FIN_COURSE"}
        if url.endswith("/R1/C2/participants"):
            return {"participants": [{"nom": "X", "numPmu": 1}, {"nom": "Y", "numPmu": 2}]}
        if url.endswith("/R1/C2"):
            return {"statut": "COURSE_ANNULEE"}
        raise pmu.PMUFetchError("Endpoint inconnu dans le test : " + url)

    monkeypatch.setattr(pmu, "_get_json", fake_get_json)
    monkeypatch.setattr(pmu, "RATE_LIMIT_SECONDS", 0.0)


def test_fetch_finished_race_maps_results(fake_pmu):
    ref = pmu.RaceRef(day=date(2024, 6, 15), reunion=1, course=1, label="", track="CHANTILLY", discipline="plat", distance_m=1600)
    df = pmu.fetch_finished_race(ref)
    assert len(df) == 3  # le non-partant est exclu
    by_horse = df.set_index("horse")["finish_position"].to_dict()
    assert by_horse["Beta"] == 1
    assert by_horse["Alpha"] == 2
    assert by_horse["Delta"] == 0  # disqualifié -> non classé
    assert (df["going"] == "bon souple").all()
    assert df.loc[df["horse"] == "Alpha", "earnings"].iloc[0] == 10000.0


def test_fetch_finished_race_raises_when_no_result(fake_pmu):
    ref = pmu.RaceRef(day=date(2024, 6, 15), reunion=1, course=2, label="", discipline="trot")
    with pytest.raises(pmu.PMUFetchError):
        pmu.fetch_finished_race(ref)


def test_collect_day_skips_failures_and_known(fake_pmu):
    frames = build_history.collect_day(date(2024, 6, 15), "all", known=set(), pause=0.0, user_agent="t", budget=None)
    assert len(frames) == 1  # C2 ignorée proprement
    frames2 = build_history.collect_day(date(2024, 6, 15), "all", known={"2024-06-15_R1_C1"}, pause=0.0, user_agent="t", budget=None)
    assert frames2 == []


def test_recompute_days_since_last_run():
    df = pd.DataFrame(
        {
            "race_id": ["a", "b", "c"],
            "race_date": ["2024-01-01", "2024-01-15", "2024-02-01"],
            "horse": ["Zed", "Zed", "Other"],
            "days_since_last_run": [None, None, None],
            "finish_position": [1, 2, 1],
        }
    )
    out = build_history.recompute_days_since_last_run(df)
    zed = out[out["horse"] == "Zed"].sort_values("race_date")
    assert pd.isna(zed["days_since_last_run"].iloc[0])
    assert zed["days_since_last_run"].iloc[1] == 14


def test_parse_args_days():
    args = build_history.parse_args(["--days", "7", "--out", "x.csv"])
    assert (args.end_date - args.start_date).days == 6
    assert args.end_date < date.today()
