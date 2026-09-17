"""Smoke test local de HorseProno Quinté V2.

Usage:
    python smoke_test_v2.py

Ce test ne contacte ni PMU ni Supabase. Il vérifie seulement que les fichiers livrés
sont compatibles et que le moteur peut produire une shortlist sur une course historique.
"""
from pathlib import Path
import json
import pandas as pd

from quinte_v2 import load_artifact, rank_quinte_v2

BASE = Path(__file__).resolve().parent
HISTORY = BASE / "validated_history.csv"
SUPPORTS = BASE / "quinte_supports_99.csv"
ARTIFACT = BASE / "quinte_v2_artifact.json"


def main() -> None:
    history = pd.read_csv(HISTORY)
    supports = pd.read_csv(SUPPORTS)
    artifact = load_artifact(ARTIFACT)

    assert len(supports) == 99, f"99 supports attendus, trouvé {len(supports)}"
    excluded = supports[(supports["race_date"].astype(str) == "2026-08-30") & (supports["meeting_number"] == 1) & (supports["race_number"] == 3)]
    assert len(excluded) == 1, "Le support exclu du 30/08/2026 R1C3 doit être présent dans la cohorte officielle"
    assert artifact.get("training_supports") == 98, "98 supports évaluables attendus dans l'artefact"
    assert artifact.get("model_name") == "HorseProno_Quinte_V2_market6_top5_challenger"
    assert artifact.get("strategy", {}).get("market_core_size") == 6
    assert artifact.get("strategy", {}).get("model_challengers") == 1

    race_id = "R1C1_2026-09-08"
    race = history[history["race_id"].astype(str).eq(race_id)].copy()
    assert len(race) == 16, f"Course test {race_id}: 16 partants attendus, trouvé {len(race)}"

    ranked = rank_quinte_v2(history, race, artifact)
    assert len(ranked) == 16
    assert ranked["quinte_rank"].tolist() == list(range(1, 17))
    assert int(ranked["selected_top7"].sum()) == 7
    assert int(ranked["shortlist_role"].eq("NOYAU_MARCHE").sum()) == 6
    assert int(ranked["shortlist_role"].eq("CHALLENGER_MODELE").sum()) == 1
    assert ranked["top5_probability"].between(0, 1).all()
    assert ranked["market_probability"].between(0, 1).all()

    top7 = ranked.head(7)["horse_number"].astype(int).tolist()
    print("OK - HorseProno Quinté V2")
    print("Artefact:", artifact.get("package_version"), artifact.get("training_end"))
    print("Course test:", race_id)
    print("Top7:", "-".join(map(str, top7)))
    print("Noyau:", "-".join(map(str, ranked[ranked.shortlist_role.eq('NOYAU_MARCHE')].horse_number.astype(int).tolist())))
    print("Challenger:", "-".join(map(str, ranked[ranked.shortlist_role.eq('CHALLENGER_MODELE')].horse_number.astype(int).tolist())))


if __name__ == "__main__":
    main()
