"""HorseProno V5.2 — budget Quinté adaptatif 30 / 50 / 70.

La décision repose uniquement sur des signaux disponibles avant la course :
- consensus marché/fondamental ;
- divergence V5.1 ;
- confiance V5.1.

La logique est volontairement prudente : 70 tickets deviennent exceptionnels.
Le générateur de combinaisons reste celui de V5.1 rétro-calibré ; ce module décide
simplement combien de ses meilleurs tickets retenir.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

V52_CONFIG = {
    "core_tickets": 30,
    "extension_tickets": 50,
    "max_tickets": 70,
    "core_min_consensus": 6,
    "core_max_divergence": 0.30,
    "core_min_confidence": 0.58,
    "max_trigger_consensus": 2,
    "max_trigger_divergence": 0.50,
    "max_trigger_confidence": 0.35,
    "base_stake_eur": 2.0,
}


def _scalar(ranked: pd.DataFrame, col: str, default: float) -> float:
    if ranked is None or ranked.empty or col not in ranked.columns:
        return float(default)
    s = pd.to_numeric(ranked[col], errors="coerce").dropna()
    return float(s.iloc[0]) if len(s) else float(default)


def choose_quinte_ticket_count(ranked: pd.DataFrame) -> dict[str, Any]:
    """Décide 30, 50 ou 70 tickets sans utiliser le résultat de la course."""
    consensus = int(round(_scalar(ranked, "v51_consensus_count", 0.0)))
    divergence = float(np.clip(_scalar(ranked, "v51_divergence_index", 0.50), 0.0, 1.0))
    confidence = float(np.clip(_scalar(ranked, "v51_consensus_confidence", 0.50), 0.0, 1.0))
    stance = str(ranked.iloc[0].get("v51_stance", "")) if ranked is not None and not ranked.empty else ""

    extreme_open = (
        consensus <= int(V52_CONFIG["max_trigger_consensus"])
        or divergence >= float(V52_CONFIG["max_trigger_divergence"])
        or confidence <= float(V52_CONFIG["max_trigger_confidence"])
    )
    strong_consensus = (
        consensus >= int(V52_CONFIG["core_min_consensus"])
        and divergence < float(V52_CONFIG["core_max_divergence"])
        and confidence >= float(V52_CONFIG["core_min_confidence"])
    )

    if extreme_open:
        n = int(V52_CONFIG["max_tickets"])
        profile = "COUVERTURE_70"
        label = "Course très ouverte — couverture maximale"
        reason = "Divergence forte / consensus très faible / confiance faible."
    elif strong_consensus:
        n = int(V52_CONFIG["core_tickets"])
        profile = "CORE_30"
        label = "Course lisible — Core 30"
        reason = "Consensus élevé, faible divergence et confiance suffisante."
    else:
        n = int(V52_CONFIG["extension_tickets"])
        profile = "EXTENSION_50"
        label = "Course intermédiaire — Extension 50"
        reason = "Signal ni assez concentré pour 30, ni assez ouvert pour 70."

    stake = float(n) * float(V52_CONFIG["base_stake_eur"])
    return {
        "ticket_count": n,
        "profile": profile,
        "label": label,
        "reason": reason,
        "consensus_count": consensus,
        "divergence": divergence,
        "confidence": confidence,
        "stance": stance,
        "base_stake_eur": float(V52_CONFIG["base_stake_eur"]),
        "full_stake_eur": stake,
    }


def select_adaptive_tickets(q70: pd.DataFrame, ranked: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    decision = choose_quinte_ticket_count(ranked)
    if q70 is None or q70.empty:
        return pd.DataFrame(), decision
    n = min(int(decision["ticket_count"]), len(q70))
    out = q70.head(n).copy().reset_index(drop=True)
    out["Ticket adaptatif"] = np.arange(1, len(out) + 1)
    return out, decision
