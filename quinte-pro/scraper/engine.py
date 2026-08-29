"""Moteur de collecte : fusionne les sources, gere les donnees incompletes.

Le contrat d'interface avec le modele est simple : pour chaque partant,
engine produit un dictionnaire enrichi avec les cles
    horse, driver, age, gains, musique, distance_pref,
    web_sentiment, data_sources
Qu'une source soit KO ou que tout le web soit coupé, le moteur retourne
TOUJOURS un objet RaceCollectResult exploitable (qualite 0..1).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from scraper.sources import (
    RateLimiter,
    WebSearch,
    build_sources,
    normalize_name,
)

logger = logging.getLogger("quinte.engine")

MUSIQUE_DISQUALIFIE_RANK = 20  # D, 0, T, A, Ret. => rang degrade


def parse_musique(musique: str | None) -> dict:
    """Parse une musique type '1p 2p (25) 0a Dm' -> indicateurs de forme."""
    if not musique:
        return {"n": 0, "top5_ratio": 0.5, "avg_rank": 10.0}
    codes = [c for c in str(musique).replace("(", " ").replace(")", " ").split()
             if c and not c.isdigit()]
    ranks = []
    for token in codes:
        tok = token.lower()
        m_rank = None
        if tok.startswith(("d", "0", "t", "a", "r", "ret")):
            m_rank = MUSIQUE_DISQUALIFIE_RANK
        else:
            digits = "".join(ch for ch in tok if ch.isdigit())
            if digits:
                m_rank = int(digits)
        if m_rank is not None:
            ranks.append(m_rank)
    if not ranks:
        return {"n": 0, "top5_ratio": 0.5, "avg_rank": 10.0}
    top5 = sum(1 for r in ranks if r <= 5) / len(ranks)
    return {"n": len(ranks), "top5_ratio": round(top5, 3), "avg_rank": round(sum(ranks) / len(ranks), 2)}


@dataclass
class EnrichedPartant:
    num: int
    horse: str
    driver: str
    age: float | None = None
    gains: float | None = None
    musique: str | None = None
    distance_pref: float | None = None      # distance ideale du cheval (m)
    web_sentiment: float = 0.0
    web_snippets: list[str] = field(default_factory=list)
    data_sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "num": self.num,
            "horse": self.horse,
            "driver": self.driver,
            "age": self.age,
            "gains": self.gains,
            "musique": self.musique,
            "distance_pref": self.distance_pref,
            "web_sentiment": self.web_sentiment,
            "data_sources": self.data_sources,
        }


@dataclass
class RaceCollectResult:
    race: dict
    partants: list[EnrichedPartant]
    sources_report: list[dict]
    quality: float = 0.0
    warnings: list[str] = field(default_factory=list)


def collect_race_data(
    race: dict,
    partants: list[dict],
    max_search_per_race: int | None = None,
    search_delay_s: float | None = None,
) -> RaceCollectResult:
    """Point d'entree du scraping temps reel.

    race     : {hippodrome, discipline, distance, date}
    partants : liste de {num, horse, driver, age?, gains?, musique?,
                         distance_pref?}
    """
    sources = build_sources()
    report = []
    lookup_by_horse: dict[str, dict] = {}
    warnings: list[str] = []

    # --- 1) sources structurees (open-pmu-api si active) ---
    for src in sources:
        if src.__class__.__name__ == "DisabledSiteAdapter":
            report.append({"source": src.name, "etat": src.describe(), "utilisee": False})
            continue
        try:
            data, etat = src.fetch_race(race)
        except Exception as exc:  # ne jamais faire planter le pronostic
            data, etat = None, f"exception: {type(exc).__name__}"
        report.append({"source": src.name, "etat": etat, "utilisee": data is not None})
        if data:
            lookup_by_horse.update(data)

    # --- 2) recherche web par partant (sentiment d'actualite) ---
    web = None
    for src in sources:
        if isinstance(src, WebSearch):
            web = src
            break
    if web and web.active:
        limiter = RateLimiter(search_delay_s if search_delay_s is not None else 1.2)
        limit = max_search_per_race if max_search_per_race else len(partants)
        for part in partants[:limit]:
            query = f'{part["horse"]} {part["driver"]} hippisme'
            try:
                limiter.wait()
                snippets, sentiment, etat = web.search(query)
            except Exception as exc:
                snippets, sentiment, etat = [], 0.0, f"exception: {type(exc).__name__}"
            if snippets:
                report.append({"source": web.name, "etat": etat, "utilisee": True})
            part["_web_sentiment"] = sentiment
            part["_web_snippets"] = snippets[:5]
        if not any(r.get("utilisee", False) for r in report if r["source"] == web.name):
            report.append({"source": web.name, "etat": "aucun snippet recupere", "utilisee": False})

    # --- 3) fusion saisie utilisateur > open-pmu-api > web ---
    enriched: list[EnrichedPartant] = []
    for part in partants:
        num = part.get("num") or len(enriched) + 1
        nom = normalize_name(part.get("horse"))
        structured = lookup_by_horse.get(nom, {})
        sources_list = ["saisie utilisateur"]
        if structured:
            sources_list.append("open-pmu-api")
        sentiment = part.get("_web_sentiment", 0.0)
        snippets = part.get("_web_snippets", [])
        if snippets:
            sources_list.append("recherche web")
        ep = EnrichedPartant(
            num=num,
            horse=str(part.get("horse", "")).strip(),
            driver=str(part.get("driver", "")).strip(),
            age=part.get("age") if part.get("age") is not None else None,
            gains=(part.get("gains") if part.get("gains") is not None
                   else structured.get("gains")),
            musique=(part.get("musique") or structured.get("musique")),
            distance_pref=part.get("distance_pref"),
            web_sentiment=sentiment,
            web_snippets=snippets,
            data_sources=sources_list,
        )
        enriched.append(ep)

    # --- 4) indicateur de qualite des donnees ---
    n_part = max(1, len(enriched))
    score = 0.25  # base : saisie utilisateur
    n_structured = sum(1 for e in enriched if "open-pmu-api" in e.data_sources)
    n_web = sum(1 for e in enriched if "recherche web" in e.data_sources)
    score += 0.45 * (n_structured / n_part) + 0.30 * (n_web / n_part)
    if n_structured == 0 and n_web == 0:
        warnings.append(
            "Aucune source externe n'a pu etre interrogee : le classement est "
            "base sur le modele seul et sur votre saisie."
        )
    return RaceCollectResult(
        race=race,
        partants=enriched,
        sources_report=report,
        quality=round(min(1.0, score), 3),
        warnings=warnings,
    )
