"""
Source web publique : programme, partants et résultats PMU (France).

AVERTISSEMENT
-------------
Les endpoints ci-dessous sont ceux consommés par le site public pmu.fr. Ils sont
accessibles sans authentification mais NE SONT PAS une API officielle documentée :
leur structure peut changer sans préavis. Ce module est donc écrit de manière
très défensive (chaque champ est lu avec un `.get()` et une valeur par défaut) et
toute erreur remonte comme `PMUFetchError` avec un message lisible, afin que
l'interface puisse basculer sur le mode CSV.

Usage responsable
-----------------
- Un User-Agent identifiable est envoyé (configurable via st.secrets["HTTP_USER_AGENT"]).
- Un délai minimal entre requêtes est respecté (RATE_LIMIT_SECONDS).
- Les résultats sont mis en cache côté application (voir app.py, st.cache_data).
- N'utilisez pas ce module pour aspirer massivement des données : pour cela,
  passez par un fournisseur de données sous licence.

Endpoints
---------
Programme du jour :
    GET {BASE}/programme/{JJMMAAAA}
Partants d'une course :
    GET {BASE}/programme/{JJMMAAAA}/R{reunion}/C{course}/participants
Rapports (résultats) définitifs :
    GET {BASE}/programme/{JJMMAAAA}/R{reunion}/C{course}/rapports-definitifs

Rafraîchissement : les fonctions ne mettent rien en cache elles-mêmes ; c'est
l'appelant (Streamlit) qui gère le cache et le bouton « Actualiser ».
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import date
from typing import Any, Dict, List, Optional

import pandas as pd
import requests

BASE_URL = "https://online.turfinfo.api.pmu.fr/rest/client/1"
DEFAULT_TIMEOUT = 10  # secondes
RATE_LIMIT_SECONDS = 0.6
_last_call = 0.0


class PMUFetchError(RuntimeError):
    """Échec de récupération/lecture des données PMU (message destiné à l'utilisateur)."""


@dataclass
class RaceRef:
    day: date
    reunion: int
    course: int
    label: str
    track: str = ""
    discipline: str = ""
    distance_m: Optional[int] = None
    start_time: str = ""

    @property
    def race_id(self) -> str:
        return f"{self.day.isoformat()}_R{self.reunion}_C{self.course}"


def _date_path(day: date) -> str:
    return day.strftime("%d%m%Y")


def _get_json(url: str, user_agent: str = "HorseProbaApp/1.0 (educational)") -> Dict[str, Any]:
    """GET JSON avec limitation de débit, timeout et gestion d'erreurs propre."""
    global _last_call
    wait = RATE_LIMIT_SECONDS - (time.monotonic() - _last_call)
    if wait > 0:
        time.sleep(wait)
    try:
        resp = requests.get(url, headers={"User-Agent": user_agent, "Accept": "application/json"}, timeout=DEFAULT_TIMEOUT)
        _last_call = time.monotonic()
    except requests.Timeout as exc:
        raise PMUFetchError("La source PMU ne répond pas (délai dépassé).") from exc
    except requests.RequestException as exc:
        raise PMUFetchError(f"Erreur réseau lors de l'accès à la source PMU : {exc}") from exc
    if resp.status_code == 204 or not resp.content:
        raise PMUFetchError("Aucune donnée disponible pour cette date/course (réponse vide).")
    if resp.status_code >= 400:
        raise PMUFetchError(f"La source PMU a renvoyé une erreur HTTP {resp.status_code}.")
    try:
        return resp.json()
    except ValueError as exc:
        raise PMUFetchError("Réponse PMU illisible (JSON invalide) — la structure a peut-être changé.") from exc


def _map_discipline(specialite: str) -> str:
    s = (specialite or "").upper()
    if any(k in s for k in ("TROT", "ATTELE", "MONTE")):
        return "trot"
    if any(k in s for k in ("HAIE", "STEEPLE", "CROSS", "OBSTACLE")):
        return "obstacle"
    return "plat"


def fetch_program(day: date, user_agent: str = "HorseProbaApp/1.0 (educational)") -> List[RaceRef]:
    """Liste les courses du jour. Lève PMUFetchError en cas d'échec."""
    data = _get_json(f"{BASE_URL}/programme/{_date_path(day)}", user_agent)
    reunions = (data.get("programme") or {}).get("reunions") or []
    refs: List[RaceRef] = []
    for reunion in reunions:
        rnum = reunion.get("numOfficiel")
        track = ((reunion.get("hippodrome") or {}).get("libelleCourt")) or ""
        for course in reunion.get("courses") or []:
            cnum = course.get("numOrdre") or course.get("numExterne")
            if rnum is None or cnum is None:
                continue
            heure = course.get("heureDepart")
            start = ""
            if isinstance(heure, (int, float)):
                start = pd.to_datetime(int(heure), unit="ms", utc=True).tz_convert("Europe/Paris").strftime("%H:%M")
            refs.append(
                RaceRef(
                    day=day,
                    reunion=int(rnum),
                    course=int(cnum),
                    label=f"R{rnum}C{cnum} {start} – {track} – {course.get('libelle', '')}".strip(),
                    track=track,
                    discipline=_map_discipline(course.get("specialite", "")),
                    distance_m=course.get("distance"),
                    start_time=start,
                )
            )
    if not refs:
        raise PMUFetchError("Aucune course trouvée pour cette date.")
    return refs


def fetch_runners(ref: RaceRef, user_agent: str = "HorseProbaApp/1.0 (educational)") -> pd.DataFrame:
    """
    Récupère les partants d'une course et les convertit au schéma canonique.
    Les cotes utilisées sont les « rapports probables » (cote de référence PMU) si présents.
    """
    url = f"{BASE_URL}/programme/{_date_path(ref.day)}/R{ref.reunion}/C{ref.course}/participants"
    data = _get_json(url, user_agent)
    participants = data.get("participants") or []
    if not participants:
        raise PMUFetchError("Aucun partant renvoyé pour cette course.")

    rows = []
    for p in participants:
        if str(p.get("statut", "PARTANT")).upper() == "NON_PARTANT":
            continue
        odds = None
        odds_probable = None
        odds_direct = None
        # On garde les DEUX cotes séparément (au lieu de ne garder que la
        # dernière connue) : la comparaison probable -> direct permet de
        # calculer le "mouvement de marché" (feature `market_drift`, voir
        # `horseproba/features.py`), un signal utilisé en pratique pour
        # repérer les chevaux "rentrés" par des parieurs informés.
        rap_direct = p.get("dernierRapportDirect")
        rap_probable = p.get("dernierRapportReference")
        if isinstance(rap_direct, dict):
            odds_direct = rap_direct.get("rapport")
        if isinstance(rap_probable, dict):
            odds_probable = rap_probable.get("rapport")
        odds = odds_direct if odds_direct is not None else odds_probable
        # Place à l'arrivée si la course est terminée (sinon None)
        ordre = p.get("ordreArrivee")
        finish = int(ordre) if isinstance(ordre, (int, float)) and ordre > 0 else None
        incident = p.get("incident")  # ex. "DISQUALIFIE", "TOMBE", "ARRETE"
        if finish is None and incident:
            finish = 0
        gains = (p.get("gainsParticipant") or {}).get("gainsCarriere")
        if gains is not None:
            gains = gains / 100.0  # centimes -> euros
        rows.append(
            {
                "race_id": ref.race_id,
                "race_date": ref.day.isoformat(),
                "track": ref.track,
                "discipline": ref.discipline,
                "distance_m": ref.distance_m,
                "going": "",
                "horse": p.get("nom", "").title(),
                "jockey": p.get("driver") or p.get("jockey") or "",
                "trainer": p.get("entraineur", ""),
                "draw": p.get("placeCorde") or p.get("numPmu"),
                "weight_kg": (p.get("handicapPoids") or 0) / 10.0 if p.get("handicapPoids") else None,
                "age": p.get("age"),
                "days_since_last_run": None,  # non fourni de façon fiable par cette source
                "musique": p.get("musique", ""),
                "career_starts": p.get("nombreCourses"),
                "career_wins": p.get("nombreVictoires"),
                "career_places": p.get("nombrePlaces"),
                "earnings": gains,
                "odds": odds,
                "odds_probable": odds_probable,
                "odds_direct": odds_direct,
                "num_pmu": p.get("numPmu"),
                "finish_position": finish,
                "incident": incident or "",
            }
        )
    if not rows:
        raise PMUFetchError("Tous les chevaux sont non-partants.")
    return pd.DataFrame(rows)


def fetch_race_going(day: date, reunion: int, course: int, user_agent: str = "HorseProbaApp/1.0 (educational)") -> Dict[str, Any]:
    """
    Détail d'une course (état du terrain, ordre d'arrivée officiel, statut).
    Renvoie un dict éventuellement vide ; ne lève jamais.
    """
    url = f"{BASE_URL}/programme/{_date_path(day)}/R{reunion}/C{course}"
    try:
        data = _get_json(url, user_agent)
    except PMUFetchError:
        return {}
    out: Dict[str, Any] = {}
    penetro = data.get("penetrometre") or {}
    out["going"] = (penetro.get("intitule") or data.get("etatTerrain") or "").strip().lower()
    out["status"] = data.get("statut", "")
    # ordreArrivee : liste de listes (ex-aequo possibles) de numéros PMU
    arrival: Dict[int, int] = {}
    place = 1
    for group in data.get("ordreArrivee") or []:
        nums = group if isinstance(group, list) else [group]
        for num in nums:
            if isinstance(num, (int, float)):
                arrival[int(num)] = place
        place += len(nums)
    out["arrival"] = arrival
    return out


def fetch_results(ref: RaceRef, user_agent: str = "HorseProbaApp/1.0 (educational)") -> Dict[int, int]:
    """
    Récupère l'ordre d'arrivée (num PMU -> place) en essayant, dans l'ordre :
      1. le détail de la course (`ordreArrivee` officiel, complet) ;
      2. les rapports définitifs (combinaison gagnante du SIMPLE_GAGNANT : 1er seulement).
    Dict vide si la course n'est pas encore courue ou si la source est indisponible.
    """
    detail = fetch_race_going(ref.day, ref.reunion, ref.course, user_agent)
    if detail.get("arrival"):
        return detail["arrival"]

    url = f"{BASE_URL}/programme/{_date_path(ref.day)}/R{ref.reunion}/C{ref.course}/rapports-definitifs"
    try:
        data = _get_json(url, user_agent)
    except PMUFetchError:
        return {}
    arrival: Dict[int, int] = {}
    for pari in data if isinstance(data, list) else []:
        if pari.get("typePari") == "SIMPLE_GAGNANT":
            for rap in pari.get("rapports") or []:
                for i, num in enumerate(str(rap.get("combinaison", "")).split("-")):
                    if num.strip().isdigit():
                        arrival[int(num)] = i + 1
    return arrival


def fetch_finished_race(ref: RaceRef, user_agent: str = "HorseProbaApp/1.0 (educational)") -> pd.DataFrame:
    """
    Partants + résultat d'une course TERMINÉE, au schéma canonique avec `finish_position`.
    Combine les participants (place incluse si disponible), le détail de la course
    (terrain, ordre d'arrivée officiel) et, en dernier recours, les rapports.

    Lève PMUFetchError si aucun résultat n'est disponible (course non courue/annulée).
    """
    runners = fetch_runners(ref, user_agent)
    detail = fetch_race_going(ref.day, ref.reunion, ref.course, user_agent)
    if detail.get("going"):
        runners["going"] = detail["going"]

    arrival = detail.get("arrival") or {}
    if not arrival and runners["finish_position"].isna().all():
        arrival = fetch_results(ref, user_agent)
    if arrival:
        mapped = runners["num_pmu"].map(lambda n: arrival.get(int(n)) if pd.notna(n) else None)
        runners["finish_position"] = runners["finish_position"].where(runners["finish_position"].notna(), mapped)

    has_winner = (pd.to_numeric(runners["finish_position"], errors="coerce") == 1).any()
    if not has_winner:
        raise PMUFetchError("Résultat indisponible : course non courue, annulée ou arrivée non publiée.")
    # Chevaux sans place connue (non classés, disqualifiés) -> 0
    runners["finish_position"] = pd.to_numeric(runners["finish_position"], errors="coerce").fillna(0).astype(int)
    return runners
