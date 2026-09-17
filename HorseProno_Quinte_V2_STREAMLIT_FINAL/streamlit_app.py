from __future__ import annotations

import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st

from quinte_v2 import load_artifact, rank_quinte_v2

# =============================================================================
# CONFIG
# =============================================================================

APP_NAME = "HorseProno Quinté V2"
PMU_BASE_URL = "https://online.turfinfo.api.pmu.fr/rest/client/1"
REQUEST_TIMEOUT = 25
HEADERS = {
    "User-Agent": "HorsePronoQuinteV2/2.0 (+https://streamlit.io)",
    "Accept": "application/json",
}

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "quinte_v2_artifact.json"
BASE_HISTORY_PATH = BASE_DIR / "validated_history.csv"

HISTORY_COLUMNS = [
    "race_id", "race_date", "discipline", "hippodrome", "distance",
    "terrain", "field_size", "horse_number", "horse_name", "jockey",
    "trainer", "odds", "draw", "weight", "age", "sex", "recent_form",
    "finish_position", "is_non_runner",
]

st.set_page_config(page_title=APP_NAME, page_icon="🏇", layout="wide")
st.title("🏇 HorseProno Quinté V2")
st.caption(
    "Le programme PMU complet de la journée est chargé automatiquement. "
    "Choisis simplement la réunion puis la course à analyser — aucun CSV n'est nécessaire."
)


# =============================================================================
# OUTILS GENERAUX
# =============================================================================

def _safe_int(value: Any) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        return int(float(value))
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        value = float(value)
        if not np.isfinite(value):
            return None
        return value
    except Exception:
        return None


def _secret(name: str) -> str | None:
    try:
        if name in st.secrets and st.secrets[name]:
            return str(st.secrets[name])
    except Exception:
        pass
    value = os.getenv(name)
    return str(value) if value else None


def _iter_dicts(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _iter_dicts(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _iter_dicts(value)


def _collect_strings(obj: Any) -> list[str]:
    out: list[str] = []
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for value in obj.values():
            out.extend(_collect_strings(value))
    elif isinstance(obj, list):
        for value in obj:
            out.extend(_collect_strings(value))
    return out


def _person_name(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("nom", "nomCourt", "libelle", "libelleCourt", "name"):
            if value.get(key):
                return str(value[key]).strip()
    return ""


# =============================================================================
# PMU : PROGRAMME COMPLET + SELECTION MANUELLE
# =============================================================================

@st.cache_data(ttl=10 * 60, show_spinner=False)
def get_programme(day_iso: str) -> dict:
    day = date.fromisoformat(day_iso)
    d = day.strftime("%d%m%Y")
    response = requests.get(
        f"{PMU_BASE_URL}/programme/{d}",
        headers=HEADERS,
        params={"specialisation": "INTERNET"},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=10 * 60, show_spinner=False)
def get_course_detail(day_iso: str, reunion: int, course: int) -> dict:
    day = date.fromisoformat(day_iso)
    d = day.strftime("%d%m%Y")
    response = requests.get(
        f"{PMU_BASE_URL}/programme/{d}/R{reunion}/C{course}",
        headers=HEADERS,
        params={"specialisation": "INTERNET"},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


@st.cache_data(ttl=5 * 60, show_spinner=False)
def get_participants(day_iso: str, reunion: int, course: int) -> dict:
    day = date.fromisoformat(day_iso)
    d = day.strftime("%d%m%Y")
    response = requests.get(
        f"{PMU_BASE_URL}/programme/{d}/R{reunion}/C{course}/participants",
        headers=HEADERS,
        params={"specialisation": "INTERNET"},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def _find_reunions(obj: Any) -> list[dict]:
    if isinstance(obj, dict):
        reunions = obj.get("reunions")
        if isinstance(reunions, list):
            return [x for x in reunions if isinstance(x, dict)]
        for value in obj.values():
            found = _find_reunions(value)
            if found:
                return found
    elif isinstance(obj, list):
        for value in obj:
            found = _find_reunions(value)
            if found:
                return found
    return []


def _meeting_number(reunion_obj: dict) -> int | None:
    return _safe_int(
        reunion_obj.get("numOfficiel")
        or reunion_obj.get("numReunion")
        or reunion_obj.get("numReunionProgramme")
        or reunion_obj.get("numero")
    )


def _course_number(course_obj: dict) -> int | None:
    return _safe_int(
        course_obj.get("numOrdre")
        or course_obj.get("numCourse")
        or course_obj.get("numOfficiel")
        or course_obj.get("numero")
    )


def _hippodrome(reunion_obj: dict) -> str:
    value = reunion_obj.get("hippodrome")
    if isinstance(value, dict):
        return str(
            value.get("libelleCourt")
            or value.get("libelleLong")
            or value.get("nom")
            or "INCONNU"
        )
    return str(value or "INCONNU")


def _course_label(course_obj: dict, course: int) -> str:
    return str(
        course_obj.get("libelle")
        or course_obj.get("nom")
        or course_obj.get("libelleCourt")
        or f"Course {course}"
    )


def exact_quinte_codes(course_obj: dict) -> list[str]:
    """Détection stricte du support Quinté officiel PMU."""
    found: set[str] = set()
    for obj in _iter_dicts(course_obj):
        for key in ("codePari", "typePari", "code"):
            value = obj.get(key)
            if value is None:
                continue
            code = str(value).strip().upper()
            if code == "E_QUINTE_PLUS":
                found.add(code)
    return sorted(found)


def _course_start_time(course_obj: dict) -> str:
    """Retourne une heure HH:MM quand le programme PMU la fournit."""
    for key in (
        "heureDepart",
        "heureDepartCourse",
        "dateHeureDepart",
        "heure",
    ):
        value = course_obj.get(key)
        if value is None:
            continue

        # Le flux PMU peut fournir un timestamp en millisecondes.
        numeric = _safe_float(value)
        if numeric is not None and numeric > 10_000_000_000:
            try:
                dt = datetime.fromtimestamp(numeric / 1000.0, tz=ZoneInfo("Europe/Paris"))
                return dt.strftime("%H:%M")
            except Exception:
                pass

        if isinstance(value, str):
            text = value.strip()
            # ISO / date-heure.
            try:
                dt = pd.to_datetime(text, errors="raise")
                if not pd.isna(dt):
                    return dt.strftime("%H:%M")
            except Exception:
                pass
            # Valeur déjà sous forme HH:MM[:SS].
            if len(text) >= 5 and ":" in text:
                return text[:5]

    return ""


def _course_program_distance(course_obj: dict) -> int | None:
    for key in ("distance", "distanceMetres", "distanceM"):
        value = _safe_int(course_obj.get(key))
        if value is not None and 500 <= value <= 10000:
            return value
    return None


def _course_program_field_size(course_obj: dict) -> int | None:
    for key in (
        "nombreDeclaresPartants",
        "nombrePartants",
        "nbPartants",
        "fieldSize",
    ):
        value = _safe_int(course_obj.get(key))
        if value is not None and value >= 0:
            return value
    return None


@st.cache_data(ttl=10 * 60, show_spinner=False)
def discover_daily_program(day_iso: str) -> pd.DataFrame:
    """Charge toutes les réunions et toutes les courses du programme PMU."""
    programme = get_programme(day_iso)
    root = programme.get("programme") if isinstance(programme, dict) else programme
    if not isinstance(root, dict):
        root = programme

    rows: list[dict] = []
    for reunion_obj in _find_reunions(root):
        reunion = _meeting_number(reunion_obj)
        if reunion is None:
            continue

        hippodrome = _hippodrome(reunion_obj)
        courses = reunion_obj.get("courses")
        if not isinstance(courses, list):
            continue

        for course_obj in courses:
            if not isinstance(course_obj, dict):
                continue
            course = _course_number(course_obj)
            if course is None:
                continue

            quinte_codes = exact_quinte_codes(course_obj)
            rows.append(
                {
                    "race_date": day_iso,
                    "meeting_number": reunion,
                    "race_number": course,
                    "hippodrome": hippodrome,
                    "label": _course_label(course_obj, course),
                    "start_time": _course_start_time(course_obj),
                    "distance_m": _course_program_distance(course_obj),
                    "field_size_pmu": _course_program_field_size(course_obj),
                    "is_quinte": bool(quinte_codes),
                    "quinte_codes": " | ".join(quinte_codes),
                }
            )

    if not rows:
        return pd.DataFrame()

    result = (
        pd.DataFrame(rows)
        .drop_duplicates(["race_date", "meeting_number", "race_number"])
        .sort_values(["meeting_number", "race_number"])
        .reset_index(drop=True)
    )
    return result


# =============================================================================
# PMU : CONVERSION DES PARTANTS VERS LE FORMAT DU MODELE
# =============================================================================

def _extract_odds(participant: dict) -> float | None:
    for key in (
        "dernierRapportDirect",
        "dernierRapportReference",
        "rapportDirect",
        "rapportReference",
    ):
        value = participant.get(key)
        if isinstance(value, dict):
            for subkey in ("rapport", "cote", "valeur", "value"):
                odds = _safe_float(value.get(subkey))
                if odds is not None and odds > 1:
                    return odds
        else:
            odds = _safe_float(value)
            if odds is not None and odds > 1:
                return odds

    for key in ("odds", "cote", "rapport"):
        odds = _safe_float(participant.get(key))
        if odds is not None and odds > 1:
            return odds
    return None


def _extract_weight(participant: dict) -> float | None:
    for key in (
        "handicapPoids",
        "poidsConditionMonte",
        "poidsMontureSurchargeDecharge",
        "poids",
    ):
        value = _safe_float(participant.get(key))
        if value is None or value <= 0:
            continue
        # Certains flux historiques expriment les poids au dixième de kg.
        if value > 200:
            value /= 10.0
        if 30 <= value <= 90:
            return value
    return None


def _normalize_sex(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"F", "FEMELLE", "FEMELLES", "JUMENT"}:
        return "FEMELLES"
    if text in {"H", "HONGRE", "HONGRES"}:
        return "HONGRES"
    if text in {"M", "MALE", "MÂLE", "MALES", "MÂLES", "ENTIER"}:
        return "MALES"
    return text


def _is_non_runner(participant: dict) -> bool:
    status = str(participant.get("statut") or "").strip().upper()
    explicit = participant.get("nonPartant")
    if explicit is True:
        return True
    return any(token in status for token in ("NON_PARTANT", "NON PARTANT", "NON-PARTANT"))


def normalize_discipline(course_detail: dict) -> str:
    text = " ".join(_collect_strings(course_detail)).upper()

    if "CROSS" in text:
        return "CROSS_COUNTRY"
    if "STEEPLE" in text:
        return "STEEPLECHASE"
    if "HAIE" in text:
        return "HAIES"
    if "MONTE" in text or "MONTÉ" in text:
        return "TROT_MONTE"
    if "ATTELE" in text or "ATTELÉ" in text:
        if "AUTOSTART" in text or "AUTO-START" in text:
            return "ATTELE_AUTOSTART"
        return "ATTELE_VOLTE"
    if "PLAT" in text:
        return "PLAT"

    discipline = str(course_detail.get("discipline") or "").strip().upper()
    return discipline or "INCONNU"


def _course_distance(course_detail: dict) -> int | None:
    for obj in _iter_dicts(course_detail):
        for key in ("distance", "distanceMetres", "distanceM"):
            value = _safe_int(obj.get(key))
            if value is not None and 500 <= value <= 10000:
                return value
    return None


def _find_participant_list(payload: Any) -> list[dict]:
    if isinstance(payload, dict):
        participants = payload.get("participants")
        if isinstance(participants, list):
            return [p for p in participants if isinstance(p, dict)]
        for value in payload.values():
            found = _find_participant_list(value)
            if found:
                return found
    elif isinstance(payload, list):
        for value in payload:
            found = _find_participant_list(value)
            if found:
                return found
    return []


def participants_to_frame(
    day_iso: str,
    reunion: int,
    course: int,
    hippodrome: str,
    course_detail: dict,
    participant_payload: dict,
) -> pd.DataFrame:
    participants = _find_participant_list(participant_payload)
    if not participants:
        raise RuntimeError("Le flux PMU ne contient aucun participant exploitable.")

    discipline = normalize_discipline(course_detail)
    distance = _course_distance(course_detail)
    race_id = f"R{reunion}C{course}_{day_iso}"

    rows: list[dict] = []
    for p in participants:
        number = _safe_int(p.get("numPmu") or p.get("numero") or p.get("num"))
        if number is None:
            continue

        horse_name = str(p.get("nom") or p.get("nomCheval") or "").strip()
        jockey = _person_name(p.get("driver")) or _person_name(p.get("jockey"))
        trainer = _person_name(p.get("entraineur")) or _person_name(p.get("trainer"))

        rows.append(
            {
                "race_id": race_id,
                "race_date": day_iso,
                "discipline": discipline,
                "hippodrome": hippodrome,
                "distance": distance,
                "terrain": "INCONNU",
                "field_size": len(participants),
                "horse_number": number,
                "horse_name": horse_name or f"N°{number}",
                "jockey": jockey,
                "trainer": trainer,
                "odds": _extract_odds(p),
                "draw": _safe_int(p.get("placeCorde")),
                "weight": _extract_weight(p),
                "age": _safe_int(p.get("age")),
                "sex": _normalize_sex(p.get("sexe")),
                "recent_form": str(p.get("musique") or ""),
                "finish_position": np.nan,
                "is_non_runner": _is_non_runner(p),
            }
        )

    race = pd.DataFrame(rows)
    if race.empty:
        raise RuntimeError("Aucun partant PMU n'a pu être converti.")

    # Le field_size utilisé par le modèle doit correspondre aux partants réels.
    race = race[~race["is_non_runner"].fillna(False).astype(bool)].copy()
    race["field_size"] = len(race)

    if race["horse_number"].duplicated().any():
        raise RuntimeError("Le flux PMU contient un numéro de cheval en double.")
    if len(race) < 5:
        raise RuntimeError("Moins de cinq partants exploitables après retrait des non-partants.")

    return race.sort_values("horse_number").reset_index(drop=True)


# =============================================================================
# HISTORIQUE : BASE FIGEE + COMPLEMENT SUPABASE OPTIONNEL
# =============================================================================

@st.cache_data(show_spinner=False)
def load_base_history(path: str) -> pd.DataFrame:
    history = pd.read_csv(path)
    history["race_date"] = pd.to_datetime(history["race_date"], errors="coerce").dt.normalize()
    return history


def _rows(response: Any) -> list[dict]:
    return list(getattr(response, "data", None) or [])


@st.cache_data(ttl=60 * 60, show_spinner=False)
def load_incremental_history(start_iso: str, end_iso: str) -> pd.DataFrame:
    """Charge uniquement les courses postérieures à la base figée et antérieures à la course cible."""
    url = _secret("SUPABASE_URL")
    # Préférer une clé de lecture limitée si elle est configurée.
    key = _secret("SUPABASE_KEY") or _secret("SUPABASE_SERVICE_KEY")
    if not url or not key:
        return pd.DataFrame()

    try:
        from supabase import create_client
    except Exception as exc:
        raise RuntimeError(
            "Le package 'supabase' est absent alors que les secrets Supabase sont configurés."
        ) from exc

    client = create_client(url, key)
    start_day = date.fromisoformat(start_iso)
    end_day = date.fromisoformat(end_iso)
    if start_day >= end_day:
        return pd.DataFrame()

    race_rows: list[dict] = []
    offset = 0
    page_size = 1000
    while True:
        response = (
            client.table("races")
            .select(
                "id,race_date,meeting_number,race_number,hippodrome,discipline,"
                "distance_m,terrain,field_size,status"
            )
            .gte("race_date", start_day.isoformat())
            .lt("race_date", end_day.isoformat())
            .order("id")
            .range(offset, offset + page_size - 1)
            .execute()
        )
        batch = _rows(response)
        race_rows.extend(batch)
        if len(batch) < page_size:
            break
        offset += page_size

    if not race_rows:
        return pd.DataFrame()

    races = pd.DataFrame(race_rows)
    races["id"] = pd.to_numeric(races["id"], errors="coerce")
    races = races[races["id"].notna()].copy()
    races["id"] = races["id"].astype(int)
    race_ids = races["id"].tolist()

    participant_rows: list[dict] = []
    for i in range(0, len(race_ids), 30):
        chunk = race_ids[i : i + 30]
        response = (
            client.table("participants")
            .select(
                "id,race_id,horse_name,horse_number,jockey_name,trainer_name,odds,"
                "weight_kg,draw,age,sex,recent_form,finish_position,is_non_runner"
            )
            .in_("race_id", chunk)
            .execute()
        )
        participant_rows.extend(_rows(response))

    if not participant_rows:
        return pd.DataFrame()

    participants = pd.DataFrame(participant_rows)
    merged = participants.merge(races, left_on="race_id", right_on="id", how="inner", suffixes=("", "_race"))

    merged["race_id_model"] = (
        "R" + merged["meeting_number"].astype(int).astype(str)
        + "C" + merged["race_number"].astype(int).astype(str)
        + "_" + pd.to_datetime(merged["race_date"]).dt.strftime("%Y-%m-%d")
    )

    out = pd.DataFrame(
        {
            "race_id": merged["race_id_model"],
            "race_date": pd.to_datetime(merged["race_date"], errors="coerce").dt.normalize(),
            "discipline": merged["discipline"],
            "hippodrome": merged["hippodrome"],
            "distance": merged["distance_m"],
            "terrain": merged["terrain"],
            "field_size": merged["field_size"],
            "horse_number": merged["horse_number"],
            "horse_name": merged["horse_name"],
            "jockey": merged["jockey_name"],
            "trainer": merged["trainer_name"],
            "odds": merged["odds"],
            "draw": merged["draw"],
            "weight": merged["weight_kg"],
            "age": merged["age"],
            "sex": merged["sex"],
            "recent_form": merged["recent_form"],
            "finish_position": merged["finish_position"],
            "is_non_runner": merged["is_non_runner"],
        }
    )

    # Reproduit la règle de validation de l'historique d'origine :
    # non-partants exclus + top 5 complet et unique + au moins 7 partants.
    out = out[~out["is_non_runner"].fillna(False).astype(bool)].copy()
    valid_ids: list[str] = []
    for rid, group in out.groupby("race_id", sort=False):
        if len(group) < 7 or group["horse_number"].duplicated().any():
            continue
        pos = pd.to_numeric(group["finish_position"], errors="coerce")
        top5 = sorted(pos[pos.between(1, 5)].astype(int).tolist())
        if top5 == [1, 2, 3, 4, 5]:
            valid_ids.append(rid)

    return out[out["race_id"].isin(valid_ids)].reset_index(drop=True)


def build_history(target_day: date) -> tuple[pd.DataFrame, str]:
    base = load_base_history(str(BASE_HISTORY_PATH)).copy()
    if base.empty:
        raise RuntimeError("L'historique validé embarqué est vide.")

    base_max = pd.to_datetime(base["race_date"], errors="coerce").max()
    if pd.isna(base_max):
        raise RuntimeError("Date maximale de l'historique embarqué illisible.")

    next_day = base_max.date() + timedelta(days=1)
    source = f"historique embarqué jusqu'au {base_max.date().isoformat()}"

    if next_day < target_day:
        incremental = load_incremental_history(next_day.isoformat(), target_day.isoformat())
        if not incremental.empty:
            base = pd.concat([base, incremental], ignore_index=True, sort=False)
            base = base.drop_duplicates(["race_id", "horse_number"], keep="last")
            inc_max = pd.to_datetime(incremental["race_date"]).max().date()
            source += f" + Supabase jusqu'au {inc_max.isoformat()}"
        else:
            source += " ; complément Supabase indisponible ou sans course validée"

    # Le modèle n'a jamais le droit de voir le jour cible ou le futur.
    base["race_date"] = pd.to_datetime(base["race_date"], errors="coerce").dt.normalize()
    base = base[base["race_date"] < pd.Timestamp(target_day)].copy()
    return base, source


# =============================================================================
# MODELE
# =============================================================================

@st.cache_data(show_spinner=False)
def load_model(path: str) -> dict:
    return load_artifact(path)


def run_prediction(history: pd.DataFrame, race: pd.DataFrame) -> pd.DataFrame:
    artifact = load_model(str(MODEL_PATH))
    return rank_quinte_v2(history, race, artifact, ranker=None)


def save_market_snapshot(
    ranked: pd.DataFrame,
    day_iso: str,
    reunion: int,
    course: int,
) -> tuple[int, str]:
    """Enregistre un snapshot uniquement avec une clé serveur Supabase.

    Une clé anon/de lecture n'est jamais utilisée pour écrire. Si aucune service key
    n'est configurée, la prédiction continue normalement sans snapshot.
    """
    url = _secret("SUPABASE_URL")
    service_key = _secret("SUPABASE_SERVICE_KEY")
    if not url or not service_key:
        return 0, "Snapshot non enregistré : SUPABASE_SERVICE_KEY non configurée."
    try:
        from supabase import create_client
        client = create_client(url, service_key)
        rows = []
        for row in ranked.itertuples(index=False):
            odds = _safe_float(getattr(row, "odds", None))
            market_probability = _safe_float(getattr(row, "market_probability", None))
            rows.append({
                "race_date": day_iso,
                "meeting_number": int(reunion),
                "race_number": int(course),
                "horse_number": int(row.horse_number),
                "odds": odds,
                "market_probability": market_probability,
                "source": "PMU_ONLINE",
            })
        if not rows:
            return 0, "Aucune cote à enregistrer."
        response = client.table("quinte_market_snapshots").insert(rows).execute()
        count = len(getattr(response, "data", None) or rows)
        return int(count), f"Snapshot de marché enregistré ({count} partants)."
    except Exception as exc:
        return 0, f"Snapshot non enregistré : {exc}"


# =============================================================================
# INTERFACE
# =============================================================================

paris_today = datetime.now(ZoneInfo("Europe/Paris")).date()
selected_day = st.date_input("📅 Programme à analyser", value=paris_today)

if st.button("🔄 Actualiser le programme", use_container_width=False):
    get_programme.clear()
    get_course_detail.clear()
    get_participants.clear()
    discover_daily_program.clear()

try:
    with st.spinner("Chargement du programme PMU complet…"):
        programme_df = discover_daily_program(selected_day.isoformat())

    if programme_df.empty:
        st.warning(
            f"Aucune course détectée dans le programme PMU du {selected_day.strftime('%d/%m/%Y')}."
        )
        st.stop()

    meeting_count = int(programme_df["meeting_number"].nunique())
    race_count = int(len(programme_df))
    quinte_count = int(programme_df["is_quinte"].fillna(False).astype(bool).sum())

    m1, m2, m3 = st.columns(3)
    m1.metric("Réunions", meeting_count)
    m2.metric("Courses", race_count)
    m3.metric("Supports Quinté+", quinte_count)

    with st.expander("📋 Voir le programme complet de la journée"):
        programme_display = programme_df.copy()
        programme_display["Course"] = (
            "R" + programme_display["meeting_number"].astype(int).astype(str)
            + "C" + programme_display["race_number"].astype(int).astype(str)
        )
        programme_display["Quinté+"] = programme_display["is_quinte"].map(
            lambda x: "⭐ Oui" if bool(x) else ""
        )
        programme_display = programme_display[
            [
                "Course", "start_time", "hippodrome", "label",
                "distance_m", "field_size_pmu", "Quinté+",
            ]
        ].rename(
            columns={
                "start_time": "Heure",
                "hippodrome": "Hippodrome",
                "label": "Course / Prix",
                "distance_m": "Distance (m)",
                "field_size_pmu": "Partants annoncés",
            }
        )
        st.dataframe(programme_display, use_container_width=True, hide_index=True)

    st.subheader("🎯 Choisir la course à analyser")

    meeting_rows = (
        programme_df[["meeting_number", "hippodrome"]]
        .drop_duplicates()
        .sort_values("meeting_number")
    )
    meeting_options: list[tuple[int, str]] = []
    for row in meeting_rows.itertuples(index=False):
        meeting_no = int(row.meeting_number)
        race_n = int((programme_df["meeting_number"] == meeting_no).sum())
        meeting_options.append(
            (meeting_no, f"R{meeting_no} — {row.hippodrome} ({race_n} courses)")
        )

    selected_meeting_label = st.selectbox(
        "Réunion",
        [label for _, label in meeting_options],
    )
    reunion = next(
        number for number, label in meeting_options if label == selected_meeting_label
    )

    meeting_program = programme_df[
        programme_df["meeting_number"] == reunion
    ].copy().sort_values("race_number")

    course_options: list[tuple[int, str]] = []
    for row in meeting_program.itertuples(index=False):
        time_text = f"{row.start_time} — " if str(row.start_time or "").strip() else ""
        distance_text = (
            f" — {int(row.distance_m)} m"
            if pd.notna(row.distance_m)
            else ""
        )
        field_text = (
            f" — {int(row.field_size_pmu)} partants"
            if pd.notna(row.field_size_pmu)
            else ""
        )
        quinte_text = " ⭐ QUINTÉ+" if bool(row.is_quinte) else ""
        label = (
            f"C{int(row.race_number)} — {time_text}{row.label}"
            f"{distance_text}{field_text}{quinte_text}"
        )
        course_options.append((int(row.race_number), label))

    selected_course_label = st.selectbox(
        "Course",
        [label for _, label in course_options],
    )
    course = next(
        number for number, label in course_options if label == selected_course_label
    )

    selected_row = meeting_program[
        meeting_program["race_number"] == course
    ].iloc[0]
    hippodrome = str(selected_row["hippodrome"])
    label = str(selected_row["label"])
    is_quinte = bool(selected_row["is_quinte"])

    if is_quinte:
        st.success(
            f"Course sélectionnée : R{reunion}C{course} — {hippodrome} — {label} · ⭐ Support Quinté+"
        )
    else:
        st.info(
            f"Course sélectionnée : R{reunion}C{course} — {hippodrome} — {label}"
        )
        st.warning(
            "Le modèle Quinté V2 a été validé en walk-forward sur des supports Quinté officiels. "
            "Il peut calculer un classement sur cette course, mais ses performances hors Quinté "
            "n'ont pas été validées par le backtest spécialisé."
        )

    if st.button("🏇 Analyser cette course", type="primary"):
        with st.spinner("Récupération des partants PMU…"):
            detail = get_course_detail(selected_day.isoformat(), reunion, course)
            participant_payload = get_participants(selected_day.isoformat(), reunion, course)
            race = participants_to_frame(
                selected_day.isoformat(),
                reunion,
                course,
                hippodrome,
                detail,
                participant_payload,
            )

        with st.spinner("Préparation de l'historique et calcul du pronostic…"):
            history, history_source = build_history(selected_day)
            ranked = run_prediction(history, race)

        snapshot_count, snapshot_message = (0, "")
        if is_quinte:
            snapshot_count, snapshot_message = save_market_snapshot(
                ranked, selected_day.isoformat(), reunion, course
            )

        top7 = ranked.head(7)

        a, b, c, d = st.columns(4)
        a.metric("Course", f"R{reunion}C{course}")
        b.metric("Hippodrome", hippodrome)
        c.metric("Discipline", str(ranked.iloc[0]["discipline"]))
        d.metric("Partants", len(ranked))

        st.subheader("🎯 Sélection Quinté V2 — Top 7")
        st.markdown(
            "## " + " - ".join(top7["horse_number"].astype(int).astype(str).tolist())
        )

        core = ranked[ranked["shortlist_role"].eq("NOYAU_MARCHE")].head(6)
        challenger = ranked[ranked["shortlist_role"].eq("CHALLENGER_MODELE")].head(1)
        st.caption(
            "Noyau marché (6) : "
            + " - ".join(core["horse_number"].astype(int).astype(str).tolist())
            + (
                " · Challenger modèle : "
                + " - ".join(challenger["horse_number"].astype(int).astype(str).tolist())
                if not challenger.empty else ""
            )
        )

        display = ranked[
            [
                "quinte_rank", "horse_number", "horse_name", "shortlist_role",
                "top5_probability", "market_probability", "market_rank",
                "model_top5_rank", "odds", "draw", "weight", "age", "jockey", "trainer",
            ]
        ].copy()
        display["top5_probability"] = (100 * display["top5_probability"]).round(1)
        display["market_probability"] = (100 * display["market_probability"]).round(1)
        display["shortlist_role"] = display["shortlist_role"].map({
            "NOYAU_MARCHE": "Noyau",
            "CHALLENGER_MODELE": "Challenger",
            "HORS_TOP7": "Hors Top7",
        }).fillna(display["shortlist_role"])
        display = display.rename(
            columns={
                "quinte_rank": "Rang shortlist",
                "horse_number": "N°",
                "horse_name": "Cheval",
                "shortlist_role": "Rôle",
                "top5_probability": "P(Top5) calibrée %",
                "market_probability": "P(marché) %",
                "market_rank": "Rang marché",
                "model_top5_rank": "Rang modèle Top5",
                "odds": "Cote PMU",
                "draw": "Corde",
                "weight": "Poids",
                "age": "Âge",
                "jockey": "Jockey / Driver",
                "trainer": "Entraîneur",
            }
        )
        st.dataframe(display, use_container_width=True, hide_index=True)

        missing_odds = int(pd.to_numeric(ranked["odds"], errors="coerce").isna().sum())
        if missing_odds:
            st.warning(
                f"{missing_odds} partant(s) sans cote PMU exploitable au moment de la lecture. "
                "Le modèle utilise alors sa valeur de repli prévue pour les cotes manquantes."
            )

        st.caption(
            f"Historique utilisé : {history_source} · {len(history):,} lignes historiques."
        )
        if is_quinte and snapshot_message:
            if snapshot_count:
                st.caption("📸 " + snapshot_message)
            else:
                st.caption("ℹ️ " + snapshot_message)

        st.download_button(
            "⬇️ Télécharger le pronostic CSV",
            data=ranked.to_csv(index=False).encode("utf-8"),
            file_name=(
                f"pronostic_quinte_v2_{selected_day.isoformat()}_R{reunion}C{course}.csv"
            ),
            mime="text/csv",
        )

        with st.expander("🔎 Données PMU récupérées"):
            st.dataframe(
                race[
                    [
                        "horse_number", "horse_name", "odds", "draw", "weight",
                        "age", "sex", "jockey", "trainer", "recent_form",
                    ]
                ],
                use_container_width=True,
                hide_index=True,
            )

        with st.expander("ℹ️ Informations modèle"):
            artifact = load_model(str(MODEL_PATH))
            st.json(
                {
                    "model_name": artifact.get("model_name"),
                    "status": artifact.get("status"),
                    "target": artifact.get("target"),
                    "training_supports": artifact.get("training_supports"),
                    "training_end": artifact.get("training_end"),
                    "strategy": artifact.get("strategy"),
                    "walk_forward_races": artifact.get("backtest", {}).get("walk_forward_races"),
                    "holdout_28_v2": artifact.get("backtest", {}).get("holdout_last_28", {}).get("v2_market6_plus_logit1"),
                    "course_selection": "programme PMU complet + sélection manuelle",
                    "selected_course_is_quinte_plus": is_quinte,
                }
            )

except requests.HTTPError as exc:
    st.error(f"Erreur HTTP PMU : {exc}")
except requests.RequestException as exc:
    st.error(f"Impossible de joindre le flux PMU : {exc}")
except Exception as exc:
    st.error(f"Erreur : {exc}")
    st.exception(exc)
