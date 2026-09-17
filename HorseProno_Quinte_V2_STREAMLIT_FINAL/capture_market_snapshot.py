"""Capture horodatée des cotes PMU d'un support Quinté+ vers Supabase.

Exemples :
  python capture_market_snapshot.py --date 2026-09-17
  python capture_market_snapshot.py --date 2026-09-17 --meeting 1 --race 4

Variables d'environnement requises :
  SUPABASE_URL
  SUPABASE_SERVICE_KEY

Le script peut être lancé par un scheduler à T-60, T-30, T-15, T-5 et T-2.
"""
from __future__ import annotations

import argparse
import os
from datetime import date
from typing import Any

import numpy as np
import requests
from supabase import create_client

PMU_BASE_URL = "https://online.turfinfo.api.pmu.fr/rest/client/1"
HEADERS = {"User-Agent": "HorsePronoQuinteV2Snapshot/2.0", "Accept": "application/json"}
TIMEOUT = 25


def safe_float(v: Any):
    try:
        x = float(v)
        return x if np.isfinite(x) else None
    except Exception:
        return None


def walk_dicts(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from walk_dicts(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk_dicts(v)


def exact_quinte(obj) -> bool:
    for d in walk_dicts(obj):
        for k in ("codePari", "typePari", "code"):
            if str(d.get(k) or "").strip().upper() == "E_QUINTE_PLUS":
                return True
    return False


def get_json(path: str):
    r = requests.get(PMU_BASE_URL + path, params={"specialisation": "INTERNET"}, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def find_quinte(day_iso: str):
    d = date.fromisoformat(day_iso).strftime("%d%m%Y")
    payload = get_json(f"/programme/{d}")
    for obj in walk_dicts(payload):
        courses = obj.get("courses") if isinstance(obj, dict) else None
        if not isinstance(courses, list):
            continue
        meeting = obj.get("numOfficiel") or obj.get("numReunion") or obj.get("numero")
        try:
            meeting = int(meeting)
        except Exception:
            continue
        for c in courses:
            if not isinstance(c, dict) or not exact_quinte(c):
                continue
            course = c.get("numOrdre") or c.get("numCourse") or c.get("numOfficiel") or c.get("numero")
            try:
                return meeting, int(course)
            except Exception:
                pass
    raise RuntimeError("Aucun support E_QUINTE_PLUS trouvé dans le programme PMU.")


def participant_list(payload):
    if isinstance(payload, dict):
        p = payload.get("participants")
        if isinstance(p, list):
            return [x for x in p if isinstance(x, dict)]
        for v in payload.values():
            got = participant_list(v)
            if got:
                return got
    elif isinstance(payload, list):
        for v in payload:
            got = participant_list(v)
            if got:
                return got
    return []


def odds_of(p: dict):
    for key in ("dernierRapportDirect", "dernierRapportReference", "rapportDirect", "rapportReference"):
        v = p.get(key)
        if isinstance(v, dict):
            for sk in ("rapport", "cote", "valeur", "value"):
                x = safe_float(v.get(sk))
                if x is not None and x > 1:
                    return x
        else:
            x = safe_float(v)
            if x is not None and x > 1:
                return x
    for key in ("odds", "cote", "rapport"):
        x = safe_float(p.get(key))
        if x is not None and x > 1:
            return x
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYY-MM-DD")
    ap.add_argument("--meeting", type=int)
    ap.add_argument("--race", type=int)
    args = ap.parse_args()

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_KEY")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL et SUPABASE_SERVICE_KEY sont requis.")

    meeting, race = args.meeting, args.race
    if meeting is None or race is None:
        meeting, race = find_quinte(args.date)

    d = date.fromisoformat(args.date).strftime("%d%m%Y")
    payload = get_json(f"/programme/{d}/R{meeting}/C{race}/participants")
    parts = participant_list(payload)
    rows0 = []
    for p in parts:
        try:
            num = int(p.get("numPmu") or p.get("numero") or p.get("num"))
        except Exception:
            continue
        status = str(p.get("statut") or "").upper()
        if p.get("nonPartant") is True or "NON_PARTANT" in status or "NON PARTANT" in status:
            continue
        rows0.append((num, odds_of(p)))

    inv = {n: (1/o if o and o > 1 else None) for n, o in rows0}
    denom = sum(x for x in inv.values() if x is not None)
    fallback = 1 / max(1, len(rows0))
    rows = []
    for num, odds in rows0:
        prob = (inv[num] / denom) if denom > 0 and inv[num] is not None else fallback
        rows.append({
            "race_date": args.date,
            "meeting_number": meeting,
            "race_number": race,
            "horse_number": num,
            "odds": odds,
            "market_probability": prob,
            "source": "PMU_ONLINE",
        })

    client = create_client(url, key)
    response = client.table("quinte_market_snapshots").insert(rows).execute()
    print(f"Snapshot R{meeting}C{race} {args.date}: {len(getattr(response, 'data', None) or rows)} partants enregistrés.")


if __name__ == "__main__":
    main()
