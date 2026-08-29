"""Couche basse du scraping : HTTP poli, timeouts, retries bornes, cache TTL disque.

Toute requete sortante passe par ici. Aucune source n'est censee etre
disponible a 100% : chaque appel est protege et renvoie un FetchResult.ok=False
au lieu de lever une exception fatale.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

logger = logging.getLogger("quinte.scraper")

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# User-Agent explicite et identifiable (bonne pratique : pas de fausse
# identite de navigateur grand public).
DEFAULT_UA = (
    "QuintePlusResearchBot/0.1 (+usage personnel; "
    "respecte robots.txt et les conditions d'utilisation des sites)"
)


@dataclass
class FetchResult:
    """Resultat normalise d'un fetch, toujours non-exceptionnel."""

    ok: bool = False
    status_code: int = -1
    data: object = None  # dict/json parse ou texte HTML
    error: str = ""
    cached: bool = False
    source: str = ""
    latency_ms: float = 0.0


@dataclass
class PoliteFetcher:
    """GET avec tete explicite, timeout, retries limites et cache disque TTL.

    Parametres
    ----------
    timeout: secondes par tentative.
    retries: nombre de nouvelles tentatives apres un echec transitoire.
    backoff: base de l'attente exponentielle entre tentatives (secondes).
    cache_ttl_seconds: duree de validite d'une reponse en cache (0 = pas de cache).
    """

    timeout: float = 8.0
    retries: int = 2
    backoff: float = 1.0
    cache_ttl_seconds: int = 3600
    user_agent: str = DEFAULT_UA
    _session: requests.Session = field(default=None, repr=False, init=False)

    def __post_init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": self.user_agent})

    # ------------------------------------------------------------------
    # cache disque
    # ------------------------------------------------------------------
    def _cache_path(self, url: str, params: dict | None) -> Path:
        key = hashlib.sha1(f"{url}|{json.dumps(params or {})}".encode()).hexdigest()[:16]
        return CACHE_DIR / f"{key}.json"

    def _cache_read(self, url: str, params: dict | None) -> object | None:
        if self.cache_ttl_seconds <= 0:
            return None
        path = self._cache_path(url, params)
        if not path.exists():
            return None
        try:
            meta = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return None
        age = time.time() - meta.get("ts", 0)
        if age > self.cache_ttl_seconds:
            path.unlink(missing_ok=True)
            return None
        return meta.get("data")

    def _cache_write(self, url: str, params: dict | None, data: object) -> None:
        if self.cache_ttl_seconds <= 0:
            return
        try:
            self._cache_path(url, params).write_text(
                json.dumps({"ts": time.time(), "data": data}, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError:  # cache non prioritaire
            pass

    # ------------------------------------------------------------------
    # fetch
    # ------------------------------------------------------------------
    def get_text(self, url: str, params: dict | None = None) -> FetchResult:
        return self._fetch(url, params, as_json=False)

    def get_json(self, url: str, params: dict | None = None) -> FetchResult:
        return self._fetch(url, params, as_json=True)

    def _fetch(self, url: str, params: dict | None, as_json: bool) -> FetchResult:
        source = url.split("//")[-1].split("/")[0]
        cached = self._cache_read(url, params)
        if cached is not None:
            return FetchResult(
                ok=True, data=cached, cached=True, source=source, latency_ms=0.0
            )

        start = time.time()
        last_error = ""
        for attempt in range(self.retries + 1):
            try:
                resp = self._session.get(url, params=params, timeout=self.timeout)
                latency = (time.time() - start) * 1000.0
                if resp.status_code == 200:
                    data = resp.json() if as_json else resp.text
                    self._cache_write(url, params, data)
                    return FetchResult(
                        ok=True,
                        status_code=200,
                        data=data,
                        source=source,
                        latency_ms=round(latency, 1),
                    )
                # 4xx = refus definitif (pas de retry), 5xx = retry
                if 400 <= resp.status_code < 500:
                    return FetchResult(
                        status_code=resp.status_code,
                        error=f"HTTP {resp.status_code} (refus de la source)",
                        source=source,
                        latency_ms=round(latency, 1),
                    )
                last_error = f"HTTP {resp.status_code}"
            except requests.RequestException as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt < self.retries:
                time.sleep(self.backoff * (2 ** attempt))
        return FetchResult(
            error=f"echec apres {self.retries + 1} tentative(s): {last_error}",
            source=source,
            latency_ms=round((time.time() - start) * 1000.0, 1),
        )
