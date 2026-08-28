"""Connecteur optionnel vers l'API interne (non officielle) de PMU.fr.

⚠️ AVERTISSEMENT IMPORTANT
Cette API n'est ni publiée ni documentée officiellement par PMU. Elle est
utilisée de façon empirique par la communauté des turfistes développeurs,
mais peut être modifiée, déplacée ou désactivée à tout moment, sans préavis,
et son usage automatisé se situe dans une zone grise juridique (droit des
bases de données). Ce module est donc conçu pour échouer PROPREMENT : toute
erreur lève ErreurConnecteurPMU, que l'application (app.py) capture pour
retomber automatiquement sur la saisie manuelle.

À réserver à un usage personnel et ponctuel, avec des appels espacés — pas
à un usage commercial ni à des requêtes en boucle.
"""

import re
from datetime import date as date_cls

import requests

# Plusieurs hôtes/identifiants client ont coexisté au fil des années dans les
# usages documentés par la communauté (l'API n'étant pas versionnée
# officiellement). On essaie dans l'ordre et on garde le premier qui répond.
CANDIDATS_BASE_URL = [
    "https://online.turfinfo.api.pmu.fr/rest/client/61",
    "https://offline.turfinfo.api.pmu.fr/rest/client/7",
]

TIMEOUT_SECONDES = 8
EN_TETES = {"User-Agent": "Mozilla/5.0 (compatible; pronostic-hippique-app/1.0)"}


class ErreurConnecteurPMU(Exception):
    """Levée lorsque l'API PMU est injoignable ou renvoie un format inattendu."""


def _get_json(chemin: str) -> dict:
    """Tente `chemin` sur chaque hôte candidat, renvoie le premier succès."""
    derniere_erreur = "aucun hôte essayé"
    for base_url in CANDIDATS_BASE_URL:
        url = f"{base_url}{chemin}"
        try:
            reponse = requests.get(url, headers=EN_TETES, timeout=TIMEOUT_SECONDES)
            if reponse.status_code == 200:
                return reponse.json()
            derniere_erreur = f"HTTP {reponse.status_code} sur {url}"
        except requests.RequestException as erreur:
            derniere_erreur = f"{type(erreur).__name__} sur {url} : {erreur}"
        except ValueError as erreur:  # réponse non-JSON
            derniere_erreur = f"Réponse non-JSON sur {url} : {erreur}"
    raise ErreurConnecteurPMU(
        f"Impossible de joindre l'API PMU (source non officielle). "
        f"Dernière erreur : {derniere_erreur}"
    )


def recuperer_reunions_du_jour(jour: date_cls = None) -> list[dict]:
    """Renvoie la liste des réunions (hippodromes) programmées pour le jour donné."""
    jour = jour or date_cls.today()
    data = _get_json(f"/programme/{jour.strftime('%d%m%Y')}")
    reunions = data.get("programme", {}).get("reunions", [])
    if not reunions:
        raise ErreurConnecteurPMU("Aucune réunion trouvée dans la réponse de l'API.")
    return reunions


def recuperer_courses_reunion(jour: date_cls, num_reunion: int) -> list[dict]:
    """Renvoie les courses d'une réunion donnée (num_reunion=1 pour R1, etc.)."""
    data = _get_json(f"/programme/{jour.strftime('%d%m%Y')}/R{num_reunion}")
    courses = (data.get("reunion") or {}).get("courses") or data.get("courses", [])
    if not courses:
        raise ErreurConnecteurPMU("Aucune course trouvée pour cette réunion.")
    return courses


def recuperer_partants(jour: date_cls, num_reunion: int, num_course: int) -> list[dict]:
    """Renvoie la liste brute des partants (avec cotes si disponibles) d'une course."""
    chemin = f"/programme/{jour.strftime('%d%m%Y')}/R{num_reunion}/C{num_course}/participants"
    data = _get_json(chemin)
    partants = data.get("participants", [])
    if not partants:
        raise ErreurConnecteurPMU("Aucun partant trouvé pour cette course.")
    return partants


def _extraire_cote(partant: dict) -> float | None:
    """Extrait la cote probable en essayant plusieurs emplacements connus.

    Le nom exact de ce champ a varié selon les versions de l'API (non
    documentée) : on essaie plusieurs chemins avant d'abandonner.
    """
    sources = [
        partant.get("dernierRapportDirect"),
        partant.get("dernierRapportReference"),
    ]
    for source in sources:
        if isinstance(source, dict) and source.get("rapport") is not None:
            try:
                return float(source["rapport"])
            except (TypeError, ValueError):
                continue
    valeur_directe = partant.get("rapport")
    if valeur_directe is not None:
        try:
            return float(valeur_directe)
        except (TypeError, ValueError):
            pass
    return None


def estimer_forme_depuis_musique(musique: str, nb_courses: int = 5) -> int | None:
    """Estime le nombre de podiums sur les dernières courses à partir de la musique.

    Heuristique simple : compte les chiffres 1, 2 ou 3 parmi les premiers
    `nb_courses` résultats détectés dans la chaîne "musique" (format PMU
    standard, ex. "3p2p1p4p0p"). Les incidents (Ret, Dai, tombé...) ne
    portant pas de chiffre de position sont ignorés. Résultat indicatif,
    à vérifier/ajuster manuellement dans le tableau de l'application.
    """
    if not musique:
        return None
    chiffres = re.findall(r"\d", musique)[:nb_courses]
    if not chiffres:
        return None
    return sum(1 for c in chiffres if c in {"1", "2", "3"})


def partant_vers_ligne(partant: dict) -> dict:
    """Convertit un partant brut de l'API en dictionnaire exploitable par l'app.

    Ne remplit que ce que l'API peut raisonnablement fournir (nom, poids,
    cote, musique brute). Les critères qualitatifs du modèle (aptitude
    distance/terrain, niveau jockey/entraîneur) restent à évaluer par
    l'utilisateur : l'API ne donne pas de statistique jockey/entraîneur
    directement exploitable pour notre échelle Fort/Moyen/Faible.
    """
    musique = partant.get("musique", "")
    return {
        "nom": partant.get("nom", "?"),
        "poids": partant.get("handicapPoids") or partant.get("poidsConditionMonte"),
        "cote": _extraire_cote(partant),
        "forme_estimee": estimer_forme_depuis_musique(musique),
        "musique_brute": musique,
        "jockey": (partant.get("driver") or {}).get("nom") or partant.get("nomJockey"),
        "entraineur": partant.get("entraineur"),
        "non_partant": partant.get("statut") == "NON_PARTANT",
    }


def partants_vers_lignes(partants: list[dict]) -> list[dict]:
    """Convertit une liste de partants bruts, en excluant les non-partants."""
    lignes = [partant_vers_ligne(p) for p in partants]
    return [ligne for ligne in lignes if not ligne["non_partant"]]
