"""
utils.py - Logique métier pour l'analyse des musiques hippiques
Application de Pronostics Hippiques - Analyse par IA
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Tuple


# ──────────────────────────────────────────────
#  CONSTANTES ET PONDÉRATIONS
# ──────────────────────────────────────────────

# Pondération des trois composantes du score global
POIDS_CHEVAL      = 0.60   # 60 % — forme propre du cheval
POIDS_JOCKEY      = 0.25   # 25 % — statistiques du jockey
POIDS_ENTRAINEUR  = 0.15   # 15 % — statistiques de l'entraîneur

# Décroissance temporelle : les courses récentes comptent plus
# Index 0 = course la plus récente (gauche de la chaîne)
DECROISSANCE_TEMPORELLE = 0.80          # facteur multiplicatif par position
MAX_COURSES_ANALYSEES   = 10            # on n'analyse pas plus de N dernières courses

# Pénalités / bonus sur les caractères spéciaux
PENALITE_DISQUALIFICATION = 9.0        # D  → traité comme la 9e place
PENALITE_CHUTE            = 9.5        # T  → pire que D (chute = danger)
PENALITE_ARRET            = 9.0        # A  → arrêt de course
PENALITE_RETRAIT          = 8.5        # R  → retiré
PENALITE_DEFERRE          = 0.0        # Pénalité nulle (événement neutre)

# Seuils pour les commentaires
SEUIL_EXCELLENT    = 80
SEUIL_BON          = 60
SEUIL_MOYEN        = 40
SEUIL_INSUFFISANT  = 20


# ──────────────────────────────────────────────
#  DATACLASS DE RÉSULTAT
# ──────────────────────────────────────────────

@dataclass
class ResultatCheval:
    """Contient tous les scores d'un cheval après analyse."""
    numero:            int
    nom:               str
    score_cheval:      float   # 0-100
    score_jockey:      float
    score_entraineur:  float
    score_global:      float
    nb_courses_cheval: int
    nb_courses_jockey: int
    nb_courses_entraineur: int
    commentaire:       str
    rang_pronostic:    int = 0


# ──────────────────────────────────────────────
#  PARSING DE LA MUSIQUE
# ──────────────────────────────────────────────

def _extraire_performances(musique: str, type_course: str) -> List[Tuple[float, bool]]:
    """
    Parse la chaîne de musique et retourne une liste de (valeur_place, est_pertinente).

    Règles :
    - Les chiffres 1-9 indiquent la place (1 = victoire).
    - La lettre qui suit un chiffre indique la discipline :
        'p' → Plat | 'a' → Attelé | 'h' ou 'm' → Haies/Steeple
    - Les parenthèses (25) indiquent l'année ; on les saute (les courses
      incluses seront moins pondérées grâce à la décroissance temporelle).
    - Caractères spéciaux : D, T, A, R → pénalités fixes.
    - Si la discipline ne correspond pas au type_course choisi, la
      performance est marquée comme non-pertinente (ignorée dans le calcul).

    Retourne : liste de tuples (valeur_place [1..9.5], pertinente [bool])
               du plus récent au plus ancien (ordre d'apparition dans la chaîne).
    """
    if not isinstance(musique, str) or musique.strip() == "":
        return []

    musique_clean = musique.strip().upper()
    type_upper    = type_course.upper()   # "PLAT" ou "ATTELÉ" / "ATTELE"

    # Normalisation du type
    is_plat    = "PLAT" in type_upper
    is_attele  = any(k in type_upper for k in ["ATTEL", "TROT"])

    performances: List[Tuple[float, bool]] = []
    i = 0
    n = len(musique_clean)

    while i < n and len(performances) < MAX_COURSES_ANALYSEES:
        c = musique_clean[i]

        # ── Parenthèses d'année : on les ignore (sauter jusqu'au ')') ──
        if c == '(':
            end = musique_clean.find(')', i)
            i = end + 1 if end != -1 else n
            continue

        # ── Chiffre → place ──
        if c.isdigit():
            place = float(c)
            i += 1
            # Cherche la lettre de discipline éventuelle
            discipline = None
            if i < n and musique_clean[i].isalpha() and musique_clean[i] not in ('D', 'T', 'A', 'R'):
                discipline = musique_clean[i]
                i += 1

            # Pertinence : discipline correspondante OU indéfinie
            pertinente = True
            if discipline is not None:
                if is_plat and discipline != 'P':
                    pertinente = False
                elif is_attele and discipline != 'A':
                    pertinente = False

            performances.append((place, pertinente))
            continue

        # ── Caractères spéciaux ──
        if c == 'D':
            performances.append((PENALITE_DISQUALIFICATION, True))
            i += 1
            continue
        if c == 'T':
            performances.append((PENALITE_CHUTE, True))
            i += 1
            continue
        if c == 'A':
            # Vérifier si c'est 'A' de Attelé (après chiffre) ou Arrêt seul
            performances.append((PENALITE_ARRET, True))
            i += 1
            continue
        if c == 'R':
            performances.append((PENALITE_RETRAIT, True))
            i += 1
            continue

        # Espace ou autre séparateur
        i += 1

    return performances


def calculer_score_musique(musique: str, type_course: str) -> Tuple[float, int]:
    """
    Calcule un score normalisé [0-100] à partir d'une chaîne de musique.

    Score élevé = bonnes performances récentes.

    Retourne : (score_normalise, nb_courses_pertinentes)
    """
    performances = _extraire_performances(musique, type_course)

    if not performances:
        return 50.0, 0  # Valeur neutre si aucune donnée

    score_pondere = 0.0
    poids_total   = 0.0
    nb_pertinentes = 0

    for idx, (place, pertinente) in enumerate(performances):
        if not pertinente:
            continue
        # Poids décroissant : course la plus récente (idx=0) a le plus grand poids
        poids = DECROISSANCE_TEMPORELLE ** idx
        # Conversion place → score (1ère place → 9 pts, 9ème → 1 pt)
        score_course = 10.0 - place   # 1→9, 2→8, ..., 9→1, D(9)→1, T(9.5)→0.5
        score_course = max(0.0, score_course)

        score_pondere += poids * score_course
        poids_total   += poids
        nb_pertinentes += 1

    if poids_total == 0:
        return 50.0, 0

    # Normalisation sur 100 : le maximum possible est 9 (si 1ère à chaque fois)
    score_brut = score_pondere / poids_total      # entre 0 et 9
    score_norme = (score_brut / 9.0) * 100.0      # 0-100

    return round(score_norme, 2), nb_pertinentes


# ──────────────────────────────────────────────
#  SCORE GLOBAL
# ──────────────────────────────────────────────

def calculer_score_global(
    score_cheval:     float,
    score_jockey:     float,
    score_entraineur: float,
    poids_cheval:     float = POIDS_CHEVAL,
    poids_jockey:     float = POIDS_JOCKEY,
    poids_entraineur: float = POIDS_ENTRAINEUR,
) -> float:
    """
    Combine les trois scores avec la pondération définie.
    Les pondérations sont personnalisables ; par défaut les constantes globales.
    Retourne un score global arrondi à 2 décimales.
    """
    # Normalisation au cas où les poids ne somment pas exactement à 1
    total_poids = poids_cheval + poids_jockey + poids_entraineur
    if total_poids <= 0:
        total_poids = 1.0
    global_score = (
        score_cheval     * (poids_cheval     / total_poids) +
        score_jockey     * (poids_jockey     / total_poids) +
        score_entraineur * (poids_entraineur / total_poids)
    )
    return round(global_score, 2)


# ──────────────────────────────────────────────
#  GÉNÉRATION DU COMMENTAIRE
# ──────────────────────────────────────────────

def _appreciation(score: float) -> str:
    if score >= SEUIL_EXCELLENT:
        return "Excellent"
    elif score >= SEUIL_BON:
        return "Bon"
    elif score >= SEUIL_MOYEN:
        return "Moyen"
    elif score >= SEUIL_INSUFFISANT:
        return "Faible"
    else:
        return "Très faible"


def generer_commentaire(resultat: ResultatCheval) -> str:
    """
    Génère un commentaire analytique personnalisé pour chaque cheval.
    """
    apprec_cheval     = _appreciation(resultat.score_cheval)
    apprec_jockey     = _appreciation(resultat.score_jockey)
    apprec_entraineur = _appreciation(resultat.score_entraineur)

    lignes = []

    # ── Analyse du cheval ──
    if resultat.nb_courses_cheval == 0:
        lignes.append("⚠️ Aucune donnée de forme disponible pour ce cheval.")
    else:
        lignes.append(
            f"📊 **Forme du cheval** ({resultat.nb_courses_cheval} course(s) analysée(s)) : "
            f"{apprec_cheval} (score {resultat.score_cheval:.1f}/100)."
        )

    # ── Analyse du jockey ──
    if resultat.nb_courses_jockey == 0:
        lignes.append("⚠️ Aucune donnée disponible pour le jockey.")
    else:
        lignes.append(
            f"🏇 **Jockey** ({resultat.nb_courses_jockey} course(s)) : "
            f"{apprec_jockey} (score {resultat.score_jockey:.1f}/100)."
        )

    # ── Analyse de l'entraîneur ──
    if resultat.nb_courses_entraineur == 0:
        lignes.append("⚠️ Aucune donnée disponible pour l'entraîneur.")
    else:
        lignes.append(
            f"🎯 **Entraîneur** ({resultat.nb_courses_entraineur} course(s)) : "
            f"{apprec_entraineur} (score {resultat.score_entraineur:.1f}/100)."
        )

    # ── Verdict global ──
    sg = resultat.score_global
    if sg >= SEUIL_EXCELLENT:
        verdict = "🟢 **Candidat N°1** — Profil très solide, à surveiller de près !"
    elif sg >= SEUIL_BON:
        verdict = "🟡 **Bonne base** — Peut jouer les premiers rôles si les conditions lui conviennent."
    elif sg >= SEUIL_MOYEN:
        verdict = "🟠 **Outsider** — Profil moyen, peut créer la surprise mais peu fiable."
    else:
        verdict = "🔴 **Difficile à conseiller** — Forme insuffisante pour viser le podium."

    lignes.append(f"\n{verdict}")

    return "  \n".join(lignes)


# ──────────────────────────────────────────────
#  ANALYSE COMPLÈTE D'UNE COURSE
# ──────────────────────────────────────────────

def analyser_course(
    df: pd.DataFrame,
    type_course: str,
    poids_cheval:     float = POIDS_CHEVAL,
    poids_jockey:     float = POIDS_JOCKEY,
    poids_entraineur: float = POIDS_ENTRAINEUR,
) -> List[ResultatCheval]:
    """
    Analyse un DataFrame de chevaux et retourne la liste des résultats triée
    par score global décroissant (meilleur pronostic en premier).

    Colonnes attendues dans df :
    - 'N°'               (int/str)
    - 'Nom'              (str) — optionnel
    - 'Musique'          (str)
    - 'Musique Jockey'   (str)
    - 'Musique Entraîneur' (str)
    - 'Type Course'      (str) — optionnel, utilise type_course global si absent

    Paramètres facultatifs :
    - poids_cheval, poids_jockey, poids_entraineur : pondérations personnalisées
    """
    resultats: List[ResultatCheval] = []
    erreurs: List[str] = []

    for _, row in df.iterrows():
        try:
            numero = int(row.get("N°", 0))
            nom    = str(row.get("Nom", f"Cheval {numero}"))

            # Type de course : ligne individuelle ou valeur globale
            tc = str(row.get("Type Course", type_course))
            if tc.strip() == "" or tc.lower() in ("nan", "none"):
                tc = type_course

            # ── Calcul des scores ──
            sc_cheval,    nb_c = calculer_score_musique(str(row.get("Musique", "")),            tc)
            sc_jockey,    nb_j = calculer_score_musique(str(row.get("Musique Jockey", "")),     tc)
            sc_entrain,   nb_e = calculer_score_musique(str(row.get("Musique Entraîneur", "")), tc)

            sg = calculer_score_global(
                sc_cheval, sc_jockey, sc_entrain,
                poids_cheval, poids_jockey, poids_entraineur,
            )

            res = ResultatCheval(
                numero            = numero,
                nom               = nom,
                score_cheval      = sc_cheval,
                score_jockey      = sc_jockey,
                score_entraineur  = sc_entrain,
                score_global      = sg,
                nb_courses_cheval     = nb_c,
                nb_courses_jockey     = nb_j,
                nb_courses_entraineur = nb_e,
                commentaire       = "",
            )
            # Génération du commentaire après avoir construit l'objet
            res.commentaire = generer_commentaire(res)
            resultats.append(res)

        except Exception as exc:
            erreurs.append(f"Erreur cheval N°{row.get('N°', '?')} : {exc}")

    # ── Tri par score global décroissant ──
    resultats.sort(key=lambda r: r.score_global, reverse=True)
    for rang, res in enumerate(resultats, start=1):
        res.rang_pronostic = rang

    return resultats, erreurs


# ──────────────────────────────────────────────
#  EXPORT EN DATAFRAME
# ──────────────────────────────────────────────

def resultats_vers_dataframe(resultats: List[ResultatCheval]) -> pd.DataFrame:
    """
    Convertit la liste de ResultatCheval en DataFrame Pandas pour affichage.
    """
    lignes = []
    for r in resultats:
        lignes.append({
            "Rang 🏆":           r.rang_pronostic,
            "N°":                r.numero,
            "Nom":               r.nom,
            "Score Global":      r.score_global,
            "Score Cheval":      r.score_cheval,
            "Score Jockey":      r.score_jockey,
            "Score Entraîneur":  r.score_entraineur,
            "Commentaire":       r.commentaire,
        })
    return pd.DataFrame(lignes)


# ──────────────────────────────────────────────
#  DONNÉES D'EXEMPLE
# ──────────────────────────────────────────────

EXEMPLE_DONNEES = pd.DataFrame({
    "N°":                  [1,           2,           3,           4,           5],
    "Nom":                 ["Tornado",   "Éclair",    "Mistral",   "Neptune",   "Orion"],
    "Musique":             ["3p1p2p1p3p","6a5a4a3a2a","1p1p1p2p1p","5p7p4p8p6p","2p3p2p1p2p"],
    "Musique Jockey":      ["2p1p3p2p",  "4a3a5a2a",  "1p2p1p3p",  "6p5p4p3p",  "1p1p2p3p"],
    "Musique Entraîneur":  ["2p3p1p4p",  "3a2a4a3a",  "1p1p2p1p",  "4p5p3p6p",  "2p2p1p3p"],
    "Type Course":         ["Plat",      "Attelé",    "Plat",      "Plat",      "Plat"],
})


def valider_dataframe(df: pd.DataFrame) -> List[str]:
    """
    Vérifie que le DataFrame contient les colonnes minimales obligatoires.
    Retourne la liste des erreurs (vide si tout est OK).
    """
    colonnes_obligatoires = ["N°", "Musique", "Musique Jockey", "Musique Entraîneur"]
    manquantes = [c for c in colonnes_obligatoires if c not in df.columns]
    erreurs = []
    if manquantes:
        erreurs.append(f"Colonnes manquantes : {', '.join(manquantes)}")

    # Vérification des lignes vides
    if df.empty:
        erreurs.append("Le tableau est vide. Veuillez saisir au moins un cheval.")

    return erreurs


# ──────────────────────────────────────────────
#  STATISTIQUES RAPIDES
# ──────────────────────────────────────────────

def statistiques_course(resultats: List[ResultatCheval]) -> Dict:
    """
    Calcule quelques statistiques globales sur la course analysée.
    """
    if not resultats:
        return {}

    scores = [r.score_global for r in resultats]
    return {
        "Nombre de partants": len(resultats),
        "Score moyen":        round(np.mean(scores), 2),
        "Score max":          round(max(scores), 2),
        "Score min":          round(min(scores), 2),
        "Écart-type":         round(np.std(scores), 2),
        "Favori N°":          resultats[0].numero,
        "Favori Nom":         resultats[0].nom,
    }
