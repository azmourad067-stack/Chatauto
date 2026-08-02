# -*- coding: utf-8 -*-
"""
Loto FDJ — Explorateur statistique & générateur de combinaison
================================================================

CADRE MÉTHODOLOGIQUE (à lire avant le code)
---------------------------------------------
Les tirages du Loto français sont réalisés avec des machines certifiées,
scellées et contrôlées par huissier à chaque tirage, précisément pour
garantir un processus i.i.d. (indépendant et identiquement distribué) :
chaque numéro a, à chaque tirage, la même probabilité de sortir,
indépendamment de l'historique. C'est une conséquence directe de la
conception du système, pas une hypothèse statistique à tester.

Conséquence mathématique directe : aucune méthode statistique appliquée
à l'historique des tirages ne peut légitimement augmenter la probabilité
de deviner le tirage futur. La probabilité de gagner le gros lot reste
strictement C(49,5) x 10 = 19 068 840 pour N'IMPORTE QUELLE combinaison,
y compris celle générée ici.

Cette application fait donc DEUX choses honnêtes et utiles :
1. Une vraie analyse statistique descriptive de l'historique fourni
   (fréquences, écarts, co-occurrences, test du chi² d'ajustement à
   l'uniforme, tendances temporelles) — utile pour comprendre les
   données, PAS pour prédire.
2. Un générateur de combinaison basé sur des heuristiques
   documentées et couramment utilisées à titre récréatif (numéros
   "chauds/froids", numéros "en retard") — présenté comme un jeu
   statistique, avec un rappel explicite à chaque étape que cela
   n'améliore pas les chances réelles de gain.

Auteur : généré avec Claude (Anthropic) — data science / Streamlit
"""

import io
import itertools
from collections import Counter
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats

# ----------------------------------------------------------------------------
# CONFIGURATION GÉNÉRALE DE LA PAGE
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Loto FDJ — Explorateur statistique",
    page_icon="🎲",
    layout="wide",
)

BOULE_MIN, BOULE_MAX = 1, 49          # plage valide des boules principales
CHANCE_MIN, CHANCE_MAX = 1, 10        # plage valide du numéro chance
NB_BOULES = 5                          # nombre de boules tirées par tirage

REQUIRED_COLUMNS = [
    "date_de_tirage",
    "boule_1", "boule_2", "boule_3", "boule_4", "boule_5",
    "numero_chance",
]
OPTIONAL_COLUMNS = [
    "combinaison_gagnante_en_ordre_croissant",
    "boule_1_second_tirage", "boule_2_second_tirage",
    "boule_3_second_tirage", "boule_4_second_tirage",
    "boule_5_second_tirage",
]

BOULE_COLS = ["boule_1", "boule_2", "boule_3", "boule_4", "boule_5"]


# ==============================================================================
# 1. CHARGEMENT ET VALIDATION DU CSV
# ==============================================================================
def load_and_validate_csv(uploaded_file):
    """
    Charge le CSV de tirages FDJ et valide sa structure.

    Retourne (df, warnings) si succès, lève ValueError sinon.
    - Sépare correctement le CSV (';' comme séparateur, format FDJ historique).
    - Convertit date_de_tirage en datetime (format JJ/MM/AAAA).
    - Vérifie la présence des colonnes obligatoires.
    - Valide les plages de valeurs : boules dans [1,49], numero_chance dans [1,10].
    - Supprime les colonnes fantômes ('Unnamed: ...') issues d'un ';' final.
    """
    warnings = []

    # --- Lecture brute -------------------------------------------------------
    try:
        raw_bytes = uploaded_file.read()
        # on tente utf-8 puis latin-1 (fichiers FDJ historiques parfois en latin-1)
        for enc in ("utf-8", "latin-1"):
            try:
                text = raw_bytes.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("Impossible de décoder le fichier (encodage non reconnu).")

        df = pd.read_csv(io.StringIO(text), sep=";")
    except pd.errors.EmptyDataError:
        raise ValueError("Le fichier CSV est vide.")
    except pd.errors.ParserError as e:
        raise ValueError(f"Erreur de parsing CSV — vérifiez le séparateur (';' attendu) : {e}")
    except Exception as e:
        raise ValueError(f"Impossible de lire le fichier : {e}")

    # --- Nettoyage colonnes fantômes -----------------------------------------
    df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
    df.columns = df.columns.str.strip()

    # --- Vérification des colonnes obligatoires -------------------------------
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "Colonnes obligatoires manquantes dans le CSV : "
            + ", ".join(missing)
            + ". Le fichier doit être au format historique FDJ (séparateur ';', "
              "colonnes date_de_tirage, boule_1..boule_5, numero_chance)."
        )

    if not any(c in df.columns for c in OPTIONAL_COLUMNS):
        warnings.append(
            "Aucune colonne optionnelle attendue (combinaison_gagnante_..., "
            "second tirage) détectée — l'analyse se limitera au tirage principal."
        )

    # --- Conversion de la date --------------------------------------------
    df["date_de_tirage"] = pd.to_datetime(
        df["date_de_tirage"], format="%d/%m/%Y", errors="coerce"
    )
    n_bad_dates = df["date_de_tirage"].isna().sum()
    if n_bad_dates:
        warnings.append(
            f"{n_bad_dates} ligne(s) avec une date invalide ont été ignorées."
        )
    df = df.dropna(subset=["date_de_tirage"])

    # --- Conversion numérique + validation des plages -------------------------
    numeric_cols = BOULE_COLS + ["numero_chance"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    n_before = len(df)
    valid_boules = df[BOULE_COLS].apply(
        lambda s: s.between(BOULE_MIN, BOULE_MAX)
    ).all(axis=1)
    valid_chance = df["numero_chance"].between(CHANCE_MIN, CHANCE_MAX)
    valid_mask = valid_boules & valid_chance & df[numeric_cols].notna().all(axis=1)

    n_invalid = (~valid_mask).sum()
    if n_invalid:
        warnings.append(
            f"{n_invalid} ligne(s) écartées : boules hors de [{BOULE_MIN}-{BOULE_MAX}] "
            f"ou numéro chance hors de [{CHANCE_MIN}-{CHANCE_MAX}]."
        )
    df = df[valid_mask].copy()

    # --- Vérification cohérence : 5 boules distinctes par tirage --------------
    n_dup = (df[BOULE_COLS].nunique(axis=1) != NB_BOULES).sum()
    if n_dup:
        warnings.append(
            f"{n_dup} tirage(s) présentent des doublons entre boule_1..boule_5 "
            "(données possiblement corrompues) — conservés mais à vérifier."
        )

    if df.empty:
        raise ValueError("Aucune ligne valide après nettoyage — vérifiez le contenu du fichier.")

    # --- Tri chronologique (plus ancien -> plus récent) pour l'analyse d'écarts
    df = df.sort_values("date_de_tirage").reset_index(drop=True)

    for col in numeric_cols:
        df[col] = df[col].astype(int)

    return df, warnings


# ==============================================================================
# 2. MODULE D'ANALYSE STATISTIQUE
# ==============================================================================
def compute_frequencies(df):
    """
    Fréquence d'apparition de chaque numéro (1-49) parmi boule_1..boule_5,
    et de chaque numéro chance (1-10).

    Base théorique : sous l'hypothèse d'un tirage uniforme sans remise de
    5 boules parmi 49, chaque numéro a une probabilité marginale de sortie
    de 5/49 à chaque tirage (loi hypergéométrique marginale). Sur N tirages,
    la fréquence attendue de chaque numéro est donc N * 5/49.
    """
    all_boules = df[BOULE_COLS].values.flatten()
    freq_boules = pd.Series(Counter(all_boules)).reindex(
        range(BOULE_MIN, BOULE_MAX + 1), fill_value=0
    ).sort_index()

    freq_chance = pd.Series(Counter(df["numero_chance"])).reindex(
        range(CHANCE_MIN, CHANCE_MAX + 1), fill_value=0
    ).sort_index()

    return freq_boules, freq_chance


def compute_gaps(df):
    """
    Écart (gap) = nombre de tirages écoulés depuis la dernière apparition
    de chaque numéro, calculé à partir du tirage le plus récent.

    Base théorique : sous indépendance, le nombre de tirages entre deux
    apparitions successives d'un numéro donné suit approximativement une
    loi géométrique de paramètre p = 5/49 (probabilité de sortie par
    tirage). L'espérance de cet écart est donc E[gap] = 1/p = 49/5 = 9.8
    tirages. On calcule ici l'écart courant (depuis le dernier tirage
    disponible) pour comparaison avec cette espérance théorique.
    """
    n = len(df)
    last_seen = {}
    for num in range(BOULE_MIN, BOULE_MAX + 1):
        last_seen[num] = None

    # on parcourt du plus récent au plus ancien pour trouver le dernier tirage
    for idx in range(n - 1, -1, -1):
        drawn = set(df.loc[idx, BOULE_COLS])
        for num in drawn:
            if last_seen[num] is None:
                last_seen[num] = n - 1 - idx  # nombre de tirages depuis la sortie

    gaps = pd.Series(
        {num: (last_seen[num] if last_seen[num] is not None else n)
         for num in range(BOULE_MIN, BOULE_MAX + 1)}
    ).sort_index()

    # même logique pour le numéro chance (p = 1/10, E[gap] = 10)
    last_seen_chance = {c: None for c in range(CHANCE_MIN, CHANCE_MAX + 1)}
    for idx in range(n - 1, -1, -1):
        c = df.loc[idx, "numero_chance"]
        if last_seen_chance[c] is None:
            last_seen_chance[c] = n - 1 - idx

    gaps_chance = pd.Series(
        {c: (last_seen_chance[c] if last_seen_chance[c] is not None else n)
         for c in range(CHANCE_MIN, CHANCE_MAX + 1)}
    ).sort_index()

    return gaps, gaps_chance


def chi_square_uniformity_test(freq_boules, n_tirages):
    """
    Test du chi² d'ajustement à la loi uniforme (marginale des boules).

    H0 : chaque numéro a une probabilité marginale de sortie de 5/49,
    identique pour tous les numéros (conforme à un tirage équitable).
    H1 : les fréquences observées s'écartent significativement de cette
    distribution attendue.

    Statistique : chi2 = somme( (observé - attendu)^2 / attendu )
    ~ loi du chi² à (49-1) degrés de liberté sous H0.

    Ce test est le vrai outil scientifique pour répondre à la question
    « y a-t-il un biais mécanique détectable ? » : un p-value élevé
    (> 0.05) indique qu'on NE PEUT PAS rejeter l'hypothèse d'équité —
    autrement dit, l'historique ne fournit aucune preuve statistique
    d'un biais exploitable.
    """
    expected = np.full(BOULE_MAX - BOULE_MIN + 1, n_tirages * NB_BOULES / BOULE_MAX)
    chi2_stat, p_value = stats.chisquare(f_obs=freq_boules.values, f_exp=expected)
    return chi2_stat, p_value, expected


def compute_cooccurrence(df, top_n=10):
    """
    Matrice de co-occurrence : nombre de fois où chaque paire de numéros
    est sortie ensemble dans le même tirage.

    Base théorique : sous indépendance des tirages et tirage sans remise
    au sein d'un même tirage, la probabilité théorique qu'une paire
    (i, j) donnée sorte ensemble dans un tirage est C(47,3)/C(49,5)
    = 10/(49*48/2) ≈ 0.0085. On compare la fréquence observée de chaque
    paire à cette attente théorique pour identifier d'éventuelles paires
    sur- ou sous-représentées (à interpréter avec prudence : sur C(49,2)
    = 1176 paires possibles, des écarts apparents sont attendus par pur
    effet du nombre de comparaisons — cf. problème des comparaisons
    multiples).
    """
    pair_counts = Counter()
    for _, row in df[BOULE_COLS].iterrows():
        for pair in itertools.combinations(sorted(row.values), 2):
            pair_counts[pair] += 1

    pair_df = pd.DataFrame(
        [(a, b, c) for (a, b), c in pair_counts.items()],
        columns=["numero_1", "numero_2", "occurrences"],
    ).sort_values("occurrences", ascending=False).reset_index(drop=True)

    return pair_df.head(top_n), pair_counts


def compute_temporal_trend(df, window=50):
    """
    Analyse de tendance temporelle : moyenne mobile de la somme des 5
    boules tirées par tirage, sur une fenêtre glissante de `window`
    tirages.

    Base théorique : la somme de 5 tirages sans remise parmi 1..49 a une
    espérance théorique constante de 5 * (49+1)/2 = 125, quel que soit
    le moment. Une moyenne mobile qui oscille autour de cette valeur sans
    dérive systématique est un signe (parmi d'autres) de stabilité du
    processus — l'absence de dérive est ATTENDUE sous indépendance, ce
    n'est pas un pattern exploitable.
    """
    sums = df[BOULE_COLS].sum(axis=1)
    rolling_mean = sums.rolling(window=window, min_periods=1).mean()
    theoretical_mean = NB_BOULES * (BOULE_MAX + BOULE_MIN) / 2
    return sums, rolling_mean, theoretical_mean


def hot_cold_zscores(freq_boules, n_tirages):
    """
    Score standardisé (z-score) de la fréquence de chaque numéro par
    rapport à sa distribution attendue.

    Sous H0 (tirage équitable), le nombre d'apparitions d'un numéro
    donné sur N tirages suit une loi Binomiale(N, 5/49), d'espérance
    N*5/49 et de variance N*(5/49)*(44/49). Le z-score standardise
    l'écart observé par cet écart-type théorique, ce qui permet de
    comparer les numéros entre eux sur une échelle commune plutôt que
    de comparer des comptages bruts.
    """
    p = NB_BOULES / BOULE_MAX
    expected = n_tirages * p
    std = np.sqrt(n_tirages * p * (1 - p))
    z_scores = (freq_boules - expected) / std
    return z_scores


# ==============================================================================
# 3. MODULE DE GÉNÉRATION DE COMBINAISON (heuristique récréative)
# ==============================================================================
#
# IMPORTANT — lire avant d'utiliser ce module :
# Les poids ci-dessous transforment des statistiques descriptives réelles
# (fréquence, écart) en probabilités d'échantillonnage pour PROPOSER une
# combinaison de façon non-uniforme. Cela ne change rien à la probabilité
# RÉELLE de gain, qui reste C(49,5)*10 = 19 068 840 pour toute combinaison,
# y compris celle-ci. Ce module existe pour transformer une intuition
# ("je préfère miser sur des numéros qui ont un profil statistique
# particulier") en un choix reproductible et documenté, pas pour battre
# le hasard.
#
# Deux heuristiques classiques, réellement utilisées dans la littérature
# récréative sur les loteries, sont proposées :
#   - "Chauds"  : sur-pondérer les numéros historiquement plus fréquents.
#     (hypothèse implicite, non prouvée : un léger biais mécanique
#     persistant — à ne pas confondre avec une preuve statistique.)
#   - "En retard" : sur-pondérer les numéros dont l'écart courant dépasse
#     l'écart théorique moyen (49/5 ≈ 9.8) — c'est la version numérique
#     du "gambler's fallacy" (croire qu'un numéro est "dû"). Statistiquement
#     invalide sous indépendance, mais très répandue : elle est incluse ici
#     à des fins d'exploration transparente, pas parce qu'elle est fondée.
#
def build_selection_weights(freq_boules, gaps, n_tirages, strategy_mix):
    """
    Construit un vecteur de poids (1 par numéro 1-49) combinant :
      - un score "chaud" normalisé (fréquence observée)
      - un score "en retard" normalisé (écart courant / écart théorique)
    selon le curseur strategy_mix in [0, 1] :
      0.0 = 100% basé sur la fréquence ("chauds")
      1.0 = 100% basé sur l'écart ("en retard")
    """
    hot_score = freq_boules.astype(float).copy()
    hot_score = (hot_score - hot_score.min()) + 1.0  # évite les poids nuls

    theoretical_gap = BOULE_MAX / NB_BOULES
    due_score = gaps.astype(float) / theoretical_gap
    due_score = (due_score - due_score.min()) + 1.0

    hot_norm = hot_score / hot_score.sum()
    due_norm = due_score / due_score.sum()

    weights = (1 - strategy_mix) * hot_norm + strategy_mix * due_norm
    weights = weights / weights.sum()
    return weights


def generate_prediction(freq_boules, gaps, freq_chance, gaps_chance,
                         n_tirages, strategy_mix=0.5, seed=None):
    """
    Génère une combinaison de 5 numéros + 1 numéro chance par tirage
    aléatoire pondéré SANS remise (np.random.choice, replace=False),
    en utilisant les poids définis par build_selection_weights.

    L'utilisation d'un tirage pondéré aléatoire (plutôt qu'un simple
    top-5 des scores) est délibérée : elle évite de proposer
    systématiquement la même combinaison "optimale" en apparence, ce qui
    serait statistiquement malhonnête étant donné qu'aucune combinaison
    n'est réellement supérieure à une autre.
    """
    rng = np.random.default_rng(seed)

    weights = build_selection_weights(freq_boules, gaps, n_tirages, strategy_mix)
    numeros = rng.choice(
        freq_boules.index.values, size=NB_BOULES, replace=False, p=weights.values
    )
    numeros = sorted(int(x) for x in numeros)

    # numéro chance : même logique de mélange fréquence / écart
    theoretical_gap_chance = CHANCE_MAX / 1
    hot_c = freq_chance.astype(float) - freq_chance.min() + 1.0
    due_c = (gaps_chance.astype(float) / theoretical_gap_chance) - \
            (gaps_chance.astype(float) / theoretical_gap_chance).min() + 1.0
    w_c = (1 - strategy_mix) * (hot_c / hot_c.sum()) + strategy_mix * (due_c / due_c.sum())
    w_c = w_c / w_c.sum()
    chance = int(rng.choice(freq_chance.index.values, size=1, p=w_c.values)[0])

    detail = pd.DataFrame({
        "numero": numeros,
        "frequence_observee": [int(freq_boules[n]) for n in numeros],
        "ecart_courant": [int(gaps[n]) for n in numeros],
        "ecart_theorique": [round(BOULE_MAX / NB_BOULES, 1)] * NB_BOULES,
        "poids_selection_%": [round(weights[n] * 100, 2) for n in numeros],
    })

    return numeros, chance, detail


# ==============================================================================
# 4. INTERFACE STREAMLIT
# ==============================================================================
def main():
    st.title("🎲 Loto FDJ — Explorateur statistique")
    st.caption(
        "Analyse descriptive d'un historique de tirages + générateur de "
        "combinaison basé sur des heuristiques statistiques transparentes."
    )

    with st.expander("⚠️ Ce que cette application fait — et ne fait pas", expanded=True):
        st.markdown(
            """
Les tirages du Loto FDJ sont réalisés avec des machines certifiées et
contrôlées à chaque tirage pour garantir un processus **indépendant et
équiprobable**. Mathématiquement, cela signifie qu'**aucune analyse de
l'historique ne peut augmenter la probabilité réelle de gagner**
(elle reste 1 chance sur 19 068 840 pour le gros lot, quelle que soit
la combinaison jouée, y compris celle proposée ici).

Cette application propose deux choses honnêtes :
- **Une vraie analyse statistique** de votre historique (fréquences,
  écarts, co-occurrences, test d'équité par chi²) — utile pour
  comprendre les données, pas pour prédire l'avenir.
- **Un générateur de combinaison** basé sur des heuristiques
  récréatives documentées (numéros "chauds", numéros "en retard"),
  avec une transparence totale sur ce qu'il fait mathématiquement.

*Si le jeu vous préoccupe pour vous ou un proche : Joueurs Info
Service — 09 74 75 13 13 (appel non surtaxé, anonyme et gratuit).*
            """
        )

    # --- Upload -----------------------------------------------------------
    st.sidebar.header("📁 Données")
    uploaded_file = st.sidebar.file_uploader(
        "Charger un historique de tirages (CSV, séparateur ';')",
        type=["csv"],
        help="Format attendu : export historique FDJ (ex. loto_201911.csv) avec "
             "colonnes date_de_tirage, boule_1..boule_5, numero_chance.",
    )

    if uploaded_file is None:
        st.info("⬅️ Chargez un fichier CSV d'historique dans la barre latérale pour commencer.")
        st.markdown(
            "**Format attendu** (extrait) : `date_de_tirage;boule_1;boule_2;"
            "boule_3;boule_4;boule_5;numero_chance;combinaison_gagnante_en_ordre_croissant;...`"
        )
        return

    # --- Chargement + validation -------------------------------------------
    try:
        df, warnings_list = load_and_validate_csv(uploaded_file)
    except ValueError as e:
        st.error(f"❌ Erreur de chargement : {e}")
        st.stop()

    for w in warnings_list:
        st.warning(f"⚠️ {w}")

    n_tirages = len(df)
    st.success(
        f"✅ {n_tirages} tirages valides chargés — du "
        f"{df['date_de_tirage'].min().strftime('%d/%m/%Y')} au "
        f"{df['date_de_tirage'].max().strftime('%d/%m/%Y')}."
    )

    # --- Calculs communs ----------------------------------------------------
    freq_boules, freq_chance = compute_frequencies(df)
    gaps, gaps_chance = compute_gaps(df)
    z_scores = hot_cold_zscores(freq_boules, n_tirages)
    chi2_stat, p_value, expected = chi_square_uniformity_test(freq_boules, n_tirages)

    tab_analyse, tab_prediction, tab_methodo = st.tabs(
        ["📊 Analyse statistique", "🎯 Générateur de combinaison", "📖 Méthodologie & limites"]
    )

    # =========================================================================
    # ONGLET ANALYSE
    # =========================================================================
    with tab_analyse:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Fréquence d'apparition (boules 1-49)")
            st.bar_chart(freq_boules)
            st.caption(
                f"Fréquence théorique attendue par numéro : "
                f"{n_tirages * NB_BOULES / BOULE_MAX:.1f} apparitions sur {n_tirages} tirages."
            )

        with col2:
            st.subheader("Écart courant (tirages depuis la dernière sortie)")
            st.bar_chart(gaps)
            st.caption(f"Écart théorique moyen attendu : {BOULE_MAX / NB_BOULES:.1f} tirages.")

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Fréquence du numéro chance (1-10)")
            st.bar_chart(freq_chance)
        with col4:
            st.subheader("Écart courant — numéro chance")
            st.bar_chart(gaps_chance)

        st.subheader("Test d'équité (chi² d'ajustement à l'uniforme)")
        cA, cB, cC = st.columns(3)
        cA.metric("Statistique χ²", f"{chi2_stat:.2f}")
        cB.metric("p-value", f"{p_value:.4f}")
        cC.metric("Degrés de liberté", BOULE_MAX - 1)
        if p_value > 0.05:
            st.info(
                "➡️ p-value > 0.05 : on ne peut **pas rejeter** l'hypothèse d'équité. "
                "L'historique fourni ne montre aucune preuve statistique d'un biais "
                "exploitable — résultat attendu pour un système correctement audité."
            )
        else:
            st.warning(
                "➡️ p-value ≤ 0.05 : écart statistiquement détectable par rapport à "
                "l'uniforme parfaite. Sur une longue série de tests, des faux positifs "
                "occasionnels sont statistiquement normaux (risque α = 5%) — ce résultat "
                "seul ne constitue pas une preuve de biais mécanique réel."
            )

        st.subheader("Numéros chauds / froids (z-score de fréquence)")
        z_df = pd.DataFrame({
            "numero": z_scores.index,
            "z_score": z_scores.values,
            "frequence": freq_boules.values,
        }).sort_values("z_score", ascending=False)
        col5, col6 = st.columns(2)
        col5.markdown("**Top 5 « chauds »**")
        col5.dataframe(z_df.head(5).reset_index(drop=True), hide_index=True)
        col6.markdown("**Top 5 « froids »**")
        col6.dataframe(z_df.tail(5).sort_values("z_score").reset_index(drop=True), hide_index=True)

        st.subheader("Paires de numéros les plus fréquentes (co-occurrence)")
        top_pairs, _ = compute_cooccurrence(df, top_n=15)
        st.dataframe(top_pairs, hide_index=True)
        st.caption(
            "Probabilité théorique qu'une paire donnée sorte ensemble à un tirage : "
            f"≈ {10 / (49 * 48 / 2):.4f}. Avec {49*48//2} paires possibles, des écarts "
            "apparents sont statistiquement attendus par simple effet du nombre de "
            "comparaisons effectuées (voir onglet Méthodologie)."
        )

        st.subheader("Tendance temporelle — somme des 5 boules par tirage")
        sums, rolling_mean, theoretical_mean = compute_temporal_trend(df)
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(df["date_de_tirage"], sums, alpha=0.25, label="Somme par tirage")
        ax.plot(df["date_de_tirage"], rolling_mean, color="crimson", label="Moyenne mobile (50 tirages)")
        ax.axhline(theoretical_mean, color="black", linestyle="--", label=f"Moyenne théorique ({theoretical_mean:.0f})")
        ax.legend()
        ax.set_ylabel("Somme des 5 boules")
        st.pyplot(fig)
        st.caption(
            "Une moyenne mobile stable autour de la valeur théorique est le "
            "comportement ATTENDU d'un processus équitable — ce n'est pas un signal "
            "exploitable pour prédire le prochain tirage."
        )

    # =========================================================================
    # ONGLET PRÉDICTION
    # =========================================================================
    with tab_prediction:
        st.subheader("Générateur de combinaison")
        st.caption(
            "Cet outil propose une combinaison via un tirage aléatoire pondéré par "
            "des statistiques réelles de votre historique. Cela ne change pas la "
            "probabilité de gain — voir l'encadré d'avertissement ci-dessus."
        )

        strategy_mix = st.slider(
            "Curseur d'heuristique : 0 = privilégier les numéros « chauds » "
            "(fréquents) — 1 = privilégier les numéros « en retard » (écart élevé)",
            min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        )
        seed = st.number_input(
            "Graine aléatoire (optionnel, pour reproduire exactement la même combinaison)",
            min_value=0, value=0, step=1,
        )
        use_seed = st.checkbox("Fixer la graine aléatoire", value=False)

        if st.button("🎲 Générer une combinaison", type="primary"):
            numeros, chance, detail = generate_prediction(
                freq_boules, gaps, freq_chance, gaps_chance, n_tirages,
                strategy_mix=strategy_mix, seed=(seed if use_seed else None),
            )

            st.markdown("### Combinaison proposée")
            badge_cols = st.columns(6)
            for i, num in enumerate(numeros):
                badge_cols[i].markdown(
                    f"<div style='background-color:#1f77b4;color:white;"
                    f"border-radius:50%;width:60px;height:60px;display:flex;"
                    f"align-items:center;justify-content:center;font-size:22px;"
                    f"font-weight:bold;margin:auto;'>{num}</div>",
                    unsafe_allow_html=True,
                )
            badge_cols[5].markdown(
                f"<div style='background-color:#d62728;color:white;"
                f"border-radius:50%;width:60px;height:60px;display:flex;"
                f"align-items:center;justify-content:center;font-size:22px;"
                f"font-weight:bold;margin:auto;'>{chance}</div>",
                unsafe_allow_html=True,
            )
            st.caption("(le dernier numéro, en rouge, est le numéro chance)")

            st.markdown("### Détail du raisonnement statistique")
            st.dataframe(detail, hide_index=True)
            st.markdown(
                f"""
- **Fréquence observée** : nombre de fois où ce numéro est sorti sur les
  {n_tirages} tirages de votre historique.
- **Écart courant** : nombre de tirages depuis sa dernière sortie
  (écart théorique moyen attendu : {BOULE_MAX/NB_BOULES:.1f} tirages).
- **Poids de sélection** : probabilité utilisée pour le tirage pondéré,
  combinant fréquence et écart selon le curseur choisi ({strategy_mix:.2f}).
                """
            )
            st.info(
                "🎯 Probabilité réelle de gain du gros lot avec cette combinaison : "
                "1 / 19 068 840 — rigoureusement identique à celle de n'importe "
                "quelle autre combinaison, y compris une combinaison choisie au hasard."
            )

    # =========================================================================
    # ONGLET MÉTHODOLOGIE
    # =========================================================================
    with tab_methodo:
        st.subheader("Cadre mathématique")
        st.markdown(
            """
**Pourquoi l'historique ne permet pas de prédire le tirage suivant**

Un tirage du Loto consiste à extraire 5 boules sans remise parmi 49,
puis 1 numéro chance parmi 10, au moyen d'un système mécanique certifié
et contrôlé à chaque tirage. Le modèle probabiliste correspondant est
un tirage **uniforme sans remise** (loi hypergéométrique pour les 5
boules, loi uniforme discrète pour le numéro chance), **indépendant
d'un tirage à l'autre**.

Sous ce modèle :
- La probabilité de sortie de chaque numéro à un tirage donné est
  constante (5/49), quel que soit l'historique passé.
- Le fait qu'un numéro ne soit pas sorti depuis longtemps
  ("il est en retard") n'augmente **pas** sa probabilité de sortie au
  tirage suivant — c'est le raisonnement fallacieux connu sous le nom
  de *gambler's fallacy* (illusion du joueur).
- De même, un numéro historiquement "chaud" n'a pas plus de chances de
  ressortir : sous indépendance, chaque tirage repart de zéro.

**Ce que les outils statistiques de cette application permettent
réellement de faire** : décrire un jeu de données (combien de fois
chaque numéro est sorti, quels écarts, quelles corrélations apparentes),
et **tester** si ces observations sont compatibles avec l'hypothèse
d'équité (test du chi²). Ils ne permettent pas de prédire un tirage
futur, car la théorie même du tirage (indépendance, équiprobabilité)
l'exclut par construction.

**Sur le problème des comparaisons multiples** : avec 49 numéros et
1176 paires possibles, il est statistiquement normal d'observer, par
pur hasard, quelques numéros ou paires en apparence "anormaux" même
sous un processus parfaitement équitable. Le test du chi² global (et
non les extrêmes isolés) est l'indicateur le plus fiable fourni ici.

**Sources générales sur les méthodes évoquées** : théorie des tests
d'ajustement (chi²), lois hypergéométrique et géométrique appliquées
aux tirages sans remise, et littérature sur le *gambler's fallacy* en
psychologie du jugement statistique — concepts standards de probabilité
et statistique, enseignés indépendamment de tout contexte de loterie.
            """
        )


if __name__ == "__main__":
    main()
