import streamlit as st
import pandas as pd
import numpy as np
import random
from collections import Counter
from itertools import combinations

st.set_page_config(page_title="Générateur de suite logique", page_icon="🔢", layout="wide")

# ----------------------------------------------------------------------
# Fonctions utilitaires
# ----------------------------------------------------------------------

def parse_excel(file) -> list[list[int]]:
    """Lit un fichier Excel dont chaque ligne contient une suite du type
    '5 - 6 - 7 - 9 - 12' (une cellule = une suite complète, séparée par des tirets)."""
    df = pd.read_excel(file, header=None)
    sequences = []
    for val in df.iloc[:, 0].dropna():
        val = str(val)
        # Sépare sur '-' (avec ou sans espaces) ou ',' ou ';'
        parts = [p.strip() for p in val.replace(";", "-").replace(",", "-").split("-")]
        nums = []
        for p in parts:
            if p.isdigit():
                nums.append(int(p))
        if nums:
            sequences.append(nums)
    return sequences


def compute_stats(sequences: list[list[int]]):
    all_numbers = [n for seq in sequences for n in seq]
    freq = Counter(all_numbers)
    number_range = (min(all_numbers), max(all_numbers))

    # Numéros "en retard" : nombre de tirages depuis leur dernière apparition
    last_seen = {}
    for idx, seq in enumerate(sequences):
        for n in seq:
            last_seen[n] = idx
    total_draws = len(sequences)
    delay = {n: (total_draws - 1 - last_seen.get(n, -1)) for n in range(number_range[0], number_range[1] + 1)}

    # Paires de numéros qui reviennent souvent ensemble
    pair_counter = Counter()
    for seq in sequences:
        for a, b in combinations(sorted(seq), 2):
            pair_counter[(a, b)] += 1

    return {
        "freq": freq,
        "range": number_range,
        "delay": delay,
        "pairs": pair_counter,
        "total_draws": total_draws,
    }


def generate_sequence(stats, method: str, k: int = 5) -> list[int]:
    lo, hi = stats["range"]
    pool = list(range(lo, hi + 1))

    if method == "Fréquence pondérée":
        weights = [stats["freq"].get(n, 0) + 1 for n in pool]  # +1 pour éviter poids nul
        chosen = set()
        attempts = 0
        while len(chosen) < k and attempts < 1000:
            pick = random.choices(pool, weights=weights, k=1)[0]
            chosen.add(pick)
            attempts += 1
        return sorted(chosen)

    if method == "Numéros en retard":
        # Priorité aux numéros qui n'apparaissent plus depuis longtemps
        sorted_by_delay = sorted(pool, key=lambda n: stats["delay"].get(n, 0), reverse=True)
        top = sorted_by_delay[: max(k * 2, k)]
        return sorted(random.sample(top, k))

    if method == "Mix (chauds + en retard)":
        hot = [n for n, _ in stats["freq"].most_common(max(k, 6))]
        cold = sorted(pool, key=lambda n: stats["delay"].get(n, 0), reverse=True)[: max(k, 6)]
        combined = list(set(hot) | set(cold))
        if len(combined) < k:
            combined = pool
        return sorted(random.sample(combined, k))

    if method == "Paires fréquentes":
        chosen = set()
        top_pairs = [p for p, _ in stats["pairs"].most_common(15)]
        random.shuffle(top_pairs)
        for a, b in top_pairs:
            if len(chosen) >= k:
                break
            chosen.add(a)
            if len(chosen) < k:
                chosen.add(b)
        # Complète aléatoirement si besoin
        remaining = [n for n in pool if n not in chosen]
        while len(chosen) < k and remaining:
            chosen.add(remaining.pop(random.randrange(len(remaining))))
        return sorted(list(chosen)[:k])

    # Aléatoire pur (référence)
    return sorted(random.sample(pool, k))


# ----------------------------------------------------------------------
# Interface Streamlit
# ----------------------------------------------------------------------

st.title("🔢 Générateur de suite logique (5 chiffres)")

st.warning(
    "⚠️ **Avertissement important** : ce générateur repose uniquement sur des statistiques "
    "descriptives (fréquences, écarts, paires récurrentes) calculées sur les données que vous "
    "fournissez. Si votre suite provient d'un tirage aléatoire (loto, etc.), **aucune méthode ne "
    "peut réellement prédire le prochain tirage** — chaque tirage reste indépendant et "
    "équiprobable. Cet outil est proposé à titre exploratoire / ludique, pas comme un outil de "
    "prédiction fiable."
)

st.markdown(
    "Importez un fichier Excel où chaque ligne contient une suite de chiffres séparés par des "
    "tirets, par exemple : `5 - 6 - 7 - 9 - 12`."
)

uploaded_file = st.file_uploader("📂 Importer le fichier Excel (.xlsx)", type=["xlsx", "xls"])

if uploaded_file is not None:
    try:
        sequences = parse_excel(uploaded_file)
    except Exception as e:
        st.error(f"Erreur lors de la lecture du fichier : {e}")
        sequences = []

    if not sequences:
        st.error("Aucune suite valide n'a été trouvée dans le fichier.")
    else:
        st.success(f"{len(sequences)} suites chargées avec succès.")

        with st.expander("📋 Aperçu des données importées"):
            st.dataframe(
                pd.DataFrame(
                    {"Suite": [" - ".join(map(str, s)) for s in sequences]}
                ),
                use_container_width=True,
            )

        stats = compute_stats(sequences)
        lo, hi = stats["range"]

        st.subheader("📊 Analyse des patterns")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Fréquence d'apparition de chaque numéro**")
            freq_df = pd.DataFrame(
                {"Numéro": list(range(lo, hi + 1))}
            )
            freq_df["Fréquence"] = freq_df["Numéro"].map(lambda n: stats["freq"].get(n, 0))
            freq_df = freq_df.set_index("Numéro")
            st.bar_chart(freq_df)

        with col2:
            st.markdown("**Numéros les plus 'en retard' (absents depuis longtemps)**")
            delay_df = pd.DataFrame(
                {"Numéro": list(range(lo, hi + 1))}
            )
            delay_df["Retard (tirages)"] = delay_df["Numéro"].map(lambda n: stats["delay"].get(n, 0))
            delay_df = delay_df.sort_values("Retard (tirages)", ascending=False).set_index("Numéro")
            st.bar_chart(delay_df)

        st.markdown("**Top 10 des paires de numéros qui reviennent le plus souvent ensemble**")
        top_pairs = stats["pairs"].most_common(10)
        if top_pairs:
            pairs_df = pd.DataFrame(
                [{"Paire": f"{a} - {b}", "Occurrences": c} for (a, b), c in top_pairs]
            )
            st.dataframe(pairs_df, use_container_width=True, hide_index=True)

        st.divider()
        st.subheader("🎯 Générer une suite de 5 chiffres")

        method = st.selectbox(
            "Méthode de génération",
            [
                "Fréquence pondérée",
                "Numéros en retard",
                "Mix (chauds + en retard)",
                "Paires fréquentes",
                "Aléatoire pur (référence)",
            ],
            help=(
                "Fréquence pondérée : privilégie les numéros historiquement fréquents.\n"
                "Numéros en retard : privilégie les numéros absents depuis longtemps.\n"
                "Mix : combine numéros fréquents et numéros en retard.\n"
                "Paires fréquentes : s'appuie sur les paires de numéros qui reviennent souvent ensemble.\n"
                "Aléatoire pur : tirage sans aucun pattern, pour comparaison."
            ),
        )

        if st.button("🎲 Générer la suite", type="primary"):
            result = generate_sequence(stats, method, k=5)
            st.markdown("### Résultat")
            cols = st.columns(5)
            for c, n in zip(cols, result):
                c.metric(label="", value=n)
            st.caption(f"Méthode utilisée : {method}")

        st.divider()
        st.subheader("🔁 Générer plusieurs suites d'un coup")
        nb_suites = st.slider("Nombre de suites à générer", 1, 10, 3)
        if st.button("Générer plusieurs suites"):
            rows = []
            for _ in range(nb_suites):
                rows.append(" - ".join(map(str, generate_sequence(stats, method, k=5))))
            st.table(pd.DataFrame({"Suite générée": rows}))

else:
    st.info("👆 Importez un fichier Excel pour commencer l'analyse.")
