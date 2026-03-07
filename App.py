import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import itertools
from scipy.special import softmax
import re
import requests
from bs4 import BeautifulSoup

# ------------------------------------------------------------------------------
# Paramètres globaux
# ------------------------------------------------------------------------------
DECAY_FACTOR = 0.3
POINTS_MAPPING = {1:10, 2:8, 3:6, 4:5, 5:4, 6:3, 7:2, 8:1}
DEFAULT_POINT = 1
PENALTY_POINT = 0

WEIGHTS = {
    'plat': {
        'score_musique': 0.25,
        'age_score': 0.10,
        'gains_log': 0.15,
        'pct_driver': 0.10,
        'pct_entraineur': 0.10,
        'corde_score': 0.08,
        'regularite': 0.07,
        'nb_perf': 0.05,
        'experience': 0.05,
        'sexe_score': 0.05,
    },
    'attelé': {
        'score_musique': 0.30,
        'age_score': 0.10,
        'gains_log': 0.15,
        'pct_driver': 0.15,
        'pct_entraineur': 0.10,
        'corde_score': 0.00,
        'regularite': 0.10,
        'nb_perf': 0.05,
        'experience': 0.05,
        'sexe_score': 0.00,
    },
    'monté': {
        'score_musique': 0.30,
        'age_score': 0.10,
        'gains_log': 0.15,
        'pct_driver': 0.15,
        'pct_entraineur': 0.10,
        'corde_score': 0.00,
        'regularite': 0.10,
        'nb_perf': 0.05,
        'experience': 0.05,
        'sexe_score': 0.00,
    },
    'obstacle': {
        'score_musique': 0.25,
        'age_score': 0.15,
        'gains_log': 0.15,
        'pct_driver': 0.05,
        'pct_entraineur': 0.15,
        'corde_score': 0.00,
        'regularite': 0.10,
        'nb_perf': 0.05,
        'experience': 0.10,
        'sexe_score': 0.00,
    }
}

# ------------------------------------------------------------------------------
# Fonctions de parsing de la musique
# ------------------------------------------------------------------------------
def parse_musique(musique_str):
    if not isinstance(musique_str, str) or musique_str.strip() == '':
        return []
    performances = []
    for part in musique_str.strip().split():
        match = re.match(r'^(\d+)([a-zA-Z]*)', part)
        if match:
            place = int(match.group(1))
            suffix = match.group(2)
            if suffix and suffix.upper() in ['D', 'A']:
                points = PENALTY_POINT
            else:
                points = POINTS_MAPPING.get(place, DEFAULT_POINT)
            performances.append(points)
        else:
            performances.append(0)
    return performances

def score_musique(performances):
    if not performances:
        return 0
    weights = np.exp(-DECAY_FACTOR * np.arange(len(performances)))
    weights /= weights.sum()
    return np.sum(np.array(performances) * weights)

# ------------------------------------------------------------------------------
# Normalisation
# ------------------------------------------------------------------------------
def normalize_series(series, method='minmax'):
    if method == 'minmax':
        if series.max() == series.min():
            return pd.Series([0.5] * len(series))
        return (series - series.min()) / (series.max() - series.min())
    elif method == 'zscore':
        if series.std() == 0:
            return pd.Series([0] * len(series))
        return (series - series.mean()) / series.std()
    return series

# ------------------------------------------------------------------------------
# Construction des features
# ------------------------------------------------------------------------------
def compute_features(df_partants, course_type, distance):
    df = df_partants.copy()

    df['performances'] = df['musique'].apply(parse_musique)
    df['score_musique_raw'] = df['performances'].apply(score_musique)
    df['nb_perf'] = df['performances'].apply(len)

    def perf_std(perf):
        if len(perf) < 2:
            return 0
        return np.std(perf)
    df['regularite_raw'] = df['performances'].apply(perf_std)

    df['gains_log'] = np.log1p(df['gains'])

    def age_score(age):
        if course_type == 'plat':
            return np.exp(-((age - 4) ** 2) / 4)
        elif course_type == 'obstacle':
            return np.exp(-((age - 5.5) ** 2) / 6)
        else:
            return np.exp(-((age - 5) ** 2) / 5)
    df['age_score'] = df['age'].apply(age_score)

    if course_type == 'plat':
        max_corde = df['corde'].max()
        if max_corde > 0:
            df['corde_score'] = 1 - (df['corde'] - 1) / (max_corde - 1)
        else:
            df['corde_score'] = 0.5
    else:
        df['corde_score'] = 0.5

    df['sexe_score'] = 0.5
    df['pct_driver'] = df['pct_driver'] / 100.0
    df['pct_entraineur'] = df['pct_entraineur'] / 100.0

    features_to_norm = [
        'score_musique_raw', 'gains_log', 'nb_perf', 'regularite_raw',
        'age_score', 'corde_score', 'pct_driver', 'pct_entraineur', 'sexe_score'
    ]
    for f in features_to_norm:
        df[f + '_norm'] = normalize_series(df[f], method='minmax')

    df['regularite_norm'] = 1 - df['regularite_raw_norm']
    df['experience_norm'] = (df['nb_perf_norm'] + df['gains_log_norm']) / 2
    df.fillna(0, inplace=True)
    return df

# ------------------------------------------------------------------------------
# Score composite
# ------------------------------------------------------------------------------
def compute_composite_score(df, course_type):
    weights = WEIGHTS.get(course_type, WEIGHTS['plat'])
    score = 0
    for feature, w in weights.items():
        col = feature + '_norm' if feature in df.columns else None
        if col is None and feature == 'experience_norm':
            col = 'experience_norm'
        if col and col in df.columns:
            score += w * df[col]
    score += np.random.normal(0, 1e-6, len(score))
    return score

# ------------------------------------------------------------------------------
# Simulation Monte Carlo
# ------------------------------------------------------------------------------
def monte_carlo_simulation(scores, n_iter=1000, noise_scale=0.1):
    n = len(scores)
    prob_matrix = np.zeros((n_iter, n))
    for i in range(n_iter):
        noisy = scores + np.random.normal(0, noise_scale, n)
        prob_matrix[i, :] = softmax(noisy)
    mean_probs = np.mean(prob_matrix, axis=0)
    std_probs = np.std(prob_matrix, axis=0)
    return mean_probs, std_probs

# ------------------------------------------------------------------------------
# Probabilités implicites du marché
# ------------------------------------------------------------------------------
def market_probs(cotes):
    inv = 1.0 / np.array(cotes)
    return inv / inv.sum()

# ------------------------------------------------------------------------------
# Génération des combinaisons
# ------------------------------------------------------------------------------
def generate_combinations(probs, n_selection=5, comb_size=3, top_k=10):
    indices_sorted = np.argsort(probs)[::-1]
    top_indices = indices_sorted[:n_selection]
    combs = list(itertools.combinations(top_indices, comb_size))
    comb_scores = [sum(probs[list(c)]) for c in combs]
    sorted_combs = sorted(zip(combs, comb_scores), key=lambda x: x[1], reverse=True)
    return sorted_combs[:top_k]

# ------------------------------------------------------------------------------
# Génération du texte d'analyse
# ------------------------------------------------------------------------------
def generer_analyse_texte(df_sorted, outsiders, bases, volatilite, confiance):
    fav = df_sorted.iloc[0]
    deux = df_sorted.iloc[1]
    texte = f"**Favori :** Le {fav['numero']} avec {fav['proba_montecarlo']:.1%}. "
    texte += f"**Deuxième base :** {deux['numero']} ({deux['proba_montecarlo']:.1%}). "

    if len(outsiders) > 0:
        texte += "**Outsiders à suivre :** "
        for _, row in outsiders.iterrows():
            texte += f"{row['numero']} (value {row['value_pct']:.0f}%), "
        texte = texte[:-2] + ". "

    if volatilite < 0.5:
        texte += "Course plutôt sélective avec un favori marqué. "
    else:
        texte += "Course ouverte et indécise. "

    if confiance > 0.8:
        texte += "Notre modèle a une confiance élevée dans cette analyse."
    elif confiance > 0.5:
        texte += "Confiance modérée dans les probabilités."
    else:
        texte += "Prudence, forte incertitude."
    return texte

# ------------------------------------------------------------------------------
# Pipeline d'analyse complète
# ------------------------------------------------------------------------------
def analyse_course(df_partants, course_type, distance):
    df = compute_features(df_partants, course_type, distance)
    df['score'] = compute_composite_score(df, course_type)

    df['proba_modele'] = softmax(df['score'].values)

    mean_probs, std_probs = monte_carlo_simulation(df['score'].values)
    df['proba_montecarlo'] = mean_probs
    df['proba_std'] = std_probs

    market_probs_array = market_probs(df['cote'].values)
    df['proba_marche'] = market_probs_array

    df['value'] = df['proba_montecarlo'] - df['proba_marche']
    df['value_pct'] = (df['value'] / df['proba_marche']) * 100

    confiance = 1 - np.mean(std_probs)
    entropie = -np.sum(mean_probs * np.log(mean_probs + 1e-10)) / np.log(len(mean_probs))
    volatilite = entropie

    df_sorted = df.sort_values('proba_montecarlo', ascending=False).reset_index(drop=True)

    bases = df_sorted.head(2)[['numero', 'proba_montecarlo']].to_dict('records')

    seuil_value = 0.02
    outsiders = df[(df['value'] > seuil_value) & (df['proba_montecarlo'] < 0.15)]
    outsiders = outsiders.sort_values('value', ascending=False)
    outsiders_list = outsiders.head(3)[['numero', 'proba_montecarlo', 'value_pct']].to_dict('records')

    trio = generate_combinations(mean_probs, n_selection=5, comb_size=3, top_k=10)
    trio_result = [{'combinaison': '-'.join(map(str, [df.loc[i, 'numero'] for i in c])), 'score': s}
                   for c, s in trio]

    quint = generate_combinations(mean_probs, n_selection=7, comb_size=5, top_k=10)
    quint_result = [{'combinaison': '-'.join(map(str, [df.loc[i, 'numero'] for i in c])), 'score': s}
                    for c, s in quint]

    analyse_texte = generer_analyse_texte(df_sorted, outsiders, bases, volatilite, confiance)

    return {
        'df': df,
        'df_sorted': df_sorted,
        'bases': bases,
        'outsiders': outsiders_list,
        'trio': trio_result,
        'quinte': quint_result,
        'confiance': confiance,
        'volatilite': volatilite,
        'analyse_texte': analyse_texte
    }

# ------------------------------------------------------------------------------
# Extraction depuis URL (inchangée)
# ------------------------------------------------------------------------------
def extract_course_info_from_url(url):
    info = {'type': 'plat', 'distance': 0, 'nb_partants': 0}
    messages = []
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        response.encoding = 'utf-8'
        if response.status_code != 200:
            st.warning(f"⚠️ Code {response.status_code}")
            return info
        soup = BeautifulSoup(response.text, 'html.parser')
        page_text = soup.get_text(" ", strip=True)

        # Distance
        distance_match = re.search(r'(\d+)\s*m', page_text, re.IGNORECASE)
        if distance_match:
            info['distance'] = int(distance_match.group(1))
            messages.append(f"✅ Distance : {info['distance']} m")
        else:
            messages.append("❌ Distance non trouvée")

        # Type
        type_lower = page_text.lower()
        if re.search(r'haies|steeple|chase|obstacle', type_lower):
            info['type'] = 'obstacle'
            messages.append("✅ Type : obstacle")
        elif re.search(r'attelé|trott', type_lower):
            info['type'] = 'attelé'
            messages.append("✅ Type : attelé")
        elif re.search(r'monté', type_lower):
            info['type'] = 'monté'
            messages.append("✅ Type : monté")
        else:
            info['type'] = 'plat'
            messages.append("ℹ️ Type par défaut : plat")

        # Nombre de partants
        found = False
        partants_match = re.search(r'(\d+)\s*[pP]artants?', page_text)
        if partants_match:
            info['nb_partants'] = int(partants_match.group(1))
            messages.append(f"✅ Partants : {info['nb_partants']}")
            found = True
        if not found:
            for tag in soup.find_all(['h1','h2','h3','span','div','p','li']):
                if tag.string and 'partant' in tag.string.lower():
                    m = re.search(r'(\d+)\s*[pP]artants?', tag.string)
                    if m:
                        info['nb_partants'] = int(m.group(1))
                        messages.append(f"✅ Partants (balise {tag.name}) : {info['nb_partants']}")
                        found = True
                        break
        if not found and soup.title:
            m = re.search(r'(\d+)\s*[pP]artants?', soup.title.string)
            if m:
                info['nb_partants'] = int(m.group(1))
                messages.append(f"✅ Partants (titre) : {info['nb_partants']}")
                found = True
        if not found:
            meta_desc = soup.find('meta', attrs={'name':'description'})
            if meta_desc and meta_desc.get('content'):
                m = re.search(r'(\d+)\s*[pP]artants?', meta_desc['content'])
                if m:
                    info['nb_partants'] = int(m.group(1))
                    messages.append(f"✅ Partants (meta) : {info['nb_partants']}")
                    found = True
        if not found:
            messages.append("❌ Nombre de partants non trouvé")
    except Exception as e:
        st.error(f"Erreur extraction : {e}")
        return info
    for msg in messages:
        st.info(msg)
    return info

# ------------------------------------------------------------------------------
# Interface Streamlit avec éditeur de données
# ------------------------------------------------------------------------------
def main():
    st.set_page_config(page_title="Analyseur de Courses Hippiques", layout="wide")
    st.title("🐎 Analyseur Probabiliste de Courses (Modèle Quantitatif)")
    st.markdown("Saisissez les informations de la course et les partants pour obtenir une analyse avancée.")

    if 'partants' not in st.session_state:
        st.session_state.partants = []
    if 'course_info' not in st.session_state:
        st.session_state.course_info = {}
    if 'extraction_done' not in st.session_state:
        st.session_state.extraction_done = False
    if 'results' not in st.session_state:
        st.session_state.results = None

    # --------------------------------------------------------------------------
    # Option URL
    # --------------------------------------------------------------------------
    with st.expander("🔗 Option : Charger les informations depuis une URL Geny.com", expanded=False):
        url_input = st.text_input("Collez l'URL de la page des partants :")
        col1, col2 = st.columns([1,5])
        with col1:
            if st.button("Charger depuis l'URL"):
                if url_input:
                    with st.spinner("Extraction..."):
                        extracted = extract_course_info_from_url(url_input)
                        if extracted['distance']>0 or extracted['nb_partants']>0:
                            st.session_state.course_info['type'] = extracted['type']
                            st.session_state.course_info['distance'] = extracted['distance']
                            st.session_state.course_info['discipline'] = ""
                            st.session_state.course_info['niveau'] = ""
                            st.session_state.extraction_done = True
                            st.success("✅ Infos chargées")
                            st.rerun()
                        else:
                            st.warning("Aucune info trouvée.")
                else:
                    st.warning("Entrez une URL.")

    # --------------------------------------------------------------------------
    # Formulaire infos course
    # --------------------------------------------------------------------------
    with st.form("course_info_form"):
        st.subheader("Informations de la course")
        col1, col2 = st.columns(2)
        default_type = st.session_state.course_info.get('type', 'plat')
        default_distance = st.session_state.course_info.get('distance', 2000)
        default_discipline = st.session_state.course_info.get('discipline', '')
        default_niveau = st.session_state.course_info.get('niveau', '')

        with col1:
            type_course = st.selectbox(
                "Type de course",
                ["plat", "attelé", "monté", "obstacle"],
                index=["plat","attelé","monté","obstacle"].index(default_type) if default_type in ["plat","attelé","monté","obstacle"] else 0
            )
            distance = st.number_input("Distance (m)", min_value=0, value=int(default_distance))
        with col2:
            discipline = st.text_input("Discipline (optionnel)", default_discipline)
            niveau = st.text_input("Niveau (optionnel)", default_niveau)

        if st.form_submit_button("Enregistrer les infos"):
            st.session_state.course_info = {
                'type': type_course,
                'distance': distance,
                'discipline': discipline,
                'niveau': niveau
            }
            st.success("Infos course enregistrées")

    # --------------------------------------------------------------------------
    # Formulaire d'ajout d'un partant (inchangé)
    # --------------------------------------------------------------------------
    st.subheader("Ajout d'un partant")
    with st.expander("Nouveau partant", expanded=True):
        with st.form("partant_form"):
            col1, col2, col3 = st.columns(3)
            with col1:
                numero = st.number_input("Numéro", min_value=1, step=1)
                sexe = st.selectbox("Sexe", ["M", "F", "H"])
                age = st.number_input("Âge", min_value=2, max_value=20, value=5)
            with col2:
                cote = st.number_input("Cote", min_value=1.0, value=10.0, step=0.1)
                gains = st.number_input("Gains (€)", min_value=0.0, value=0.0)
                pct_driver = st.number_input("% victoire driver", min_value=0.0, max_value=100.0, value=0.0)
            with col3:
                pct_entraineur = st.number_input("% victoire entraineur", min_value=0.0, max_value=100.0, value=0.0)
                corde = st.number_input("Numéro corde (plat)", min_value=0, value=0)
                musique = st.text_input("Musique (ex: 1a 2a 3a)", "")
            if st.form_submit_button("Ajouter ce partant"):
                partant = {
                    'numero': numero,
                    'sexe': sexe,
                    'age': age,
                    'cote': cote,
                    'gains': gains,
                    'pct_driver': pct_driver,
                    'pct_entraineur': pct_entraineur,
                    'corde': corde,
                    'musique': musique
                }
                st.session_state.partants.append(partant)
                st.success(f"Partant {numero} ajouté")

    # --------------------------------------------------------------------------
    # Affichage et édition des partants
    # --------------------------------------------------------------------------
    st.subheader("Partants saisis")
    if st.session_state.partants:
        df_display = pd.DataFrame(st.session_state.partants)
        # S'assurer du bon ordre des colonnes
        colonnes = ['numero', 'sexe', 'age', 'cote', 'gains', 'pct_driver', 'pct_entraineur', 'corde', 'musique']
        df_display = df_display[colonnes]

        # Éditeur de données
        edited_df = st.data_editor(
            df_display,
            use_container_width=True,
            num_rows="dynamic",
            key="partants_editor",
            column_config={
                "numero": "Numéro",
                "sexe": "Sexe",
                "age": "Âge",
                "cote": "Cote",
                "gains": "Gains (€)",
                "pct_driver": "% Driver",
                "pct_entraineur": "% Entraîneur",
                "corde": "Corde",
                "musique": "Musique"
            }
        )

        # Bouton pour appliquer les modifications
        col1, col2, col3 = st.columns([1,1,2])
        with col1:
            if st.button("💾 Mettre à jour les données"):
                # Convertir l'édited_df en liste de dicts et mettre à jour session_state
                st.session_state.partants = edited_df.to_dict('records')
                st.success("Données mises à jour")
                # Si des résultats existent, on les efface pour forcer une nouvelle analyse
                st.session_state.results = None
                st.rerun()

        with col2:
            if st.button("🗑️ Réinitialiser la liste"):
                st.session_state.partants = []
                st.session_state.results = None
                st.rerun()

        with col3:
            # Si des résultats existent, on propose de réanalyser
            if st.session_state.results is not None:
                if st.button("🔄 Réanalyser avec les données modifiées", type="primary"):
                    # Relance l'analyse
                    if not st.session_state.course_info:
                        st.error("Infos course manquantes")
                    elif len(st.session_state.partants) < 2:
                        st.error("Pas assez de partants")
                    else:
                        with st.spinner("Analyse en cours..."):
                            progress_bar = st.progress(0)
                            for i in range(100):
                                progress_bar.progress(i+1)
                                np.random.rand()
                            df_partants = pd.DataFrame(st.session_state.partants)
                            results = analyse_course(
                                df_partants,
                                st.session_state.course_info['type'],
                                st.session_state.course_info['distance']
                            )
                            st.session_state.results = results
                            progress_bar.empty()
                            st.success("Analyse terminée !")
                            st.rerun()
    else:
        st.info("Aucun partant saisi.")

    # --------------------------------------------------------------------------
    # Bouton d'analyse initial (si aucun résultat)
    # --------------------------------------------------------------------------
    if st.session_state.results is None and st.button("Analyser la course", type="primary"):
        if not st.session_state.course_info:
            st.error("Veuillez d'abord enregistrer les informations de la course.")
        elif len(st.session_state.partants) < 2:
            st.error("Ajoutez au moins deux partants.")
        else:
            with st.spinner("Calcul en cours... (simulation Monte Carlo 1000 itérations)"):
                progress_bar = st.progress(0)
                for i in range(100):
                    progress_bar.progress(i + 1)
                    np.random.rand()
                df_partants = pd.DataFrame(st.session_state.partants)
                results = analyse_course(
                    df_partants,
                    st.session_state.course_info['type'],
                    st.session_state.course_info['distance']
                )
                st.session_state.results = results
                progress_bar.empty()
                st.success("Analyse terminée !")
                st.rerun()

    # --------------------------------------------------------------------------
    # Affichage des résultats
    # --------------------------------------------------------------------------
    if st.session_state.results is not None:
        res = st.session_state.results
        df_sorted = res['df_sorted']

        st.header("Résultats de l'analyse")

        # Tableau des probabilités
        st.subheader("📊 Probabilités de victoire")
        display_df = df_sorted[['numero', 'age', 'cote', 'proba_montecarlo', 'proba_marche', 'value_pct']].copy()
        display_df['proba_montecarlo'] = display_df['proba_montecarlo'].map('{:.1%}'.format)
        display_df['proba_marche'] = display_df['proba_marche'].map('{:.1%}'.format)
        display_df['value_pct'] = display_df['value_pct'].map('{:.1f}%'.format)
        display_df.columns = ['Numéro', 'Âge', 'Cote', 'Proba Modèle', 'Proba Marché', 'Value (%)']
        st.dataframe(display_df, use_container_width=True)

        # Graphique comparatif
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=df_sorted['numero'].astype(str),
            y=df_sorted['proba_montecarlo'],
            name='Modèle',
            marker_color='royalblue'
        ))
        fig.add_trace(go.Bar(
            x=df_sorted['numero'].astype(str),
            y=df_sorted['proba_marche'],
            name='Marché',
            marker_color='lightcoral'
        ))
        fig.update_layout(
            title="Comparaison Modèle vs Marché",
            xaxis_title="Numéro du cheval",
            yaxis_title="Probabilité",
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Bases et outsiders
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("🎯 Bases ultra solides")
            for base in res['bases']:
                st.write(f"**Cheval {base['numero']}** – probabilité {base['proba_montecarlo']:.1%}")
        with col2:
            st.subheader("💎 Outsiders à value")
            for out in res['outsiders']:
                st.write(f"**Cheval {out['numero']}** – proba {out['proba_montecarlo']:.1%} (value {out['value_pct']:.0f}%)")

        # Combinaisons
        st.subheader("🔢 Top 10 combinaisons Trio (ordre indifférent)")
        for i, comb in enumerate(res['trio'], 1):
            st.write(f"{i}. {comb['combinaison']} (score {comb['score']:.3f})")

        st.subheader("🔢 Top 10 combinaisons Quinté (ordre indifférent)")
        for i, comb in enumerate(res['quinte'], 1):
            st.write(f"{i}. {comb['combinaison']} (score {comb['score']:.3f})")

        # Indices
        st.subheader("📈 Indices de confiance et volatilité")
        st.write(f"**Indice de confiance global :** {res['confiance']:.2f}")
        st.write(f"**Indice de volatilité :** {res['volatilite']:.2f}")

        # Analyse texte
        st.subheader("📝 Analyse automatique")
        st.markdown(res['analyse_texte'])

if __name__ == "__main__":
    main()
