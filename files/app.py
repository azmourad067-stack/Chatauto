"""Application Streamlit — Outil de pronostic hippique (galop plat).

Deux scores calculés par course, à partir de 6 paramètres saisis par cheval :
- Score intrinsèque : estime la valeur du cheval indépendamment de la cote
  (forme, aptitude distance, aptitude terrain, poids porté, jockey/entraîneur).
- Score de valeur : compare le score intrinsèque à la cote du marché pour
  repérer les chevaux potentiellement sous-cotés ou surcotés.
"""

import streamlit as st
import pandas as pd
from datetime import date as date_cls

from pmu_connector import (
    recuperer_reunions_du_jour,
    recuperer_courses_reunion,
    recuperer_partants,
    partants_vers_lignes,
    ErreurConnecteurPMU,
)
from models import (
    COL_NOM,
    COL_FORME,
    COL_POIDS,
    COL_DISTANCE,
    COL_TERRAIN,
    COL_JOCKEY,
    COL_COTE,
    OPTIONS_APTITUDE,
    OPTIONS_JOCKEY,
    dataframe_vide,
)
from scoring import calculer_score_intrinseque, calculer_score_valeur, POIDS_DEFAUT
from data_logger import preparer_export, dataframe_vers_csv_bytes

st.set_page_config(page_title="Pronostic Hippique", page_icon="🐎", layout="wide")

st.title("🐎 Outil de pronostic hippique — Galop plat")
st.caption(
    "Score intrinsèque (indépendant de la cote) + score de valeur "
    "(comparaison à la cote du marché)."
)

# --- État partagé pour la table des chevaux (édité par la saisie manuelle
# ET par l'auto-remplissage) ---
if "df_chevaux" not in st.session_state:
    st.session_state.df_chevaux = dataframe_vide(8)
if "editor_version" not in st.session_state:
    st.session_state.editor_version = 0
if "nb_chevaux_input" not in st.session_state:
    st.session_state.nb_chevaux_input = 8


def _remplacer_df_chevaux(nouveau_df: pd.DataFrame) -> None:
    """Remplace le tableau des chevaux et force le rafraîchissement du widget."""
    st.session_state.df_chevaux = nouveau_df
    st.session_state.editor_version += 1


# --- 1. Auto-remplissage optionnel via l'API non officielle de PMU.fr ---
with st.expander("1. Auto-remplissage depuis PMU.fr (optionnel, source non officielle)"):
    st.warning(
        "Cette fonctionnalité s'appuie sur une API interne de PMU.fr, "
        "non documentée officiellement. Elle peut être instable ou "
        "indisponible à tout moment. En cas d'échec, complète simplement "
        "le tableau à la main plus bas — rien n'est bloquant."
    )

    date_course = st.date_input("Date de la course", value=date_cls.today())

    if st.button("1️⃣ Charger les réunions du jour"):
        try:
            st.session_state.reunions_dispo = recuperer_reunions_du_jour(date_course)
            st.session_state.pop("courses_dispo", None)
            st.success(f"{len(st.session_state.reunions_dispo)} réunion(s) trouvée(s).")
        except ErreurConnecteurPMU as erreur:
            st.error(f"Échec du chargement des réunions : {erreur}")
            st.session_state.pop("reunions_dispo", None)

    if "reunions_dispo" in st.session_state and st.session_state.reunions_dispo:
        options_reunions = {}
        for r in st.session_state.reunions_dispo:
            libelle_hippodrome = (r.get("hippodrome") or {}).get("libelleCourt") or r.get("libelle", "?")
            options_reunions[f"R{r.get('numOrdre', '?')} — {libelle_hippodrome}"] = r.get("numOrdre")

        choix_reunion = st.selectbox("Réunion", list(options_reunions.keys()), key="choix_reunion")

        if st.button("2️⃣ Charger les courses de cette réunion"):
            try:
                num_reunion = options_reunions[choix_reunion]
                st.session_state.courses_dispo = recuperer_courses_reunion(date_course, num_reunion)
                st.session_state.num_reunion_choisie = num_reunion
                st.success(f"{len(st.session_state.courses_dispo)} course(s) trouvée(s).")
            except ErreurConnecteurPMU as erreur:
                st.error(f"Échec du chargement des courses : {erreur}")
                st.session_state.pop("courses_dispo", None)

    if "courses_dispo" in st.session_state and st.session_state.courses_dispo:
        options_courses = {}
        for c in st.session_state.courses_dispo:
            libelle_course = c.get("libelle") or c.get("libelleCourt") or "?"
            options_courses[f"C{c.get('numOrdre', '?')} — {libelle_course} ({c.get('distance', '?')} m)"] = c.get("numOrdre")

        choix_course = st.selectbox("Course", list(options_courses.keys()), key="choix_course")

        if st.button("3️⃣ Récupérer les partants et pré-remplir le tableau", type="primary"):
            try:
                num_course = options_courses[choix_course]
                partants_bruts = recuperer_partants(
                    date_course, st.session_state.num_reunion_choisie, num_course
                )
                lignes = partants_vers_lignes(partants_bruts)
                if not lignes:
                    raise ErreurConnecteurPMU("Aucun partant exploitable dans la réponse.")

                nouveau_df = pd.DataFrame(
                    {
                        COL_NOM: [l["nom"] for l in lignes],
                        COL_FORME: [
                            l["forme_estimee"] if l["forme_estimee"] is not None else 2
                            for l in lignes
                        ],
                        COL_POIDS: [l["poids"] if l["poids"] else 58.0 for l in lignes],
                        COL_DISTANCE: ["Neutre"] * len(lignes),
                        COL_TERRAIN: ["Neutre"] * len(lignes),
                        COL_JOCKEY: ["Moyen"] * len(lignes),
                        COL_COTE: [l["cote"] if l["cote"] else 10.0 for l in lignes],
                    }
                )
                _remplacer_df_chevaux(nouveau_df)
                st.session_state.nb_chevaux_input = len(lignes)
                st.success(
                    f"{len(lignes)} partant(s) chargé(s) dans le tableau ci-dessous. "
                    "Vérifie/ajuste l'aptitude distance, l'aptitude terrain et le "
                    "niveau jockey/entraîneur : l'API ne fournit pas ces évaluations "
                    "qualitatives, et la forme estimée à partir de la musique est "
                    "une approximation à confirmer."
                )
                noms_jockeys = [f"{l['nom']} → {l['jockey']}" for l in lignes if l.get("jockey")]
                if noms_jockeys:
                    st.caption("Jockeys/drivers détectés : " + " · ".join(noms_jockeys))
            except ErreurConnecteurPMU as erreur:
                st.error(
                    f"Échec de la récupération des partants : {erreur}. "
                    "Complète le tableau manuellement ci-dessous."
                )

# --- 2. Informations générales de la course ---
st.subheader("2. Informations générales de la course")
col1, col2, col3 = st.columns(3)
with col1:
    nb_chevaux = st.number_input(
        "Nombre de partants", min_value=2, max_value=25, key="nb_chevaux_input"
    )
with col2:
    distance = st.number_input(
        "Distance de la course (m)", min_value=800, max_value=5000, value=2000, step=100
    )
with col3:
    terrain_jour = st.selectbox("Terrain du jour", ["Bon", "Souple", "Lourd", "Léger"])

# --- Pondération ajustable (barre latérale) ---
with st.sidebar:
    st.header("Pondération du score intrinsèque")
    st.caption("Les 5 poids doivent idéalement sommer à 100%.")
    poids_forme = st.slider("Forme récente", 0, 100, int(POIDS_DEFAUT["forme"] * 100))
    poids_distance = st.slider("Aptitude distance", 0, 100, int(POIDS_DEFAUT["distance"] * 100))
    poids_terrain = st.slider("Aptitude terrain", 0, 100, int(POIDS_DEFAUT["terrain"] * 100))
    poids_poids = st.slider("Poids porté", 0, 100, int(POIDS_DEFAUT["poids"] * 100))
    poids_jockey = st.slider("Jockey / Entraîneur", 0, 100, int(POIDS_DEFAUT["jockey"] * 100))

    total_poids = poids_forme + poids_distance + poids_terrain + poids_poids + poids_jockey
    if total_poids != 100:
        st.warning(f"Somme actuelle des poids : {total_poids}% (idéalement 100%)")

    poids_utilisateur = {
        "forme": poids_forme / 100,
        "distance": poids_distance / 100,
        "terrain": poids_terrain / 100,
        "poids": poids_poids / 100,
        "jockey": poids_jockey / 100,
    }

    st.divider()
    st.caption(
        "Cet outil produit une estimation basée sur les paramètres saisis, "
        "pas une garantie de résultat. À utiliser comme aide à la réflexion."
    )

# --- 3. Saisie des chevaux ---
st.subheader("3. Données des partants")
st.caption(
    "Renseigne les 6 paramètres pour chaque cheval du peloton, ou utilise "
    "l'auto-remplissage ci-dessus puis ajuste les valeurs qualitatives."
)

if len(st.session_state.df_chevaux) != nb_chevaux:
    _remplacer_df_chevaux(dataframe_vide(nb_chevaux))

df_edite = st.data_editor(
    st.session_state.df_chevaux,
    num_rows="fixed",
    use_container_width=True,
    column_config={
        COL_FORME: st.column_config.NumberColumn(
            "Forme (podiums /5)", min_value=0, max_value=5, step=1
        ),
        COL_POIDS: st.column_config.NumberColumn(
            "Poids (kg)", min_value=45.0, max_value=70.0, step=0.5
        ),
        COL_DISTANCE: st.column_config.SelectboxColumn(
            "Aptitude distance", options=OPTIONS_APTITUDE
        ),
        COL_TERRAIN: st.column_config.SelectboxColumn(
            "Aptitude terrain", options=OPTIONS_APTITUDE
        ),
        COL_JOCKEY: st.column_config.SelectboxColumn(
            "Jockey/Entraîneur", options=OPTIONS_JOCKEY
        ),
        COL_COTE: st.column_config.NumberColumn(
            "Cote probable", min_value=1.01, step=0.1
        ),
    },
    key=f"editeur_chevaux_v{st.session_state.editor_version}",
)

# --- 4. Calcul et affichage des résultats ---
if st.button("Calculer les pronostics", type="primary"):
    resultats = calculer_score_intrinseque(df_edite, poids_utilisateur)
    resultats = calculer_score_valeur(resultats)

    st.subheader("4. Résultats")

    tab_classement, tab_valeur, tab_graph = st.tabs(
        ["Classement (score intrinsèque)", "Score de valeur", "Graphique comparatif"]
    )

    with tab_classement:
        classement = resultats.sort_values("score_intrinseque", ascending=False)
        st.dataframe(
            classement[
                [
                    COL_NOM,
                    "score_intrinseque",
                    "note_forme",
                    "note_distance",
                    "note_terrain",
                    "note_poids",
                    "note_jockey",
                ]
            ].round(2),
            use_container_width=True,
            hide_index=True,
        )

    with tab_valeur:
        valeur = resultats.sort_values("ecart_valeur", ascending=False)
        affichage_valeur = valeur[
            [COL_NOM, COL_COTE, "proba_modele", "proba_marche", "ecart_valeur"]
        ].copy()
        affichage_valeur["proba_modele"] = (
            (affichage_valeur["proba_modele"] * 100).round(1).astype(str) + " %"
        )
        affichage_valeur["proba_marche"] = (
            (affichage_valeur["proba_marche"] * 100).round(1).astype(str) + " %"
        )
        affichage_valeur["ecart_valeur"] = (valeur["ecart_valeur"] * 100).round(1)
        st.dataframe(affichage_valeur, use_container_width=True, hide_index=True)
        st.caption(
            "Écart positif = cheval potentiellement sous-coté par le marché "
            "selon le modèle. Écart négatif = potentiellement surcoté."
        )

    with tab_graph:
        graph_data = resultats.set_index(COL_NOM)[["proba_modele", "proba_marche"]]
        graph_data.columns = ["Modèle", "Marché (cote)"]
        st.bar_chart(graph_data)

    st.subheader("5. Export")
    export_df = preparer_export(resultats, {"distance": distance, "terrain": terrain_jour})
    csv_bytes = dataframe_vers_csv_bytes(export_df)
    st.download_button(
        "Télécharger les résultats en CSV",
        data=csv_bytes,
        file_name=f"pronostic_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )
    st.caption(
        "Astuce : complète la colonne « Résultat réel » après la course. "
        "En accumulant ces fichiers au fil du temps, tu te constitues une "
        "base historique utile pour calibrer les poids plus tard."
    )
