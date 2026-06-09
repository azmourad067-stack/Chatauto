"""
app.py - Interface Streamlit pour l'Application de Pronostics Hippiques
Application de Pronostics Hippiques - Analyse par IA
"""

import streamlit as st
import pandas as pd
import numpy as np
import io
import time


st.set_page_config(
    page_title="Pronostics Hippiques IA",
    page_icon="🏇",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
#  CSS PERSONNALISÉ
# ──────────────────────────────────────────────

st.markdown("""
<style>
    /* Police et fond général */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Titre principal */
    .main-title {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        color: white;
        padding: 2rem 2.5rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 32px rgba(15, 52, 96, 0.4);
    }
    .main-title h1 { font-size: 2.4rem; font-weight: 700; margin: 0; }
    .main-title p  { font-size: 1.05rem; opacity: 0.85; margin-top: 0.5rem; }

    /* Carte métrique */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102,126,234,0.35);
    }
    .metric-card h3 { margin: 0; font-size: 2rem; font-weight: 700; }
    .metric-card p  { margin: 0.3rem 0 0; font-size: 0.85rem; opacity: 0.9; }

    /* Favori */
    .favori-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(240,93,251,0.35);
    }
    .favori-card h3 { margin: 0; font-size: 1.6rem; font-weight: 700; }
    .favori-card p  { margin: 0.3rem 0 0; font-size: 0.85rem; opacity: 0.9; }

    /* Tableau de résultats */
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* Séparateur */
    .sep { border: none; border-top: 2px solid #e2e8f0; margin: 2rem 0; }

    /* Badge rang */
    .rang-1 { color: #FFD700; font-weight: 700; font-size: 1.2rem; }
    .rang-2 { color: #C0C0C0; font-weight: 700; }
    .rang-3 { color: #CD7F32; font-weight: 700; }

    /* Info box */
    .info-box {
        background: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .warn-box {
        background: #FFFBEB;
        border-left: 4px solid #F59E0B;
        padding: 1rem 1.2rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
#  SIDEBAR — OPTIONS GLOBALES
# ──────────────────────────────────────────────

with st.sidebar:
    st.image(
        "https://cdn-icons-png.flaticon.com/512/2936/2936886.png",
        width=80,
    )
    st.markdown("## ⚙️ Paramètres")

    type_course_global = st.selectbox(
        "🏁 Type de course",
        options=["Plat", "Trot Attelé"],
        index=0,
        help="Sélectionnez la discipline principale. "
             "Les musiques de discipline différente seront ignorées dans le calcul.",
    )

    st.markdown("---")
    st.markdown("### ⚖️ Pondérations")

    poids_cheval = st.slider(
        "Poids Cheval (%)",
        min_value=10, max_value=90,
        value=int(POIDS_CHEVAL * 100),
        step=5,
        help="Importance de la forme propre du cheval.",
    ) / 100

    poids_jockey = st.slider(
        "Poids Jockey (%)",
        min_value=5, max_value=60,
        value=int(POIDS_JOCKEY * 100),
        step=5,
        help="Importance des performances du jockey.",
    ) / 100

    poids_entraineur_calc = max(0.05, 1.0 - poids_cheval - poids_jockey)
    couleur_info = "🟢" if poids_entraineur_calc >= 0.10 else "🟡"
    st.info(
        f"{couleur_info} Poids Entraîneur calculé : **{poids_entraineur_calc*100:.0f}%**"
    )
    if poids_cheval + poids_jockey >= 0.95:
        st.warning("⚠️ La somme Cheval + Jockey est très élevée. "
                   "Le poids Entraîneur est limité à 5% minimum.")

    st.markdown("---")
    st.markdown("### 📖 À propos")
    st.markdown(
        """
        **Application de Pronostics Hippiques v1.0**

        Algorithme basé sur :
        - 📊 Analyse pondérée de la musique
        - 📉 Décroissance temporelle
        - 🔄 Multi-discipline (Plat / Attelé)

        ⚠️ *Usage informatif uniquement.*
        """
    )

# ──────────────────────────────────────────────
#  EN-TÊTE PRINCIPAL
# ──────────────────────────────────────────────

st.markdown("""
<div class="main-title">
  <h1>🏇 Application de Pronostics Hippiques</h1>
  <p>Analysez les musiques des chevaux, jockeys et entraîneurs pour établir vos pronostics</p>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  INSTRUCTIONS
# ──────────────────────────────────────────────

with st.expander("📋 Comment utiliser cette application ?", expanded=False):
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        ### 🚀 Étapes
        1. **Choisissez le type de course** dans la barre latérale (Plat ou Trot Attelé).
        2. **Saisissez les données** dans le tableau ci-dessous (ou collez depuis Excel).
        3. **Importez un fichier Excel** via le bouton dédié si vous avez un fichier prêt.
        4. **Cliquez sur "Analyser"** pour obtenir les pronostics.
        5. **Exportez les résultats** au format CSV si nécessaire.
        """)
    with col_b:
        st.markdown("""
        ### 🎵 Format de la Musique
        | Symbole | Signification |
        |---------|--------------|
        | `1p`-`9p` | Place en course de Plat |
        | `1a`-`9a` | Place en Trot Attelé |
        | `(25)` | Année de la course |
        | `D` | Disqualification |
        | `T` | Chute |
        | `A` | Arrêt / Abandon |
        | `R` | Retiré |

        *Exemple : `3p1p2p1a` = 3ème Plat, 1er Plat, 2ème Plat, 1er Attelé*
        """)

st.markdown('<hr class="sep">', unsafe_allow_html=True)

# ──────────────────────────────────────────────
#  TABS PRINCIPAUX
# ──────────────────────────────────────────────

tab_saisie, tab_resultats, tab_detail = st.tabs([
    "📝 Saisie des données",
    "🏆 Pronostics",
    "🔍 Analyse détaillée",
])

# ════════════════════════════════════════════
#  TAB 1 — SAISIE DES DONNÉES
# ════════════════════════════════════════════

with tab_saisie:
    st.markdown("### 🐴 Tableau des partants")

    # ── Chargement depuis Excel ──
    col_import1, col_import2 = st.columns([2, 1])
    with col_import1:
        fichier_excel = st.file_uploader(
            "📂 Importer un fichier Excel (.xlsx / .xls)",
            type=["xlsx", "xls"],
            help="Le fichier doit contenir les colonnes : N°, Nom, Musique, "
                 "Musique Jockey, Musique Entraîneur",
        )
    with col_import2:
        st.markdown("<br>", unsafe_allow_html=True)
        charger_exemple = st.button("📋 Charger les données d'exemple", use_container_width=True)

    # ── Initialisation du DataFrame en session ──
    if "df_course" not in st.session_state:
        st.session_state["df_course"] = EXEMPLE_DONNEES.copy()

    if charger_exemple:
        st.session_state["df_course"] = EXEMPLE_DONNEES.copy()
        st.success("✅ Données d'exemple chargées !")

    if fichier_excel is not None:
        try:
            df_import = pd.read_excel(fichier_excel)
            # Nettoyage des colonnes (trim espaces)
            df_import.columns = [str(c).strip() for c in df_import.columns]

            # Vérification minimale
            erreurs_import = valider_dataframe(df_import)
            if erreurs_import:
                st.error("❌ Erreur dans le fichier importé :\n" + "\n".join(erreurs_import))
            else:
                # Compléter les colonnes manquantes optionnelles
                if "Nom" not in df_import.columns:
                    df_import["Nom"] = [f"Cheval {n}" for n in df_import["N°"]]
                if "Type Course" not in df_import.columns:
                    df_import["Type Course"] = type_course_global

                st.session_state["df_course"] = df_import
                st.success(f"✅ {len(df_import)} chevaux importés depuis le fichier Excel !")
        except Exception as e:
            st.error(f"❌ Impossible de lire le fichier : {e}")

    # ── Éditeur de données interactif ──
    st.markdown(
        '<div class="info-box">💡 Vous pouvez modifier directement les cellules, '
        'ajouter ou supprimer des lignes.</div>',
        unsafe_allow_html=True,
    )

    df_edite = st.data_editor(
        st.session_state["df_course"],
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "N°": st.column_config.NumberColumn(
                "N°", help="Numéro du cheval", min_value=1, step=1, required=True
            ),
            "Nom": st.column_config.TextColumn(
                "Nom", help="Nom du cheval (optionnel)"
            ),
            "Musique": st.column_config.TextColumn(
                "Musique 🐴",
                help="Musique du cheval (ex: 3p1p2p1p3p)",
                width="medium",
            ),
            "Musique Jockey": st.column_config.TextColumn(
                "Musique Jockey 🏇",
                help="Musique du jockey (ex: 2p1p3p2p)",
                width="medium",
            ),
            "Musique Entraîneur": st.column_config.TextColumn(
                "Musique Entraîneur 🎯",
                help="Musique de l'entraîneur (ex: 1p2p3p1p)",
                width="medium",
            ),
            "Type Course": st.column_config.SelectboxColumn(
                "Type Course",
                options=["Plat", "Attelé", "Haies", "Steeple"],
                help="Type de course pour ce cheval (remplace le paramètre global si renseigné)",
            ),
        },
        key="data_editor_main",
        height=350,
    )

    # Sauvegarde en session
    st.session_state["df_course"] = df_edite

    # ── Bouton principal d'analyse ──
    st.markdown("<br>", unsafe_allow_html=True)
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        lancer_analyse = st.button(
            "🔍 Lancer l'analyse et générer les pronostics",
            type="primary",
            use_container_width=True,
        )

    if lancer_analyse:
        df_a_analyser = st.session_state["df_course"].copy()
        # Nettoyage : supprimer les lignes entièrement vides
        df_a_analyser.dropna(how="all", inplace=True)
        df_a_analyser.reset_index(drop=True, inplace=True)

        erreurs_valid = valider_dataframe(df_a_analyser)
        if erreurs_valid:
            st.error("❌ " + "\n".join(erreurs_valid))
        else:
            with st.spinner("⏳ Analyse en cours..."):
                time.sleep(0.6)  # UX : légère pause pour l'animation
                resultats, erreurs_analyse = analyser_course(
                    df_a_analyser,
                    type_course_global,
                    poids_cheval=poids_cheval,
                    poids_jockey=poids_jockey,
                    poids_entraineur=poids_entraineur_calc,
                )

            # Stockage en session
            st.session_state["resultats"]     = resultats
            st.session_state["type_course"]   = type_course_global
            st.session_state["df_analysee"]   = df_a_analyser
            st.session_state["analyse_faite"] = True

            if erreurs_analyse:
                st.warning("⚠️ Avertissements :\n" + "\n".join(erreurs_analyse))

            st.success(
                f"✅ Analyse terminée ! **{len(resultats)}** chevaux analysés pour "
                f"une course de **{type_course_global}**. "
                "👉 Consultez l'onglet **Pronostics** pour les résultats."
            )
            # Forcer le passage sur l'onglet résultats via un indicateur
            st.balloons()


# ════════════════════════════════════════════
#  TAB 2 — PRONOSTICS
# ════════════════════════════════════════════

with tab_resultats:
    if not st.session_state.get("analyse_faite"):
        st.markdown(
            '<div class="warn-box">⚠️ Aucune analyse effectuée. '
            "Veuillez d'abord saisir vos données dans l'onglet <strong>Saisie des données</strong> "
            "et cliquer sur <strong>Lancer l'analyse</strong>.</div>",
            unsafe_allow_html=True,
        )
    else:
        resultats = st.session_state["resultats"]
        tc        = st.session_state["type_course"]
        stats     = statistiques_course(resultats)

        # ── En-tête des stats ──
        st.markdown(f"### 🏁 Résultats — Course de {tc}")

        col_s1, col_s2, col_s3, col_s4 = st.columns(4)
        with col_s1:
            st.markdown(
                f'<div class="metric-card"><h3>{stats.get("Nombre de partants",0)}</h3>'
                f'<p>Partants</p></div>', unsafe_allow_html=True
            )
        with col_s2:
            st.markdown(
                f'<div class="metric-card"><h3>{stats.get("Score moyen",0):.1f}</h3>'
                f'<p>Score moyen</p></div>', unsafe_allow_html=True
            )
        with col_s3:
            st.markdown(
                f'<div class="favori-card"><h3>N° {stats.get("Favori N°","?")}</h3>'
                f'<p>Favori — {stats.get("Favori Nom","")}</p></div>', unsafe_allow_html=True
            )
        with col_s4:
            st.markdown(
                f'<div class="metric-card"><h3>{stats.get("Score max",0):.1f}</h3>'
                f'<p>Meilleur score</p></div>', unsafe_allow_html=True
            )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── TOP 3 en cards ──
        st.markdown("### 🥇 Top 3 du pronostic")
        top3 = resultats[:3]
        medailles = ["🥇", "🥈", "🥉"]
        cols_top = st.columns(len(top3))
        for i, (col, res) in enumerate(zip(cols_top, top3)):
            with col:
                couleur = ["#FFD700", "#C0C0C0", "#CD7F32"][i]
                st.markdown(
                    f"""
                    <div style="background:linear-gradient(135deg,{couleur}33,{couleur}11);
                                border:2px solid {couleur};border-radius:14px;
                                padding:1.2rem;text-align:center;">
                        <div style="font-size:2.5rem">{medailles[i]}</div>
                        <div style="font-size:1.3rem;font-weight:700">N° {res.numero}</div>
                        <div style="font-size:1rem;color:#444">{res.nom}</div>
                        <div style="font-size:1.8rem;font-weight:700;color:#1a1a2e">
                            {res.score_global:.1f}<span style="font-size:0.9rem">/100</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Tableau complet ──
        st.markdown("### 📊 Classement complet")
        df_resultats = resultats_vers_dataframe(resultats)

        # Affichage sans la colonne commentaire dans le tableau principal
        df_affichage = df_resultats.drop(columns=["Commentaire"])
        st.dataframe(
            df_affichage,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Rang 🏆":          st.column_config.NumberColumn("Rang 🏆", format="%d"),
                "Score Global":     st.column_config.ProgressColumn(
                    "Score Global", min_value=0, max_value=100, format="%.1f"
                ),
                "Score Cheval":     st.column_config.ProgressColumn(
                    "Score Cheval 🐴", min_value=0, max_value=100, format="%.1f"
                ),
                "Score Jockey":     st.column_config.ProgressColumn(
                    "Score Jockey 🏇", min_value=0, max_value=100, format="%.1f"
                ),
                "Score Entraîneur": st.column_config.ProgressColumn(
                    "Score Entraîneur 🎯", min_value=0, max_value=100, format="%.1f"
                ),
            },
        )

        # ── Graphique en barres ──
        st.markdown("### 📈 Comparatif des scores")
        try:
            import plotly.graph_objects as go

            noms_chevaux = [f"N°{r.numero} {r.nom}" for r in resultats]
            fig = go.Figure()
            fig.add_trace(go.Bar(
                name="Score Cheval",
                x=noms_chevaux,
                y=[r.score_cheval for r in resultats],
                marker_color="#667eea",
            ))
            fig.add_trace(go.Bar(
                name="Score Jockey",
                x=noms_chevaux,
                y=[r.score_jockey for r in resultats],
                marker_color="#f093fb",
            ))
            fig.add_trace(go.Bar(
                name="Score Entraîneur",
                x=noms_chevaux,
                y=[r.score_entraineur for r in resultats],
                marker_color="#4ade80",
            ))
            fig.add_trace(go.Scatter(
                name="Score Global",
                x=noms_chevaux,
                y=[r.score_global for r in resultats],
                mode="lines+markers",
                line=dict(color="#f5576c", width=3),
                marker=dict(size=10),
            ))
            fig.update_layout(
                barmode="group",
                plot_bgcolor="white",
                paper_bgcolor="white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis=dict(range=[0, 105], title="Score (/100)"),
                xaxis=dict(title=""),
                height=420,
                margin=dict(l=40, r=40, t=40, b=80),
            )
            st.plotly_chart(fig, use_container_width=True)

        except ImportError:
            # Fallback sur st.bar_chart si Plotly n'est pas disponible
            df_chart = pd.DataFrame({
                "Score Global":     [r.score_global for r in resultats],
                "Score Cheval":     [r.score_cheval for r in resultats],
                "Score Jockey":     [r.score_jockey for r in resultats],
                "Score Entraîneur": [r.score_entraineur for r in resultats],
            }, index=[f"N°{r.numero}" for r in resultats])
            st.bar_chart(df_chart)

        # ── Export CSV ──
        st.markdown("<br>", unsafe_allow_html=True)
        csv_data = df_resultats.to_csv(index=False, encoding="utf-8-sig")
        st.download_button(
            label="⬇️ Exporter les résultats (CSV)",
            data=csv_data,
            file_name=f"pronostics_{tc.replace(' ', '_').lower()}.csv",
            mime="text/csv",
            use_container_width=False,
        )


# ════════════════════════════════════════════
#  TAB 3 — ANALYSE DÉTAILLÉE
# ════════════════════════════════════════════

with tab_detail:
    if not st.session_state.get("analyse_faite"):
        st.markdown(
            '<div class="warn-box">⚠️ Aucune analyse effectuée. '
            "Veuillez d'abord lancer l'analyse depuis l'onglet <strong>Saisie des données</strong>.</div>",
            unsafe_allow_html=True,
        )
    else:
        resultats = st.session_state["resultats"]

        st.markdown("### 🔍 Fiche détaillée par cheval")

        # Sélecteur de cheval
        options_cheval = {
            f"N°{r.numero} — {r.nom} (Score: {r.score_global:.1f})": r
            for r in resultats
        }
        cheval_selec = st.selectbox(
            "Sélectionnez un cheval",
            options=list(options_cheval.keys()),
        )
        res_selec = options_cheval[cheval_selec]

        # ── Fiche détaillée ──
        col_d1, col_d2 = st.columns([1, 2])

        with col_d1:
            # Jauge de score
            st.markdown("#### Score Global")
            couleur_score = (
                "#22c55e" if res_selec.score_global >= 80
                else "#f59e0b" if res_selec.score_global >= 50
                else "#ef4444"
            )
            st.markdown(
                f"""
                <div style="text-align:center; padding:1.5rem; background:#f8fafc;
                            border-radius:16px; border:2px solid {couleur_score};">
                    <div style="font-size:4rem; font-weight:800; color:{couleur_score}">
                        {res_selec.score_global:.1f}
                    </div>
                    <div style="color:#64748b; font-size:1rem">/100</div>
                    <div style="font-size:1.5rem; margin-top:0.5rem">
                        {'🟢' if res_selec.score_global >= 80
                         else '🟡' if res_selec.score_global >= 50
                         else '🔴'}
                        Rang #{res_selec.rang_pronostic}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("<br>", unsafe_allow_html=True)
            # Détail des scores composants
            st.markdown("#### Décomposition")
            df_decompo = pd.DataFrame({
                "Composante": ["🐴 Cheval (60%)", "🏇 Jockey (25%)", "🎯 Entraîneur (15%)"],
                "Score":      [res_selec.score_cheval, res_selec.score_jockey, res_selec.score_entraineur],
                "Courses":    [res_selec.nb_courses_cheval, res_selec.nb_courses_jockey, res_selec.nb_courses_entraineur],
            })
            st.dataframe(df_decompo, hide_index=True, use_container_width=True)

        with col_d2:
            st.markdown("#### 📝 Analyse IA")
            st.markdown(res_selec.commentaire, unsafe_allow_html=False)

            st.markdown("---")

            # Données brutes
            st.markdown("#### 🎵 Musiques brutes analysées")
            if st.session_state.get("df_analysee") is not None:
                df_src = st.session_state["df_analysee"]
                ligne_cheval = df_src[df_src["N°"].astype(str) == str(res_selec.numero)]
                if not ligne_cheval.empty:
                    row = ligne_cheval.iloc[0]
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.markdown("**🐴 Cheval**")
                        st.code(str(row.get("Musique", "N/A")))
                    with col_m2:
                        st.markdown("**🏇 Jockey**")
                        st.code(str(row.get("Musique Jockey", "N/A")))
                    with col_m3:
                        st.markdown("**🎯 Entraîneur**")
                        st.code(str(row.get("Musique Entraîneur", "N/A")))

        # ── Outil de test de musique ──
        st.markdown("---")
        st.markdown("### 🧪 Outil de test de musique")
        st.markdown(
            '<div class="info-box">Testez ici le calcul du score pour n\'importe quelle '
            "chaîne de musique.</div>",
            unsafe_allow_html=True,
        )

        from utils import calculer_score_musique, _extraire_performances

        col_t1, col_t2, col_t3 = st.columns([2, 1, 1])
        with col_t1:
            test_musique = st.text_input(
                "Saisissez une musique à tester",
                value="3p1p2p1p3p",
                placeholder="ex: 3p1p2p1a3p",
            )
        with col_t2:
            test_type = st.selectbox("Type de course", ["Plat", "Trot Attelé"], key="test_type")
        with col_t3:
            st.markdown("<br>", unsafe_allow_html=True)
            calculer = st.button("▶️ Calculer", use_container_width=True)

        if calculer and test_musique:
            score_test, nb_test = calculer_score_musique(test_musique, test_type)
            perf_test = _extraire_performances(test_musique.upper(), test_type)

            st.success(f"✅ Score calculé : **{score_test:.2f}/100** sur **{nb_test}** course(s) pertinente(s)")

            # Tableau des performances parsées
            if perf_test:
                df_perf = pd.DataFrame([
                    {
                        "Course N°": i + 1,
                        "Place": p[0],
                        "Pertinente": "✅" if p[1] else "❌ (discipline différente)",
                    }
                    for i, p in enumerate(perf_test)
                ])
                st.dataframe(df_perf, hide_index=True, use_container_width=True)

# ──────────────────────────────────────────────
#  PIED DE PAGE
# ──────────────────────────────────────────────

st.markdown(
    """
    <hr style="margin-top:3rem;border:none;border-top:1px solid #e2e8f0;">
    <p style="text-align:center;color:#94a3b8;font-size:0.85rem;">
        🏇 Application de Pronostics Hippiques v1.0 —
        <em>Usage informatif uniquement. Le jeu peut être dangereux.</em>
    </p>
    """,
    unsafe_allow_html=True,
)
