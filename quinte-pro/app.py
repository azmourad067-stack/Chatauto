"""Application Streamlit : saisie d'une course -> pronostic top 5.

Lancement local :  streamlit run app.py
Deploiement : Streamlit Community Cloud (voir README.md).

Flux : saisie -> scraper.engine.collect_race_data (web temps reel)
       -> model.predict.predict_top5 (modele entraîne) -> affichage.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.predict import load_metrics, load_model, predict_top5  # noqa: E402
from scraper.engine import collect_race_data  # noqa: E402

st.set_page_config(page_title="Quinté+ Predictor", page_icon="🐎", layout="wide")

DISCIPLINES = ["Plat", "Trot attelé", "Trot monté", "Obstacle"]
_MAP_DISCIPLINE = {
    "Plat": "plat",
    "Trot attelé": "trot_attelle",
    "Trot monté": "trot_monte",
    "Obstacle": "obstacle",
}

# ----------------------------------------------------------------------
# etat de session : liste des partants
# ----------------------------------------------------------------------
if "partants" not in st.session_state:
    st.session_state.partants = [
        {"num": 1, "horse": "", "driver": "", "age": None, "gains": None, "musique": ""},
        {"num": 2, "horse": "", "driver": "", "age": None, "gains": None, "musique": ""},
        {"num": 3, "horse": "", "driver": "", "age": None, "gains": None, "musique": ""},
        {"num": 4, "horse": "", "driver": "", "age": None, "gains": None, "musique": ""},
        {"num": 5, "horse": "", "driver": "", "age": None, "gains": None, "musique": ""},
    ]


def _clean_num(v, conv=float):
    try:
        return conv(v) if v not in (None, "") else None
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------------
# header
# ----------------------------------------------------------------------
st.title("🐎 Quinté+ Predictor")
st.caption(
    "Pronostic assisté : recherche web en temps réel + modèle de scoring entraîné. "
    "Indication statistique uniquement — aucune garantie de résultat."
)

# ----------------------------------------------------------------------
# saisie de la course
# ----------------------------------------------------------------------
with st.form("course"):
    c1, c2, c3, c4 = st.columns(4)
    hippodrome = c1.text_input("Hippodrome", value="Vincennes")
    discipline = c2.selectbox("Discipline", DISCIPLINES)
    distance = c3.number_input("Distance (m)", min_value=800, max_value=5000, value=2700, step=100)
    date_course = c4.text_input("Date (JJ/MM/AAAA, optionnel)", value="",
                                help="Requis pour interroger open-pmu-api (arrivées passées)")
    st.form_submit_button("Enregistrer la course", use_container_width=True)

st.divider()
st.subheader("Partants")

# ----------------------------------------------------------------------
# edition des partants
# ----------------------------------------------------------------------
partants_bruts: list[dict] = []
to_delete: list[int] = []
for idx, part in enumerate(st.session_state.partants):
    cols = st.columns([1, 3, 3, 1.5, 2, 2.5, 1])
    num = cols[0].number_input("N°", min_value=1, max_value=30, value=int(part["num"] or idx + 1),
                               key=f"num_{idx}", label_visibility="collapsed")
    horse = cols[1].text_input("Cheval", value=part["horse"], key=f"horse_{idx}",
                               placeholder="Nom du cheval", label_visibility="collapsed")
    driver = cols[2].text_input("Driver", value=part["driver"], key=f"driver_{idx}",
                                placeholder="Nom du driver", label_visibility="collapsed")
    age = cols[3].number_input("Âge", min_value=2, max_value=15, value=int(part["age"] or 5),
                               key=f"age_{idx}", label_visibility="collapsed")
    gains = cols[4].number_input("Gains (€)", min_value=0, value=int(part["gains"] or 0),
                                 step=1000, key=f"gains_{idx}", label_visibility="collapsed")
    musique = cols[5].text_input("Musique", value=part["musique"], key=f"musique_{idx}",
                                 placeholder="ex: 1p 2p 4p (25) 3p", label_visibility="collapsed")
    if cols[6].button("🗑", key=f"del_{idx}", help="Supprimer ce partant"):
        to_delete.append(idx)
    partants_bruts.append({
        "num": num,
        "horse": horse.strip(),
        "driver": driver.strip(),
        "age": _clean_num(age, int),
        "gains": _clean_num(gains),
        "musique": musique.strip(),
    })

# suppression (a faire apres la boucle pour ne pas casser les cles)
for idx in sorted(to_delete, reverse=True):
    if len(st.session_state.partants) > 3:
        st.session_state.partants.pop(idx)

b_add, b_run, _ = st.columns([1, 2, 3])
if b_add.button("+ Ajouter un partant"):
    st.session_state.partants.append(
        {"num": len(st.session_state.partants) + 1, "horse": "", "driver": "",
         "age": None, "gains": None, "musique": ""}
    )
    st.rerun()

# ----------------------------------------------------------------------
# pronostic
# ----------------------------------------------------------------------
if b_run.button("🔎 Générer le pronostic", type="primary", use_container_width=True):
    partants_ok = [p for p in partants_bruts if p["horse"]]
    if len(partants_ok) < 5:
        st.error(f"Au moins 5 partants avec un nom de cheval sont requis (actuel: {len(partants_ok)}).")
    else:
        race = {
            "hippodrome": hippodrome.strip(),
            "discipline": _MAP_DISCIPLINE[discipline],
            "distance": int(distance),
            "date": date_course.strip() or None,
        }
        with st.spinner("Recherche web en temps réel puis scoring du modèle…"):
            try:
                collect = collect_race_data(race, partants_ok)
            except Exception as exc:  # filet de securite
                st.warning(f"Collecte web défaillante ({exc}) : pronostic sur saisie seule.")
                from scraper.engine import EnrichedPartant

                collect = type("R", (), {
                    "partants": [EnrichedPartant(num=p["num"], horse=p["horse"],
                                                 driver=p["driver"], age=p["age"],
                                                 gains=p["gains"], musique=p["musique"],
                                                 data_sources=["saisie utilisateur"])
                                 for p in partants_ok],
                    "sources_report": [{"source": "tout", "etat": "exception", "utilisee": False}],
                    "quality": 0.25, "warnings": ["Collecte défaillante."], "race": race,
                })()

            bundle = load_model()
            out = predict_top5(race, [p.to_dict() for p in collect.partants], bundle)

        # ---- avertissements ----
        for w in collect.warnings:
            st.warning(w)

        # ---- top 5 ----
        st.subheader(f"🏆 Pronostic — Top 5 ({hippodrome}, {discipline}, {distance} m)")
        medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]
        cols = st.columns(5)
        for i, row in enumerate(out["top5"]):
            with cols[i]:
                st.markdown(
                    f"""
                    <div style="border:1px solid #d8dee8;border-radius:12px;padding:14px;
                                background:#ffffff;text-align:center;height:100%">
                      <div style="font-size:26px">{medals[i]}</div>
                      <div style="font-size:15px;font-weight:700;min-height:44px">{row['cheval']}</div>
                      <div style="font-size:12px;color:#5a6472">{row['driver']}</div>
                      <div style="margin-top:8px;font-size:20px;font-weight:800;color:{'#1B3A6B'}">
                        {row['proba_top5']:.0%}</div>
                      <div style="font-size:11px;color:#8a94a3">P(top 5)</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # ---- tableau detail ----
        with st.expander("📊 Classement complet et scores", expanded=False):
            detail = []
            for r in out["classement_complet"]:
                detail.append({
                    "Rang": r["rang"], "N°": r["num"], "Cheval": r["cheval"],
                    "Driver": r["driver"], "Score modèle": r["score"],
                    "P(top 5)": f"{r['proba_top5']:.1%}",
                    "Sentiment web": r["sentiment"], "Sources": ", ".join(r["sources"]),
                })
            st.dataframe(detail, use_container_width=True, hide_index=True,
                         column_config={"Score modèle": st.column_config.NumberColumn(format="%.3f")})

        # ---- provenance des donnees ----
        with st.expander("🔎 Sources interrogées en temps réel", expanded=False):
            for s in collect.sources_report:
                icone = "✅" if s.get("utilisee") else ("⚠️" if "inactif" in s.get("etat", "") else "⛔")
                st.markdown(f"{icone} **{s['source']}** — {s['etat']}")
            st.caption(f"Qualité globale des données collectées : **{collect.quality:.0%}**")

        # ---- actualites ----
        snippets = [sn for p in collect.partants for sn in p.web_snippets]
        with st.expander("📰 Actualités collectées (snippets)", expanded=False):
            if snippets:
                for sn in snippets[:12]:
                    st.markdown(f"- {sn}")
            else:
                st.caption("Aucun snippet récupéré (source web indisponible ou désactivée).")

        st.info(
            "⚠️ Les probabilités affichées sont des sorties de modèles statistiques, "
            "pas des garanties. Jouez avec modération — jeux d'argent interdits aux mineurs "
            "(ANJ : 09 74 75 13 13)."
        )

# ----------------------------------------------------------------------
# onglet modele
# ----------------------------------------------------------------------
with st.expander("🤖 Modèle & entraînement", expanded=False):
    metrics = load_metrics()
    if metrics:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("RMSE place (norm.)", metrics.get("rmse_place_norm"))
        m2.metric("Corrélation Spearman", metrics.get("spearman_place"))
        m3.metric("AUC top-5", metrics.get("auc_top5"))
        m4.metric("Partants d'entraînement", f"{metrics.get('n_lignes', 0):,}".replace(",", " "))
        st.caption(f"Entraîné le {metrics.get('train_date', '?')} — source: {metrics.get('source', '?')}")
    else:
        st.warning("Aucune métrique trouvée : lancez d'abord `python model/train.py`.")

    if st.button("🔄 Ré-entraîner le modèle (quelques minutes)"):
        import subprocess

        with st.spinner("Entraînement en cours…"):
            proc = subprocess.run([sys.executable, str(ROOT / "model" / "train.py")],
                                  capture_output=True, text=True)
        if proc.returncode == 0:
            st.success("Entraînement terminé, métriques actualisées. Rechargez la page.")
        else:
            st.error(proc.stderr[-1500:])

st.divider()
st.caption(
    "Code fourni à but éducatif. Respectez les conditions d'utilisation des sites interrogés, "
    "le fichiers robots.txt et la réglementation (ANJ). Voir README.md pour les limites."
)
