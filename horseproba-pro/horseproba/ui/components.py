"""
Composants d'interface Streamlit.

Tous les composants reçoivent des DataFrames déjà calculés et ne font aucun
calcul métier : cela garde la logique testable hors de Streamlit.
"""

from __future__ import annotations

from typing import Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from ..features import FEATURE_LABELS

DISCLAIMER_MD = """
**⚠️ Transparence sur les limites du modèle**

- Ce pronostic est une **estimation de probabilités**, pas une prédiction du résultat.
  Un cheval estimé à 30 % **perd 7 fois sur 10**. L'incertitude est intrinsèque aux courses
  (incidents, tactique, état du terrain, forme du jour…).
- Le modèle n'utilise que les informations fournies : données incomplètes ou erronées
  ⇒ probabilités dégradées. Sans historique de résultats, il repose sur des coefficients
  *a priori* génériques, non calibrés sur votre population de courses.
- Les cotes de référence évoluent jusqu'au départ ; une « valeur » détectée peut disparaître.
- Aucun modèle ne garantit de gain. Les paris exposent à un **risque de perte financière** et
  peuvent créer une **dépendance**. Jouez de façon responsable : *Joueurs Info Service*
  **09 74 75 13 13** (appel non surtaxé). Interdit aux mineurs.
"""


def render_disclaimer(expanded: bool = False) -> None:
    with st.expander("Limites du modèle et jeu responsable", expanded=expanded):
        st.markdown(DISCLAIMER_MD)


def empty_runner_template(n: int = 8) -> pd.DataFrame:
    """Tableau vide pré-rempli pour la saisie manuelle d'une course."""
    return pd.DataFrame(
        {
            "horse": [f"Cheval {i + 1}" for i in range(n)],
            "draw": list(range(1, n + 1)),
            "odds": [None] * n,
            "musique": [""] * n,
            "jockey": [""] * n,
            "trainer": [""] * n,
            "weight_kg": [None] * n,
            "age": [None] * n,
            "days_since_last_run": [None] * n,
            "career_starts": [None] * n,
            "career_wins": [None] * n,
            "career_places": [None] * n,
            "earnings": [None] * n,
        }
    )


def render_prediction_table(pred: pd.DataFrame) -> None:
    """Tableau de classement lisible avec barres de probabilité."""
    cols = ["rank", "horse", "p_win", "p_place", "fair_odds", "odds", "market_p", "value"]
    optional = ["jockey", "trainer", "musique", "draw"]
    show = [c for c in cols if c in pred.columns]
    extra = [c for c in optional if c in pred.columns]
    table = pred[show + extra].copy()
    if "is_outsider_pick" in pred.columns:
        table.insert(1, "outsider", pred["is_outsider_pick"].map({True: "🎯 Outsider", False: ""}))
    table = table.rename(
        columns={
            "rank": "Rang",
            "outsider": "",
            "horse": "Cheval",
            "p_win": "P(victoire)",
            "p_place": "P(placé top 3)",
            "fair_odds": "Cote juste",
            "odds": "Cote marché",
            "market_p": "P(marché)",
            "value": "Écart modèle − marché",
            "jockey": "Jockey/Driver",
            "trainer": "Entraîneur",
            "musique": "Musique",
            "draw": "N°/Corde",
        }
    )
    config: Dict[str, object] = {
        "P(victoire)": st.column_config.ProgressColumn("P(victoire)", format="%.1f %%", min_value=0, max_value=100),
        "P(placé top 3)": st.column_config.ProgressColumn("P(placé top 3)", format="%.1f %%", min_value=0, max_value=100),
        "P(marché)": st.column_config.NumberColumn("P(marché)", format="%.1f %%"),
        "Écart modèle − marché": st.column_config.NumberColumn("Écart modèle − marché", format="%+.1f %%", help="Positif : le modèle estime le cheval plus probable que le marché (valeur potentielle). Négatif : le marché le surestime selon le modèle."),
        "Cote juste": st.column_config.NumberColumn("Cote juste", format="%.1f"),
        "Cote marché": st.column_config.NumberColumn("Cote marché", format="%.1f"),
    }
    # Toutes les probabilités sont affichées en pourcentage (0-100).
    for c in ["P(victoire)", "P(placé top 3)", "P(marché)", "Écart modèle − marché"]:
        if c in table.columns:
            table[c] = table[c] * 100
    st.dataframe(table, hide_index=True, use_container_width=True, column_config=config)


def render_outsiders_panel(pred: pd.DataFrame) -> None:
    """
    Met en avant les OUTSIDERS À POTENTIEL détectés (voir `model.flag_outsiders`) :
    partants peu joués par le marché mais que le modèle juge sensiblement
    sous-évalués. Section volontairement discrète (ce n'est pas une
    recommandation de pari) — voir le disclaimer.
    """
    if "is_outsider_pick" not in pred.columns:
        return
    outsiders = pred[pred["is_outsider_pick"]].sort_values("value", ascending=False)
    if outsiders.empty:
        st.caption("🔎 Aucun outsider à potentiel détecté sur cette course (marché et modèle globalement d'accord).")
        return
    st.markdown("**🎯 Outsiders à potentiel détectés** — cote élevée, mais le modèle leur donne une vraie chance :")
    for _, r in outsiders.iterrows():
        st.markdown(
            f"- **{r['horse']}** — cote {r.get('odds', float('nan')):.1f}, "
            f"P(victoire) modèle {100 * r['p_win']:.1f} % vs marché {100 * (r.get('market_p') or 0):.1f} % "
            f"(écart +{100 * r['value']:.1f} pts)"
        )
    st.caption(
        "Signal d'attention à examiner (voir « Pourquoi ce classement ? » ci-dessous), pas une recommandation de "
        "pari : le modèle ne connaît ni la marge du PMU, ni votre gestion de bankroll."
    )


def render_probability_chart(pred: pd.DataFrame) -> None:
    """Barres horizontales : P(victoire) du modèle vs marché."""
    df = pred.sort_values("p_win", ascending=True)
    fig = go.Figure()
    fig.add_bar(y=df["horse"], x=df["p_win"] * 100, name="Modèle", orientation="h", marker_color="#1b7f5c")
    if df["market_p"].notna().any():
        fig.add_bar(y=df["horse"], x=df["market_p"] * 100, name="Marché (cotes)", orientation="h", marker_color="#b8c9c0")
    fig.update_layout(
        barmode="group",
        xaxis_title="Probabilité de victoire (%)",
        yaxis_title="",
        height=max(320, 28 * len(df) + 120),
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", y=1.08),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_contributions_chart(pred: pd.DataFrame, features: List[str], horse: str) -> None:
    """Décomposition de la force d'un cheval par variable (waterfall simplifié)."""
    row = pred[pred["horse"] == horse]
    if row.empty:
        st.info("Cheval introuvable.")
        return
    row = row.iloc[0]
    data = pd.DataFrame(
        {
            "Variable": [FEATURE_LABELS.get(f, f) for f in features],
            "Contribution": [float(row.get(f"contrib_{f}", 0.0)) for f in features],
        }
    ).sort_values("Contribution")
    data["Sens"] = data["Contribution"].apply(lambda v: "Favorable" if v >= 0 else "Défavorable")
    fig = px.bar(
        data, x="Contribution", y="Variable", orientation="h", color="Sens",
        color_discrete_map={"Favorable": "#1b7f5c", "Défavorable": "#c0504d"},
    )
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), xaxis_title="Contribution à la force (échelle logit)", yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)


def render_explanations(lines: List[str]) -> None:
    for line in lines:
        st.markdown(f"- {line}")


def render_coefficients(coefs: Dict[str, float]) -> None:
    df = pd.DataFrame(
        {"Variable": [FEATURE_LABELS.get(k, k) for k in coefs], "Coefficient β": list(coefs.values())}
    ).sort_values("Coefficient β")
    fig = px.bar(df, x="Coefficient β", y="Variable", orientation="h", color="Coefficient β", color_continuous_scale=["#c0504d", "#e8e8e8", "#1b7f5c"], color_continuous_midpoint=0)
    fig.update_layout(height=440, margin=dict(l=10, r=10, t=30, b=10), coloraxis_showscale=False, yaxis_title="")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "Un coefficient positif signifie qu'un écart-type au-dessus de la moyenne de la course sur cette "
        "variable augmente la force du cheval. Les variables sont centrées par course et réduites globalement."
    )


def render_calibration_chart(calib: pd.DataFrame) -> None:
    if calib.empty:
        st.info("Pas assez de données pour tracer la calibration.")
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Calibration parfaite", line=dict(dash="dash", color="#999")))
    fig.add_trace(
        go.Scatter(
            x=calib["predite"], y=calib["observee"], mode="markers+lines", name="Modèle",
            marker=dict(size=(calib["effectif"] / calib["effectif"].max() * 25 + 6), color="#1b7f5c"),
            text=[f"n={e}" for e in calib["effectif"]],
        )
    )
    fig.update_layout(xaxis_title="Probabilité prédite", yaxis_title="Fréquence observée de victoire", height=380, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)


def render_training_log_chart(log: pd.DataFrame) -> None:
    """
    Courbe de progression du modèle au fil des ré-entraînements successifs
    (`models/training_log.csv`) : log-loss du candidat à chaque tentative,
    comparé au repère uniforme et (si disponible) au marché. Les points
    creux (promotion refusée) montrent le garde-fou à l'œuvre.
    """
    if log.empty or "candidate_logloss_backtest" not in log.columns:
        st.info("Pas encore d'historique d'entraînement (lancez `scripts/train_model.py`).")
        return
    df = log.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.sort_values("timestamp")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["candidate_logloss_backtest"], mode="markers+lines", name="Candidat (backtest)",
        marker=dict(size=10, color=df["promoted"].map({True: "#1b7f5c", False: "#c0504d"}) if "promoted" in df.columns else "#1b7f5c",
                    symbol=df["promoted"].map({True: "circle", False: "circle-open"}) if "promoted" in df.columns else "circle"),
    ))
    if "uniform_logloss_backtest" in df.columns:
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["uniform_logloss_backtest"], mode="lines", name="Repère uniforme", line=dict(dash="dot", color="#999")))
    if "market_logloss_backtest" in df.columns and df["market_logloss_backtest"].notna().any():
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["market_logloss_backtest"], mode="lines", name="Marché", line=dict(dash="dash", color="#c9a227")))
    fig.update_layout(
        yaxis_title="Log-loss (plus bas = meilleur)", xaxis_title="",
        height=380, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=1.1),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("● plein = candidat promu en production · ○ creux = promotion refusée par le garde-fou (production conservée).")


def render_live_performance_chart(daily: pd.DataFrame) -> None:
    """
    Performance RÉELLE des pronostics publiés au fil du temps (journal des
    prédictions résolu, `evaluate.summarize_live_predictions`) — par
    opposition au backtest, qui ne rejoue que le passé.
    """
    if daily.empty:
        st.info(
            "Pas encore de pronostic résolu. Lancez `scripts/log_predictions.py` avant les courses, puis "
            "`scripts/build_history.py` + `scripts/resolve_predictions.py` une fois les résultats connus "
            "(automatique via la GitHub Action planifiée)."
        )
        return
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["logloss_moyen"], mode="markers+lines", name="Log-loss quotidien", marker=dict(color="#1b7f5c")))
    fig.update_layout(yaxis_title="Log-loss moyen (plus bas = meilleur)", height=340, margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Partants évalués", int(daily["n_partants"].sum()))
    c2.metric("Log-loss moyen (période)", f"{daily['logloss_moyen'].mean():.3f}")
    if daily["favori_modele_gagnant"].notna().any():
        c3.metric("Favori du modèle gagnant", f"{100 * daily['favori_modele_gagnant'].mean():.0f} %")
