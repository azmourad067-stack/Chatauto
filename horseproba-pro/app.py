"""
Point d'entrée Streamlit — Pronostics hippiques probabilistes.

Lancement local :   streamlit run app.py
Déploiement :       Streamlit Community Cloud (voir README.md)

Architecture :
    app.py                  ← orchestration de l'interface (ce fichier)
    horseproba/data/        ← chargement CSV, source web PMU, données synthétiques
    horseproba/features.py  ← ingénierie des variables
    horseproba/model.py     ← logit conditionnel (Bradley-Terry / Plackett-Luce)
    horseproba/model_gbm.py ← modèle gradient boosting (interactions non-linéaires)
    horseproba/ensemble.py  ← combinaison logit + GBM
    horseproba/registry.py  ← modèle de production persisté + journal d'entraînement
    horseproba/evaluate.py  ← métriques, backtest walk-forward, suivi live
    horseproba/ui/          ← composants d'affichage

Le modèle utilisé pour pronostiquer est, par défaut, le MODÈLE DE PRODUCTION
persisté par `scripts/train_model.py` (voir onglet « Auto-apprentissage ») —
et non plus systématiquement ré-entraîné à la volée à chaque session : c'est
plus rapide, et c'est la trace tangible de l'auto-apprentissage au fil des
jours. L'entraînement à la volée sur un historique choisi reste disponible
dans la barre latérale pour l'exploration et la pédagogie.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
import streamlit as st

from horseproba import __version__, registry
from horseproba.data import pmu
from horseproba.data.loader import (
    DataValidationError,
    load_runners_csv,
    load_sample_history,
    sample_dataset_path,
    validate_runners,
)
from horseproba.data.synthetic import generate_history
from horseproba.ensemble import EnsembleModel
from horseproba.evaluate import EvalReport, backtest, summarize_live_predictions
from horseproba.features import FEATURE_LABELS
from horseproba.model import ConditionalLogitModel, flag_outsiders
from horseproba.model_gbm import GradientBoostingRanker
from horseproba.ui import (
    empty_runner_template,
    render_calibration_chart,
    render_coefficients,
    render_contributions_chart,
    render_disclaimer,
    render_explanations,
    render_live_performance_chart,
    render_outsiders_panel,
    render_prediction_table,
    render_probability_chart,
    render_training_log_chart,
)

AnyModel = Union[ConditionalLogitModel, EnsembleModel]


def _features_of(model: AnyModel) -> List[str]:
    return model.features if hasattr(model, "features") else model.logit.features


def _logit_of(model: AnyModel) -> ConditionalLogitModel:
    """Composante logit d'un modèle (identité si déjà un logit) — pour l'affichage des coefficients."""
    return model.logit if isinstance(model, EnsembleModel) else model

# --------------------------------------------------------------------------- #
# Configuration de la page
# --------------------------------------------------------------------------- #
st.set_page_config(page_title="HorseProba – Pronostics hippiques probabilistes", page_icon="🐎", layout="wide")


def _secret(key: str, default):
    """Lecture tolérante de st.secrets (absent en local si aucun fichier secrets.toml)."""
    try:
        return st.secrets.get(key, default)
    except Exception:  # noqa: BLE001 — pas de secrets configurés
        return default


USER_AGENT: str = str(_secret("HTTP_USER_AGENT", "HorseProbaApp/1.0 (educational)"))
WEB_ENABLED: bool = bool(_secret("ENABLE_WEB_FETCH", True))


# --------------------------------------------------------------------------- #
# Fonctions mises en cache
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False)
def cached_synthetic(n_races: int, seed: int) -> pd.DataFrame:
    return generate_history(n_races=n_races, seed=seed)


@st.cache_data(show_spinner=False)
def cached_sample() -> pd.DataFrame:
    return load_sample_history()


COLLECTED_HISTORY_PATH = Path("data/history_pmu.csv")
PREDICTIONS_LOG_PATH = Path("data/predictions_log.csv")
REGISTRY_DIR = registry.DEFAULT_REGISTRY_DIR


@st.cache_data(show_spinner=False)
def cached_collected(mtime: float) -> pd.DataFrame:
    """Historique réel produit par scripts/build_history.py (clé de cache : date de modification)."""
    df, _ = load_runners_csv(COLLECTED_HISTORY_PATH)
    return df


@st.cache_data(show_spinner=False, ttl=600)
def cached_program(day: date) -> List[pmu.RaceRef]:
    return pmu.fetch_program(day, USER_AGENT)


@st.cache_data(show_spinner=False, ttl=120)
def cached_runners(
    day: date, reunion: int, course: int, label: str, track: str, discipline: str, distance: Optional[int]
) -> pd.DataFrame:
    ref = pmu.RaceRef(
        day=day, reunion=reunion, course=course, label=label, track=track, discipline=discipline, distance_m=distance
    )
    return pmu.fetch_runners(ref, USER_AGENT)


@st.cache_resource(show_spinner=False)
def fit_ensemble_cached(history_key: str, l2: float, gbm_weight: float, _history: pd.DataFrame) -> EnsembleModel:
    """Entraîne (et met en cache) un ensemble logit+GBM sur l'historique choisi.
    `history_key` + `l2` + `gbm_weight` forment la clé de cache."""
    model = EnsembleModel(logit=ConditionalLogitModel(l2=l2), gbm=GradientBoostingRanker(), gbm_weight=gbm_weight)
    model.fit(_history)
    return model


@st.cache_resource(show_spinner=False)
def fit_logit_cached(history_key: str, l2: float, _history: pd.DataFrame) -> ConditionalLogitModel:
    """Entraîne (et met en cache) le logit conditionnel seul (mode pédagogique/rapide)."""
    model = ConditionalLogitModel(l2=l2)
    model.fit(_history)
    return model


@st.cache_resource(show_spinner=False)
def cached_production_model(mtime: float) -> Optional[EnsembleModel]:
    """Modèle de production persisté par `scripts/train_model.py` (clé de cache : mtime de meta.json)."""
    return registry.load_model(registry.production_dir(REGISTRY_DIR))


@st.cache_data(show_spinner=False)
def cached_production_meta(mtime: float) -> Optional[dict]:
    return registry.load_metadata(registry.production_dir(REGISTRY_DIR))


@st.cache_data(show_spinner=False)
def cached_training_log(mtime: float) -> pd.DataFrame:
    return registry.load_training_log(REGISTRY_DIR)


@st.cache_data(show_spinner=False)
def cached_live_performance(mtime: float) -> pd.DataFrame:
    if not PREDICTIONS_LOG_PATH.exists():
        return pd.DataFrame()
    try:
        log = pd.read_csv(PREDICTIONS_LOG_PATH)
    except Exception:  # noqa: BLE001
        return pd.DataFrame()
    return summarize_live_predictions(log)


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner=False)
def cached_backtest(history_key: str, l2: float, gbm_weight: float, _history: pd.DataFrame) -> EvalReport:
    if gbm_weight <= 0:
        return backtest(_history, l2=l2)
    return backtest(
        _history,
        model_factory=lambda: EnsembleModel(
            logit=ConditionalLogitModel(l2=l2), gbm=GradientBoostingRanker(), gbm_weight=gbm_weight
        ),
    )


# --------------------------------------------------------------------------- #
# Barre latérale : historique d'entraînement et paramètres
# --------------------------------------------------------------------------- #

# Barre latérale : choix du modèle et historique d'entraînement
# --------------------------------------------------------------------------- #
@dataclass
class ModelChoice:
    mode: str  # "production" | "train_ensemble" | "train_logit" | "prior"
    l2: float = 1.0
    gbm_weight: float = 0.35


def sidebar_history() -> Tuple[pd.DataFrame, str]:
    st.sidebar.header("1 · Historique (pour entraîner à la volée)")
    has_collected = COLLECTED_HISTORY_PATH.exists()
    options = []
    if has_collected:
        options.append("Historique PMU réel (data/history_pmu.csv)")
    options += [
        "Démonstration (synthétique)",
        "Exemple CSV embarqué",
        "Mon fichier CSV (résultats)",
        "Aucun",
    ]
    source = st.sidebar.radio(
        "Source de l'historique",
        options,
        help="Ne sert que si vous choisissez d'entraîner un modèle « à la volée » ci-dessous. "
        "Le modèle de production, lui, a déjà été entraîné par `scripts/train_model.py`.",
    )
    history = pd.DataFrame()
    key = "none"
    if source.startswith("Historique PMU réel"):
        try:
            mtime = COLLECTED_HISTORY_PATH.stat().st_mtime
            history = cached_collected(mtime)
            key = f"collected_{int(mtime)}"
            if "finish_position" not in history.columns:
                st.sidebar.error("`data/history_pmu.csv` ne contient pas de colonne finish_position.")
                history = pd.DataFrame()
        except DataValidationError as exc:
            st.sidebar.error(f"Historique réel illisible : {exc}")
    elif source == "Démonstration (synthétique)":
        n_races = st.sidebar.slider("Nombre de courses simulées", 100, 1000, 300, 50)
        seed = st.sidebar.number_input("Graine aléatoire", 0, 9999, 42)
        history = cached_synthetic(n_races, int(seed))
        key = f"synthetic_{n_races}_{seed}"
    elif source == "Exemple CSV embarqué":
        history = cached_sample()
        key = "sample"
    elif source == "Mon fichier CSV (résultats)":
        up = st.sidebar.file_uploader("CSV avec colonne finish_position", type=["csv"], key="hist_upload")
        if up is not None:
            try:
                history, warns = load_runners_csv(up.getvalue())
                if "finish_position" not in history.columns:
                    st.sidebar.error("La colonne `finish_position` est requise pour un historique.")
                    history = pd.DataFrame()
                else:
                    key = f"upload_{up.name}_{up.size}"
                    st.sidebar.success(f"{history['race_id'].nunique()} courses · {len(history)} partants")
                for w in warns:
                    st.sidebar.warning(w)
            except DataValidationError as exc:
                st.sidebar.error(str(exc))

    if not history.empty:
        st.sidebar.caption(f"Historique sélectionné : **{history['race_id'].nunique()}** courses, **{len(history)}** partants.")
    return history, key


def sidebar_model_choice(history_available: bool) -> ModelChoice:
    st.sidebar.header("2 · Modèle de pronostic")
    prod_meta = cached_production_meta(_mtime(registry.production_dir(REGISTRY_DIR) / "meta.json"))
    options = []
    if prod_meta is not None:
        options.append("🏭 Modèle de production (auto-appris)")
    options.append("🔧 Entraîner à la volée — Ensemble (logit + boosting)")
    options.append("🔧 Entraîner à la volée — Logit conditionnel seul")
    options.append("📐 Coefficients a priori (sans historique)")

    mode_label = st.sidebar.radio(
        "Quel modèle utiliser ?",
        options,
        help="Le modèle de production est celui journalisé par la boucle d'auto-apprentissage "
        "(`scripts/train_model.py`) : recommandé pour un usage réel. Les autres options servent "
        "à l'exploration pédagogique sur un historique de votre choix.",
    )
    if prod_meta is not None:
        trained_at = str(prod_meta.get("saved_at", "?"))[:19].replace("T", " ")
        gbm_state = "GBM actif" if prod_meta.get("gbm_fitted") else "GBM non entraîné (repli logit)"
        st.sidebar.caption(f"Production : entraîné le {trained_at} UTC · {gbm_state}.")

    if mode_label.startswith("🏭"):
        return ModelChoice(mode="production")
    if mode_label.startswith("📐"):
        return ModelChoice(mode="prior")

    if not history_available:
        st.sidebar.warning("Aucun historique sélectionné ci-dessus : repli sur les coefficients a priori.")
        return ModelChoice(mode="prior")

    l2 = st.sidebar.slider(
        "Régularisation L2 (λ)", 0.1, 10.0, 1.0, 0.1,
        help="Plus λ est grand, plus les coefficients restent proches des valeurs a priori (utile si peu de courses).",
    )
    if mode_label.startswith("🔧 Entraîner à la volée — Ensemble"):
        gbm_weight = st.sidebar.slider(
            "Poids du boosting dans l'ensemble", 0.0, 0.8, 0.35, 0.05,
            help="0 = logit seul. Le GBM ne s'active de toute façon qu'à partir de ~40 courses gagnantes exploitables.",
        )
        return ModelChoice(mode="train_ensemble", l2=l2, gbm_weight=gbm_weight)
    return ModelChoice(mode="train_logit", l2=l2)


def sidebar_footer() -> None:
    st.sidebar.markdown("---")
    if st.sidebar.button("🔄 Actualiser les données web", help="Vide le cache des données PMU."):
        cached_program.clear()
        cached_runners.clear()
        st.sidebar.success("Cache vidé.")


# --------------------------------------------------------------------------- #
# Sélection / saisie de la course à pronostiquer
# --------------------------------------------------------------------------- #
def race_input_section() -> Optional[pd.DataFrame]:
    """Renvoie un DataFrame de partants (une course) ou None."""
    mode = st.radio(
        "Comment fournir la course ?",
        ["📡 Programme du jour (PMU, web)", "📄 Importer un CSV", "✍️ Saisie manuelle"],
        horizontal=True,
    )
    if mode.startswith("📡"):
        return _pmu_input()
    if mode.startswith("📄"):
        return _csv_input()
    return _manual_input()


def _pmu_input() -> Optional[pd.DataFrame]:
    if not WEB_ENABLED:
        st.warning("La récupération web est désactivée (ENABLE_WEB_FETCH=false). Utilisez l'import CSV.")
        return None
    st.caption(
        "Source : endpoints JSON publics du site pmu.fr (non officiels, sans clé). "
        "En cas d'indisponibilité, basculez sur l'import CSV."
    )
    c1, c2 = st.columns([1, 3])
    with c1:
        day = st.date_input(
            "Date",
            value=date.today(),
            min_value=date.today() - timedelta(days=60),
            max_value=date.today() + timedelta(days=2),
        )
    try:
        with st.spinner("Chargement du programme…"):
            refs = cached_program(day)
    except pmu.PMUFetchError as exc:
        st.error(f"Programme indisponible : {exc}")
        return None
    except Exception as exc:  # noqa: BLE001 — ne jamais faire crasher l'interface
        st.error(f"Erreur inattendue lors de la récupération du programme : {exc}")
        return None

    with c2:
        labels = [r.label for r in refs]
        choice = st.selectbox("Course", labels)
    ref = refs[labels.index(choice)]
    try:
        with st.spinner("Chargement des partants…"):
            runners = cached_runners(
                ref.day, ref.reunion, ref.course, ref.label, ref.track, ref.discipline, ref.distance_m
            )
        runners, warns = validate_runners(runners)
        for w in warns:
            st.warning(w)
        st.success(f"{len(runners)} partants chargés — {ref.track}, {ref.discipline}, {ref.distance_m or '?'} m")
        return runners
    except (pmu.PMUFetchError, DataValidationError) as exc:
        st.error(f"Partants indisponibles : {exc}")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Erreur inattendue : {exc}")
    return None


def _csv_input() -> Optional[pd.DataFrame]:
    st.caption(
        "Colonnes minimales : `race_id`, `horse`. Recommandées : `odds`, `musique`, `jockey`, `trainer`, "
        "`draw`, `weight_kg`… (voir onglet Méthode)."
    )
    up = st.file_uploader("Fichier CSV des partants", type=["csv"], key="race_upload")
    use_sample = st.checkbox("Utiliser le fichier d'exemple `data/sample_races.csv`", value=up is None)
    try:
        if up is not None:
            df, warns = load_runners_csv(up.getvalue())
        elif use_sample:
            df, warns = load_runners_csv(sample_dataset_path())
        else:
            return None
    except DataValidationError as exc:
        st.error(str(exc))
        return None
    for w in warns:
        st.warning(w)
    races = list(df["race_id"].unique())
    race = st.selectbox("Course à pronostiquer", races, index=len(races) - 1)
    return df[df["race_id"] == race].reset_index(drop=True)


def _manual_input() -> Optional[pd.DataFrame]:
    st.caption(
        "Renseignez au moins le nom des chevaux. Les cotes (décimales, ex. 4.5) et la musique "
        "(ex. `1p 3p 2p 0p`) améliorent nettement le pronostic."
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        n = st.number_input("Nombre de partants", 2, 24, 8)
    with c2:
        discipline = st.selectbox("Discipline", ["plat", "trot", "obstacle"])
    with c3:
        distance = st.number_input("Distance (m)", 800, 7000, 1600, 100)
    with c4:
        going = st.text_input("Terrain", "bon")
    template = empty_runner_template(int(n))
    edited = st.data_editor(template, num_rows="dynamic", use_container_width=True, hide_index=True, key="manual_editor")
    df = edited.copy()
    df["race_id"] = "saisie_manuelle"
    df["discipline"] = discipline
    df["distance_m"] = distance
    df["going"] = going
    df["race_date"] = date.today().isoformat()
    try:
        df, warns = validate_runners(df)
    except DataValidationError as exc:
        st.error(str(exc))
        return None
    for w in warns:
        if "recommandées" not in w:  # bruit inutile en saisie manuelle
            st.warning(w)
    return df


# --------------------------------------------------------------------------- #
# Onglet Pronostic
# --------------------------------------------------------------------------- #
def tab_prediction(model: AnyModel) -> None:
    st.subheader("Course à pronostiquer")
    runners = race_input_section()
    if runners is None or runners.empty:
        st.info("Fournissez une course pour obtenir un pronostic.")
        return

    n_places = st.slider("Nombre de places payées (pour P(placé))", 2, 5, 3 if len(runners) >= 8 else 2)
    try:
        pred = model.predict(runners, n_places=n_places)
        pred["is_outsider_pick"] = flag_outsiders(pred)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Le calcul du pronostic a échoué : {exc}")
        return

    st.subheader("Classement par probabilité")
    render_disclaimer(expanded=False)
    top = pred.iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Favori du modèle", top["horse"], f"{100 * top['p_win']:.1f} % de victoire")
    if pred["market_p"].notna().any():
        c2.metric(
            "Favori du marché", pred.loc[pred["market_p"].idxmax(), "horse"], f"{100 * pred['market_p'].max():.1f} %"
        )
    else:
        c2.metric("Favori du marché", "n/d", "cotes absentes")
    p = pred["p_win"].clip(lower=1e-9)
    openness = float(-(p * p.apply(math.log)).sum() / math.log(len(pred))) if len(pred) > 1 else 0.0
    c3.metric(
        "Course ouverte ?",
        f"{100 * openness:.0f} %",
        help="Entropie normalisée : 100 % = tous les chevaux à égalité, 0 % = un seul cheval possible.",
    )

    render_prediction_table(pred)
    render_outsiders_panel(pred)
    render_probability_chart(pred)

    st.subheader("Pourquoi ce classement ?")
    horse = st.selectbox("Détailler un cheval", pred["horse"].tolist())
    idx = int(pred.index[pred["horse"] == horse][0])
    left, right = st.columns([1, 1])
    with left:
        st.markdown(
            f"**{horse}** — rang {pred.loc[idx, 'rank']} · P(victoire) {100 * pred.loc[idx, 'p_win']:.1f} % · "
            f"P(placé) {100 * pred.loc[idx, 'p_place']:.1f} %"
        )
        render_explanations(model.explain(pred, idx))
        if pd.notna(pred.loc[idx, "value"]):
            v = float(pred.loc[idx, "value"])
            if v > 0.02:
                st.success(
                    f"Le modèle estime ce cheval **{100 * v:.1f} points** au-dessus du marché : valeur potentielle "
                    "(à confirmer avec les cotes définitives)."
                )
            elif v < -0.02:
                st.warning(f"Le marché le juge **{100 * -v:.1f} points** plus probable que le modèle : cote jugée trop courte.")
            else:
                st.info("Modèle et marché sont d'accord sur ce cheval.")
    with right:
        render_contributions_chart(pred, _features_of(model), horse)

    export = pred.drop(columns=[c for c in pred.columns if c.startswith("contrib_")])
    st.download_button(
        "⬇️ Télécharger le pronostic (CSV)",
        export.to_csv(index=False).encode("utf-8"),
        file_name="pronostic.csv",
        mime="text/csv",
    )


# --------------------------------------------------------------------------- #
# Onglet Modèle & backtest
# --------------------------------------------------------------------------- #
def tab_model(model: AnyModel, history: pd.DataFrame, history_key: str, l2: float, gbm_weight: float) -> None:
    logit = _logit_of(model)
    st.subheader("Coefficients estimés (composante logit conditionnel)")
    fr = logit.fit_result
    if fr is None or fr.n_races == 0:
        st.info("Aucun historique : coefficients a priori (non estimés).")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Courses d'entraînement", fr.n_races)
        c2.metric("Partants", fr.n_runners)
        c3.metric(
            "Pseudo-R² (McFadden)",
            f"{fr.pseudo_r2:.3f}",
            help="0 = pas mieux que le hasard, 1 = parfait. 0,15–0,35 est typique d'un bon modèle hippique.",
        )
        c4.metric("Convergence", "✅" if fr.converged else "⚠️")
        if not fr.converged:
            st.caption(fr.message)
    render_coefficients(logit.coefs)

    if isinstance(model, EnsembleModel):
        st.subheader("Composante Gradient Boosting")
        gfr = model.gbm.fit_result
        if gfr is None or not gfr.fitted:
            st.info(
                (gfr.message if gfr else "Non entraîné.") + " L'ensemble repose donc entièrement sur le logit "
                f"(poids GBM configuré : {model.gbm_weight:.0%}, effectif : 0 %)."
            )
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Courses gagnantes exploitées", gfr.n_races)
            c2.metric("Partants", gfr.n_runners)
            c3.metric("Poids dans l'ensemble", f"{model.gbm_weight:.0%}")
            if gfr.feature_importances:
                imp = pd.DataFrame(
                    {"Variable": [FEATURE_LABELS.get(k, k) for k in gfr.feature_importances],
                     "Importance": list(gfr.feature_importances.values())}
                ).sort_values("Importance", ascending=False).head(8)
                st.caption("Variables les plus utiles au GBM (dégradation de la log-vraisemblance si mélangées) :")
                st.dataframe(imp, hide_index=True, use_container_width=True)

    st.subheader("Backtest hors-échantillon (walk-forward)")
    st.caption(
        "Le modèle est entraîné sur les courses les plus anciennes puis évalué sur les suivantes, "
        "par blocs successifs. Aucune information future n'est utilisée."
    )
    if history.empty or history["race_id"].nunique() < 10:
        st.info("Au moins 10 courses terminées sont nécessaires pour un backtest (utilisez un historique entraîné à la volée).")
        return
    if st.button("▶️ Lancer le backtest"):
        try:
            with st.spinner("Backtest en cours…"):
                report = cached_backtest(history_key, l2, gbm_weight, history)
        except ValueError as exc:
            st.error(str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            st.error(f"Backtest impossible : {exc}")
            return
        c1, c2 = st.columns([1, 1])
        with c1:
            st.dataframe(report.as_table(), hide_index=True, use_container_width=True)
            if report.logloss_market is not None:
                if report.logloss_model < report.logloss_market:
                    st.success(
                        "Le modèle bat le marché en log-loss sur cet historique (rare sur des données réelles : "
                        "vérifiez la qualité des cotes utilisées)."
                    )
                else:
                    st.info(
                        "Le marché reste meilleur que le modèle en log-loss : résultat attendu sur des données "
                        "réelles, les cotes agrègent énormément d'information."
                    )
        with c2:
            st.markdown("**Calibration** — les points doivent suivre la diagonale.")
            render_calibration_chart(report.calibration)


# --------------------------------------------------------------------------- #
# Onglet Auto-apprentissage
# --------------------------------------------------------------------------- #
def tab_learning() -> None:
    st.subheader("Modèle de production")
    prod_dir = registry.production_dir(REGISTRY_DIR)
    meta = cached_production_meta(_mtime(prod_dir / "meta.json"))
    if meta is None:
        st.info(
            "Aucun modèle de production pour l'instant. Lancez `python scripts/train_model.py` après avoir "
            "constitué un historique (`python scripts/build_history.py --days 30`), ou attendez la prochaine "
            "exécution planifiée de la GitHub Action `learning_loop.yml`."
        )
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Entraîné le", str(meta.get("saved_at", "?"))[:19].replace("T", " ") + " UTC")
        c2.metric("Courses utilisées", meta.get("logit_n_races", "?"))
        c3.metric("Pseudo-R² (logit)", f"{meta.get('logit_pseudo_r2', float('nan')):.3f}" if meta.get("logit_pseudo_r2") is not None else "n/d")
        c4.metric("GBM actif", "Oui" if meta.get("gbm_fitted") else "Non")
        metrics = meta.get("metrics") or {}
        if metrics.get("backtest_logloss") is not None:
            st.caption(
                f"Log-loss backtest au moment de la promotion : **{metrics['backtest_logloss']:.4f}** "
                f"(uniforme : {metrics.get('backtest_logloss_uniform', float('nan')):.4f}"
                + (f", marché : {metrics['backtest_logloss_market']:.4f}" if metrics.get("backtest_logloss_market") is not None else "")
                + f") — motif de promotion : {metrics.get('promotion_reason', 'n/d')}."
            )

    st.subheader("Progression du modèle au fil des ré-entraînements")
    log = cached_training_log(_mtime(REGISTRY_DIR / "training_log.csv"))
    render_training_log_chart(log)
    if not log.empty:
        with st.expander("Journal brut des entraînements"):
            st.dataframe(log.sort_values("timestamp", ascending=False), hide_index=True, use_container_width=True)

    st.subheader("Performance réelle des pronostics publiés")
    st.caption(
        "Contrairement au backtest (qui rejoue le passé), ceci mesure la qualité des pronostics tels qu'ils ont "
        "été journalisés AVANT les courses par `scripts/log_predictions.py`, une fois les résultats connus."
    )
    daily = cached_live_performance(_mtime(PREDICTIONS_LOG_PATH))
    render_live_performance_chart(daily)

    st.markdown("---")
    st.markdown(
        """
**Comment fonctionne la boucle d'auto-apprentissage ?**

1. `scripts/build_history.py` collecte chaque jour les nouvelles courses terminées.
2. `scripts/resolve_predictions.py` rapproche les pronostics publiés la veille des résultats.
3. `scripts/train_model.py` ré-entraîne un modèle candidat et ne le **promeut en production que
   s'il n'est pas significativement pire** que l'actuel (backtest walk-forward, garde-fou ±5 % par défaut).
4. `scripts/log_predictions.py` journalise les pronostics du jour avec le modèle (nouvellement promu ou non).

Ces 4 étapes s'enchaînent automatiquement chaque jour via la GitHub Action
`.github/workflows/learning_loop.yml`. Voir le README pour le détail.
"""
    )


# --------------------------------------------------------------------------- #
# Onglet Méthode
# --------------------------------------------------------------------------- #
def tab_method() -> None:
    st.subheader("Approche statistique")
    st.markdown(
        r"""
**Modèle : ensemble « logit conditionnel + gradient boosting »**

*Composante 1 — logit conditionnel (Bradley-Terry / Plackett-Luce généralisé).*
Chaque cheval $i$ reçoit une force latente $s_i = \beta \cdot x_i$ où $x_i$ regroupe ses
variables (cote, forme, jockey…) **centrées par rapport aux autres partants de la course**.
La probabilité de victoire est un softmax intra-course :

$$P(i \text{ gagne}) = \frac{e^{s_i}}{\sum_j e^{s_j}}$$

Ce choix est justifié parce qu'une course est un **choix discret à un seul gagnant** :
les probabilités somment à 1 par construction, contrairement à une régression logistique
cheval-par-cheval. La probabilité de **placé** découle des mêmes forces via le modèle de
Plackett-Luce (énumération exacte ou Monte-Carlo). Les coefficients sont estimés par maximum
de vraisemblance conditionnelle **pénalisé (ridge)** autour de valeurs a priori, ce qui rend le
modèle robuste avec peu de courses. Références : McFadden (1974) ; Bolton & Chapman (1986) ;
Benter (1994).

*Composante 2 — gradient boosting (`HistGradientBoostingClassifier`).* Un classifieur binaire
« gagnant / non-gagnant » sur arbres boostés, dont le score est reconverti en force latente puis
renormalisé par le même softmax intra-course. Il capture des **interactions non-linéaires**
(ex. tel driver + tel terrain + telle distance) que le logit, additif, ne peut pas représenter —
c'est le principal levier pour repérer des **outsiders à potentiel** que le marché sous-évalue.
Ne s'active qu'à partir d'un historique suffisant (~40 courses gagnantes exploitables), sinon
l'ensemble repose entièrement sur le logit.

Les deux probabilités sont combinées par **moyenne géométrique pondérée** (opinion pooling
log-linéaire), un poids modeste (35 % par défaut) étant donné au boosting, le modèle le plus
récent de l'application. Voir `horseproba/ensemble.py` pour le détail.

**Détection d'outsiders.** Un partant est signalé « 🎯 Outsider » si sa cote est élevée, que le
modèle l'estime sensiblement au-dessus du marché, et qu'il lui donne une probabilité de victoire
non négligeable (voir `horseproba/model.py::flag_outsiders`). C'est un signal d'attention, pas
une recommandation de pari.

**Auto-apprentissage.** Le modèle affiché par défaut (« Modèle de production ») est ré-entraîné
et journalisé automatiquement chaque jour, avec un garde-fou qui ne le remplace que s'il n'est
pas significativement pire que l'actuel — voir l'onglet « 🧠 Auto-apprentissage » pour le détail
et la courbe de progression.

**Variables utilisées**
"""
    )
    st.table(pd.DataFrame({"Variable": list(FEATURE_LABELS.keys()), "Signification": list(FEATURE_LABELS.values())}))

    st.subheader("Sources de données et rafraîchissement")
    st.markdown(
        """
| Source | Type | Clé requise | Rafraîchissement |
|---|---|---|---|
| Import CSV | Fichier utilisateur | Non | Re-téléverser le fichier |
| Jeu d'exemple `data/sample_races.csv` | Embarqué | Non | Éditer le fichier dans le dépôt |
| Données synthétiques | Générées (Plackett-Luce) | Non | Graine / nombre de courses |
| Programme, partants **et cotes probable/directe** PMU | Web public **non officiel** (`online.turfinfo.api.pmu.fr`) | Non | Bouton « Actualiser », cache 2–10 min |
| Historique collecté (`scripts/build_history.py`) | Web public, incrémental, committé en CSV | Non | Automatique (GitHub Action quotidienne) |
| APIs tierces (The Racing API…) | Payant / documenté | Oui (`st.secrets`) | Non intégré par défaut |

La source PMU est consultée avec un User-Agent identifiable, un délai entre requêtes et un
cache ; elle peut cesser de fonctionner sans préavis, auquel cas l'application bascule sur le CSV.
Consultez les conditions d'utilisation du site avant tout usage intensif.
"""
    )
    st.subheader("Format CSV attendu")
    st.code(
        "race_id,race_date,track,discipline,distance_m,going,horse,jockey,trainer,draw,weight_kg,age,"
        "days_since_last_run,musique,career_starts,career_wins,career_places,earnings,odds,finish_position\n"
        "2024-05-12_R1_C3,2024-05-12,ParisLongchamp,plat,1600,bon,Golden Arrow,M. Guyon,A. Fabre,3,57.0,4,21,"
        "1p 2p 1p 3p 4p 1p,12,5,9,186000,2.8,1",
        language="text",
    )
    st.caption(
        "`finish_position` n'est nécessaire que pour l'historique d'entraînement/backtest. "
        "Colonnes `odds_probable`/`odds_direct` optionnelles (mouvement de cote). "
        "Séparateur `,` ou `;`, encodage UTF-8 ou Latin-1."
    )
    render_disclaimer(expanded=True)


# --------------------------------------------------------------------------- #
# Programme principal
# --------------------------------------------------------------------------- #
def main() -> None:
    st.sidebar.title("🐎 HorseProba")
    st.sidebar.caption(f"v{__version__} · ensemble logit + boosting, auto-appris")

    history, history_key = sidebar_history()
    choice = sidebar_model_choice(history_available=not history.empty)
    sidebar_footer()

    st.title("🐎 Pronostics hippiques probabilistes")
    st.markdown(
        "Classement des partants par **probabilité de victoire et de placé**, combinant un modèle "
        "logit conditionnel (Bradley-Terry généralisé) et un modèle de gradient boosting, sur cotes, "
        "forme, jockey, entraîneur, corde, poids, aptitude terrain/distance et mouvement de cote. "
        "Les probabilités sont des **estimations incertaines**, pas des certitudes."
    )

    model: AnyModel
    try:
        if choice.mode == "production":
            prod = cached_production_model(_mtime(registry.production_dir(REGISTRY_DIR) / "meta.json"))
            if prod is None:
                st.warning("Aucun modèle de production disponible : repli sur les coefficients a priori.")
                model = ConditionalLogitModel()
            else:
                model = prod
        elif choice.mode == "train_ensemble":
            with st.spinner("Estimation de l'ensemble (logit + boosting)…"):
                model = fit_ensemble_cached(history_key, choice.l2, choice.gbm_weight, history)
        elif choice.mode == "train_logit":
            with st.spinner("Estimation du logit conditionnel…"):
                model = fit_logit_cached(history_key, choice.l2, history)
        else:
            model = ConditionalLogitModel()
    except Exception as exc:  # noqa: BLE001
        st.error(f"L'entraînement a échoué ({exc}). Utilisation des coefficients a priori.")
        model = ConditionalLogitModel()

    t1, t2, t3, t4 = st.tabs(["🎯 Pronostic", "📈 Modèle & backtest", "🧠 Auto-apprentissage", "📚 Méthode & données"])
    with t1:
        tab_prediction(model)
    with t2:
        tab_model(model, history, history_key, choice.l2, choice.gbm_weight)
    with t3:
        tab_learning()
    with t4:
        tab_method()


main()
