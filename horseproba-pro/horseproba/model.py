"""
Modèle probabiliste de pronostic : LOGIT CONDITIONNEL (Bradley-Terry / Plackett-Luce généralisé)

Pourquoi ce modèle ?
====================
1. Une course est un problème de CHOIX DISCRET : exactement un cheval gagne parmi
   n partants. Une régression logistique classique (un cheval = une observation
   indépendante « gagne / ne gagne pas ») ignore cette contrainte et produit des
   probabilités qui ne somment pas à 1 dans la course. Le logit conditionnel
   (McFadden, 1974) modélise directement P(i gagne | partants de la course) :

        P(i gagne) = exp(β·x_i) / Σ_j exp(β·x_j)      (softmax intra-course)

   Ce modèle est mathématiquement identique à un modèle de Bradley-Terry
   (comparaison par paires) étendu à n compétiteurs : c'est le modèle de Luce
   (1959) / Plackett (1975) pour le 1er rang. C'est LE modèle de référence de la
   littérature sur les paris hippiques (Bolton & Chapman 1986 « Searching for
   positive returns at the track » ; Benter 1994, système de Hong Kong).

2. La « force » β·x_i est une fonction linéaire de features interprétables :
   chaque coefficient dit combien un écart-type de la variable (au sein de la
   course) modifie la force du cheval sur l'échelle logit. On obtient donc une
   EXPLICATION naturelle du classement (contributions par variable).

3. Régularisation L2 : le nombre de courses disponibles est souvent faible ;
   la pénalité ridge stabilise les coefficients et évite le sur-apprentissage
   sur des variables très corrélées (forme récente ↔ cote, par exemple).

4. Combinaison avec le marché : la cote est incluse comme feature. Le marché est
   un excellent prédicteur (sagesse des foules) ; le modèle apprend combien il
   faut lui faire confiance et quelles informations il sous-pondère. Sans
   historique, on retombe sur des coefficients « a priori » raisonnables
   (PRIOR_COEFS) : l'application reste utilisable même sans données d'entraînement.

Extension au placé
==================
La probabilité d'être placé (top 3) est calculée par Plackett-Luce : on énumère
les arrivées possibles jusqu'au 3e rang (exact pour les petits pelotons), ou par
simulation Monte-Carlo quand n est grand. Ce n'est pas un second modèle : ce sont
les mêmes forces latentes qui déterminent tous les rangs, cohérence garantie.

Estimation
==========
La log-vraisemblance conditionnelle est concave : on la maximise avec L-BFGS
(scipy). Le gradient est analytique : ∂ℓ/∂β = Σ_courses (x_gagnant − Σ_i p_i x_i).
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .features import FEATURE_COLUMNS, FEATURE_LABELS, build_features, within_race_standardize

# Séparateur utilisé pour encoder les clés composées (tuples) dans le JSON du
# registre de modèle, puisque JSON n'autorise que des clés de type chaîne.
_KEY_SEP = "\x1f"


def _encode_tuple_dict(d: Dict[Tuple[str, str], float]) -> Dict[str, float]:
    return {f"{a}{_KEY_SEP}{b}": v for (a, b), v in d.items()}


def _decode_tuple_dict(d: Dict[str, float]) -> Dict[Tuple[str, str], float]:
    out: Dict[Tuple[str, str], float] = {}
    for k, v in d.items():
        a, _, b = k.partition(_KEY_SEP)
        out[(a, b)] = v
    return out


# Coefficients a priori (échelle : features centrées/réduites) utilisés lorsqu'aucun
# historique n'est disponible. Ils traduisent un consensus raisonnable de la
# littérature : le marché domine, puis la forme, puis les hommes (jockey/entraîneur),
# puis les aptitudes contextuelles (terrain/distance/piste), volontairement modestes
# a priori car ce sont les variables les plus fines / les plus sujettes au bruit tant
# qu'elles n'ont pas été calibrées sur un historique réel.
PRIOR_COEFS: Dict[str, float] = {
    "log_odds_implied": 1.00,
    "market_drift": 0.20,
    "form_score": 0.35,
    "form_win_rate": 0.10,
    "form_place_rate": 0.10,
    "dq_rate": -0.15,
    "career_win_rate": 0.15,
    "career_place_rate": 0.10,
    "log_earnings_per_start": 0.10,
    "freshness": 0.08,
    "jockey_win_rate": 0.15,
    "trainer_win_rate": 0.12,
    "jockey_horse_synergy": 0.08,
    "going_aptitude": 0.10,
    "distance_aptitude": 0.10,
    "track_aptitude": 0.08,
    "draw_rel": -0.08,
    "weight_rel": -0.06,
    "age_rel": -0.02,
}


@dataclass
class FitResult:
    coefs: Dict[str, float]
    n_races: int
    n_runners: int
    log_likelihood: float
    baseline_log_likelihood: float  # modèle uniforme 1/n
    converged: bool
    message: str = ""

    @property
    def pseudo_r2(self) -> float:
        """McFadden R² : 1 − ℓ(modèle)/ℓ(uniforme). 0.1–0.3 est déjà très bon pour ce type de problème."""
        if self.baseline_log_likelihood == 0:
            return 0.0
        return float(1.0 - self.log_likelihood / self.baseline_log_likelihood)


@dataclass
class ConditionalLogitModel:
    """
    Modèle logit conditionnel régularisé (ridge).

    Attributs
    ---------
    features   : liste des colonnes utilisées
    coefs      : coefficients β (dict)
    l2         : force de la régularisation L2
    prior      : coefficients a priori vers lesquels la pénalité attire β
                 (ridge centré sur le prior, i.e. estimation MAP gaussienne)
    """

    features: List[str] = field(default_factory=lambda: list(FEATURE_COLUMNS))
    coefs: Dict[str, float] = field(default_factory=lambda: dict(PRIOR_COEFS))
    l2: float = 1.0
    prior: Dict[str, float] = field(default_factory=lambda: dict(PRIOR_COEFS))
    fit_result: Optional[FitResult] = None
    jockey_rates: Dict[str, float] = field(default_factory=dict)
    trainer_rates: Dict[str, float] = field(default_factory=dict)
    combo_rates: Dict[Tuple[str, str], float] = field(default_factory=dict)
    going_rates: Dict[Tuple[str, str], float] = field(default_factory=dict)
    distance_rates: Dict[Tuple[str, str], float] = field(default_factory=dict)
    track_rates: Dict[Tuple[str, str], float] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    # Préparation
    # ------------------------------------------------------------------ #
    def _design(self, runners: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, List[np.ndarray]]:
        """Construit features standardisées, matrice X et index des groupes (courses)."""
        feats = build_features(
            runners,
            jockey_rates=self.jockey_rates,
            trainer_rates=self.trainer_rates,
            combo_rates=self.combo_rates,
            going_rates=self.going_rates,
            distance_rates=self.distance_rates,
            track_rates=self.track_rates,
        )
        feats = within_race_standardize(feats, self.features)
        X = feats[self.features].to_numpy(dtype=float)
        groups = [np.asarray(idx) for _, idx in feats.groupby("race_id", sort=False).indices.items()]
        return feats, X, groups

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    # ------------------------------------------------------------------ #
    # Estimation
    # ------------------------------------------------------------------ #
    def fit(self, history: pd.DataFrame) -> FitResult:
        """
        Estime β par maximum de vraisemblance conditionnelle pénalisée sur un
        historique comportant `finish_position` (1 = gagnant).

        Les courses sans gagnant identifiable (ex. dead-heat non géré, données
        manquantes) sont ignorées. Les taux jockey/entraîneur sont estimés sur ce
        même historique (attention aux fuites en backtest : utiliser `evaluate.backtest`
        qui recalcule tout sur la fenêtre d'entraînement uniquement).
        """
        from .features import compute_entity_rates, compute_grouped_rates, _norm_key, going_bucket, distance_bucket

        hist = history.copy()
        hist["finish_position"] = pd.to_numeric(hist["finish_position"], errors="coerce")
        self.jockey_rates = compute_entity_rates(hist, "jockey")
        self.trainer_rates = compute_entity_rates(hist, "trainer")
        if "jockey" in hist.columns and "horse" in hist.columns:
            self.combo_rates = compute_grouped_rates(
                hist, lambda h: list(zip(h["jockey"].map(_norm_key), h["horse"].map(_norm_key))), k=30.0
            )
        if "going" in hist.columns:
            self.going_rates = compute_grouped_rates(
                hist, lambda h: list(zip(h["horse"].map(_norm_key), h["going"].map(going_bucket))), k=25.0
            )
        if "distance_m" in hist.columns:
            self.distance_rates = compute_grouped_rates(
                hist, lambda h: list(zip(h["horse"].map(_norm_key), h["distance_m"].map(distance_bucket))), k=25.0
            )
        if "track" in hist.columns:
            self.track_rates = compute_grouped_rates(
                hist, lambda h: list(zip(h["horse"].map(_norm_key), h["track"].map(_norm_key))), k=25.0
            )

        feats, X, groups = self._design(hist)
        y = feats["finish_position"].to_numpy()

        # Ne garder que les courses avec exactement un gagnant
        valid_groups = [g for g in groups if np.sum(y[g] == 1) == 1]
        if len(valid_groups) < 5:
            res = FitResult(dict(self.prior), len(valid_groups), int(sum(len(g) for g in valid_groups)),
                            0.0, 0.0, False, "Historique insuffisant (< 5 courses) : coefficients a priori conservés.")
            self.coefs = dict(self.prior)
            self.fit_result = res
            return res

        winners = [int(g[np.argmax(y[g] == 1)]) for g in valid_groups]
        beta0 = np.array([self.prior.get(f, 0.0) for f in self.features])
        lam = self.l2

        def neg_ll_and_grad(beta: np.ndarray):
            ll = 0.0
            grad = np.zeros_like(beta)
            for g, w in zip(valid_groups, winners):
                z = X[g] @ beta
                p = self._softmax(z)
                ll += np.log(max(p[np.where(g == w)[0][0]], 1e-12))
                grad += X[w] - p @ X[g]
            # pénalité ridge centrée sur le prior (MAP gaussien)
            diff = beta - beta0
            ll -= 0.5 * lam * diff @ diff
            grad -= lam * diff
            return -ll, -grad

        opt = minimize(neg_ll_and_grad, beta0, jac=True, method="L-BFGS-B", options={"maxiter": 500})
        beta = opt.x
        self.coefs = {f: float(b) for f, b in zip(self.features, beta)}

        # Log-vraisemblance non pénalisée et baseline
        ll = 0.0
        ll0 = 0.0
        for g, w in zip(valid_groups, winners):
            p = self._softmax(X[g] @ beta)
            ll += np.log(max(p[np.where(g == w)[0][0]], 1e-12))
            ll0 += np.log(1.0 / len(g))
        self.fit_result = FitResult(
            coefs=dict(self.coefs),
            n_races=len(valid_groups),
            n_runners=int(sum(len(g) for g in valid_groups)),
            log_likelihood=float(ll),
            baseline_log_likelihood=float(ll0),
            converged=bool(opt.success),
            message=str(opt.message),
        )
        return self.fit_result

    # ------------------------------------------------------------------ #
    # Prédiction
    # ------------------------------------------------------------------ #
    def predict(self, runners: pd.DataFrame, n_places: int = 3, mc_samples: int = 20000, seed: int = 7) -> pd.DataFrame:
        """
        Renvoie `runners` enrichi de :
            strength        force latente β·x (échelle logit)
            p_win           probabilité de victoire (softmax intra-course)
            p_place         probabilité de finir dans les `n_places` premiers (Plackett-Luce)
            fair_odds       cote « juste » = 1 / p_win
            market_p        probabilité implicite du marché (si cotes disponibles)
            value           p_win − market_p (positif = le modèle voit de la valeur)
            rank            classement par p_win
            contrib_<feat>  contribution de chaque feature à la force (β_k · x_k)
        """
        feats, X, groups = self._design(runners)
        beta = np.array([self.coefs.get(f, 0.0) for f in self.features])
        out = feats.copy()
        out["strength"] = X @ beta
        out["p_win"] = 0.0
        out["p_place"] = 0.0
        for k, f in enumerate(self.features):
            out[f"contrib_{f}"] = X[:, k] * beta[k]

        rng = np.random.default_rng(seed)
        for g in groups:
            s = out["strength"].to_numpy()[g]
            p = self._softmax(s)
            out.iloc[g, out.columns.get_loc("p_win")] = p
            out.iloc[g, out.columns.get_loc("p_place")] = plackett_luce_topk(s, min(n_places, len(g)), rng, mc_samples)

        out["fair_odds"] = 1.0 / out["p_win"].clip(lower=1e-6)
        if "odds" in out.columns:
            odds = pd.to_numeric(out["odds"], errors="coerce")
            inv = (1.0 / odds.where(odds > 1)).fillna(0.0)
            # normalisation intra-course des probabilités implicites
            tot = inv.groupby(out["race_id"]).transform("sum").replace(0, np.nan)
            out["market_p"] = (inv / tot).fillna(np.nan)
        else:
            out["market_p"] = np.nan
        out["value"] = out["p_win"] - out["market_p"]
        out["rank"] = out.groupby("race_id")["p_win"].rank(ascending=False, method="first").astype(int)
        return out.sort_values(["race_id", "rank"]).reset_index(drop=True)

    # ------------------------------------------------------------------ #
    # Explications
    # ------------------------------------------------------------------ #
    def explain(self, predicted: pd.DataFrame, horse_index: int, top_k: int = 4) -> List[str]:
        """
        Produit des phrases lisibles justifiant la force d'un cheval : les `top_k`
        contributions (positives ou négatives) les plus importantes.
        """
        row = predicted.iloc[horse_index]
        contribs = {f: row.get(f"contrib_{f}", 0.0) for f in self.features}
        ordered = sorted(contribs.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
        lines = []
        for f, c in ordered:
            if abs(c) < 0.02:
                continue
            label = FEATURE_LABELS.get(f, f)
            direction = "favorable" if c > 0 else "défavorable"
            lines.append(f"{label} : {direction} ({c:+.2f} sur l'échelle logit)")
        if not lines:
            lines.append("Profil moyen : aucune variable ne le distingue nettement du lot.")
        return lines

    # ------------------------------------------------------------------ #
    # Persistance
    # ------------------------------------------------------------------ #
    def to_json(self) -> str:
        return json.dumps(
            {
                "features": self.features,
                "coefs": self.coefs,
                "l2": self.l2,
                "jockey_rates": self.jockey_rates,
                "trainer_rates": self.trainer_rates,
                "combo_rates": _encode_tuple_dict(self.combo_rates),
                "going_rates": _encode_tuple_dict(self.going_rates),
                "distance_rates": _encode_tuple_dict(self.distance_rates),
                "track_rates": _encode_tuple_dict(self.track_rates),
            },
            ensure_ascii=False, indent=2,
        )

    @classmethod
    def from_json(cls, text: str) -> "ConditionalLogitModel":
        d = json.loads(text)
        m = cls(features=d["features"], coefs=d["coefs"], l2=d.get("l2", 1.0))
        m.jockey_rates = d.get("jockey_rates", {})
        m.trainer_rates = d.get("trainer_rates", {})
        m.combo_rates = _decode_tuple_dict(d.get("combo_rates", {}))
        m.going_rates = _decode_tuple_dict(d.get("going_rates", {}))
        m.distance_rates = _decode_tuple_dict(d.get("distance_rates", {}))
        m.track_rates = _decode_tuple_dict(d.get("track_rates", {}))
        return m


# ---------------------------------------------------------------------- #
# Plackett-Luce : probabilité de finir dans les k premiers
# ---------------------------------------------------------------------- #

def plackett_luce_topk(strength: np.ndarray, k: int, rng: Optional[np.random.Generator] = None, mc_samples: int = 20000) -> np.ndarray:
    """
    P(cheval i finit dans les k premiers) sous le modèle de Plackett-Luce avec
    forces exp(strength).

    - Énumération exacte des k-permutations si n·(n−1)·…·(n−k+1) ≤ 4000
      (par ex. n ≤ 16 pour k = 3) ;
    - sinon, simulation Monte-Carlo (échantillonnage séquentiel sans remise).
    """
    n = len(strength)
    if k >= n:
        return np.ones(n)
    w = np.exp(strength - strength.max())
    total = w.sum()

    n_perms = 1
    for j in range(k):
        n_perms *= n - j
    if n_perms <= 4000:
        probs = np.zeros(n)
        for perm in itertools.permutations(range(n), k):
            p = 1.0
            remaining = total
            for idx in perm:
                p *= w[idx] / remaining
                remaining -= w[idx]
            for idx in perm:
                probs[idx] += p
        return probs

    rng = rng or np.random.default_rng(0)
    counts = np.zeros(n)
    p_full = w / total
    for _ in range(mc_samples):
        # Astuce de Gumbel : argsort de log(w) + Gumbel ⇔ tirage Plackett-Luce
        gumbel = rng.gumbel(size=n)
        order = np.argsort(-(np.log(p_full) + gumbel))[:k]
        counts[order] += 1
    return counts / mc_samples


# ---------------------------------------------------------------------- #
# Détection d'outsiders à potentiel
# ---------------------------------------------------------------------- #

def flag_outsiders(
    predicted: pd.DataFrame,
    min_odds: float = 8.0,
    min_edge: float = 0.03,
    min_p_win: float = 0.03,
) -> pd.Series:
    """
    Repère les « outsiders à potentiel » dans une sortie de `.predict(...)` :
    des partants PEU joués par le marché (grosse cote) mais que le modèle juge
    sensiblement sous-évalués par rapport à ce marché.

    Un partant est marqué outsider si les 3 conditions sont réunies :
      1. cote marché >= `min_odds`               → ce n'est pas un favori
      2. value = p_modèle − p_marché >= `min_edge` → écart significatif, pas du
         bruit d'arrondi (0.03 = 3 points de probabilité, un seuil volontairement
         conservateur)
      3. p_win modèle >= `min_p_win`               → le modèle lui donne une
         vraie chance de gagner, pas juste "un peu moins improbable"

    C'est un signal d'ATTENTION à examiner (via `.explain()`), pas une
    recommandation de pari : le modèle n'intègre ni la marge du PMU, ni une
    logique de bankroll, ni une garantie de calibration parfaite. Voir le
    README (section « Limites ») avant tout usage réel.
    """
    if predicted.empty:
        return pd.Series([], dtype=bool, index=predicted.index)
    odds = pd.to_numeric(predicted.get("odds", pd.Series(np.nan, index=predicted.index)), errors="coerce")
    value = pd.to_numeric(predicted.get("value", pd.Series(0.0, index=predicted.index)), errors="coerce").fillna(0.0)
    p_win = pd.to_numeric(predicted.get("p_win", pd.Series(0.0, index=predicted.index)), errors="coerce").fillna(0.0)
    return ((odds >= min_odds) & (value >= min_edge) & (p_win >= min_p_win)).fillna(False)
