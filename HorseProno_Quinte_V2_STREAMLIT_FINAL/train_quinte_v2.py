"""Entraîne et backteste HorseProno Quinté V2 en walk-forward strict.

Usage:
  python train_quinte_v2.py \
    --history validated_history.csv \
    --supports quinte_supports_99.csv \
    --output-dir models/quinte_v2
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from quinte_candidate import prepare
from quinte_v2 import add_relative_features

EXCLUDED_SUPPORTS = {("2026-08-30", 1, 3): "Ex aequo dans l'arrivée validée"}
MIN_TRAIN_SUPPORTS = 40
BASE_C = 0.005
RANKER_PARAMS = dict(
    objective="lambdarank",
    metric="ndcg",
    n_estimators=100,
    learning_rate=0.03,
    num_leaves=7,
    max_depth=3,
    min_child_samples=30,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=2.0,
    reg_alpha=0.2,
    random_state=42,
    verbosity=-1,
)


def support_race_id(row: pd.Series) -> str:
    return f"R{int(row.meeting_number)}C{int(row.race_number)}_{pd.Timestamp(row.race_date).date().isoformat()}"


def load_dataset(history_path: str, supports_path: str):
    h = pd.read_csv(history_path)
    h["race_date"] = pd.to_datetime(h["race_date"], errors="coerce").dt.normalize()
    s = pd.read_csv(supports_path)
    s["race_date"] = pd.to_datetime(s["race_date"], errors="coerce").dt.normalize()
    s = s.sort_values(["race_date", "meeting_number", "race_number"]).reset_index(drop=True)
    s["race_id"] = s.apply(support_race_id, axis=1)
    s["is_evaluable"] = True
    s["exclusion_reason"] = ""
    for (day, r, c), reason in EXCLUDED_SUPPORTS.items():
        m = (
            s["race_date"].eq(pd.Timestamp(day))
            & s["meeting_number"].eq(r)
            & s["race_number"].eq(c)
        )
        s.loc[m, "is_evaluable"] = False
        s.loc[m, "exclusion_reason"] = reason

    available = set(h["race_id"].dropna().astype(str))
    missing = s.loc[s["is_evaluable"] & ~s["race_id"].isin(available), "race_id"].tolist()
    if missing:
        raise RuntimeError(f"Supports évaluables absents de l'historique: {missing}")

    # Validation générique de l'arrivée 1..5 ; l'ex aequo connu reste exclu explicitement.
    for idx, row in s.iterrows():
        if not bool(row["is_evaluable"]):
            continue
        g = h[h["race_id"].eq(row["race_id"])]
        pos = pd.to_numeric(g["finish_position"], errors="coerce")
        top5 = sorted(pos[pos.between(1, 5)].astype(int).tolist())
        if top5 != [1, 2, 3, 4, 5]:
            s.loc[idx, "is_evaluable"] = False
            s.loc[idx, "exclusion_reason"] = "Arrivée Top5 incomplète ou non unique"

    ev = s[s["is_evaluable"]].copy().reset_index(drop=True)
    q = pd.concat([h[h["race_id"].eq(rid)] for rid in ev["race_id"]], ignore_index=True)
    # Conserve l'ordre chronologique des supports.
    order_map = {rid: i for i, rid in enumerate(ev["race_id"].tolist())}
    q["support_order"] = q["race_id"].map(order_map).astype(int)
    q = q.sort_values(["support_order", "horse_number"]).reset_index(drop=True)

    base = prepare(h, q).reset_index(drop=True)
    full = add_relative_features(base, q)
    pos = pd.to_numeric(q["finish_position"], errors="coerce")
    y_top5 = pos.between(1, 5).astype(int).to_numpy()
    relevance = np.asarray(np.where(pos.between(1, 5), 6 - pos.fillna(99), 0), dtype=int)
    return h, s, ev, q, base, full, y_top5, relevance


def fit_logit(X: pd.DataFrame, y: np.ndarray):
    scaler = StandardScaler().fit(X)
    model = LogisticRegression(C=BASE_C, max_iter=2000, solver="lbfgs", random_state=42)
    model.fit(scaler.transform(X), y)
    return scaler, model


def fit_ranker(X: pd.DataFrame, yrel: np.ndarray, groups: np.ndarray):
    model = lgb.LGBMRanker(**RANKER_PARAMS)
    model.fit(X, yrel, group=groups.tolist())
    return model


def dcg_at_k(rels, k):
    rels = np.asarray(rels, dtype=float)[:k]
    if len(rels) == 0:
        return 0.0
    return float(np.sum((2**rels - 1) / np.log2(np.arange(2, len(rels) + 2))))


def ndcg(order_numbers, rel_by_number, k):
    rels = [rel_by_number.get(int(n), 0.0) for n in order_numbers[:k]]
    ideal = sorted(rel_by_number.values(), reverse=True)
    denom = dcg_at_k(ideal, k)
    return dcg_at_k(rels, k) / denom if denom > 0 else 0.0


def metric_row(g: pd.DataFrame, order_numbers: list[int], method: str, meta: dict):
    pos = pd.to_numeric(g["finish_position"], errors="coerce")
    actual_top5 = set(g.loc[pos.between(1, 5), "horse_number"].astype(int))
    winner = int(g.loc[pos.eq(1), "horse_number"].iloc[0])
    top5 = order_numbers[:5]
    top7 = order_numbers[:7]
    hits5 = len(actual_top5.intersection(top5))
    hits7 = len(actual_top5.intersection(top7))
    rel_by_number = {}
    for _, r in g.iterrows():
        p = pd.to_numeric(pd.Series([r["finish_position"]]), errors="coerce").iloc[0]
        rel_by_number[int(r["horse_number"])] = float(6 - p) if pd.notna(p) and 1 <= p <= 5 else 0.0
    return {
        **meta,
        "method": method,
        "hits_top5_in_top5": hits5,
        "hits_top5_in_top7": hits7,
        "four_of_five_top7": int(hits7 >= 4),
        "five_of_five_top7": int(hits7 == 5),
        "winner_at1": int(order_numbers[0] == winner),
        "winner_in5": int(winner in top5),
        "winner_in7": int(winner in top7),
        "ndcg5": ndcg(order_numbers, rel_by_number, 5),
        "ndcg7": ndcg(order_numbers, rel_by_number, 7),
        "top7": "-".join(map(str, top7)),
    }


def aggregate(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"races": 0}
    return {
        "races": int(len(df)),
        "avg_hits_top5_in_top5": float(df.hits_top5_in_top5.mean()),
        "avg_hits_top5_in_top7": float(df.hits_top5_in_top7.mean()),
        "four_of_five_top7_rate": float(df.four_of_five_top7.mean()),
        "five_of_five_top7_rate": float(df.five_of_five_top7.mean()),
        "winner_at1_rate": float(df.winner_at1.mean()),
        "winner_in5_rate": float(df.winner_in5.mean()),
        "winner_in7_rate": float(df.winner_in7.mean()),
        "ndcg5": float(df.ndcg5.mean()),
        "ndcg7": float(df.ndcg7.mean()),
    }


def paired_bootstrap(v2: pd.DataFrame, market: pd.DataFrame, samples=5000, seed=42):
    v2 = v2.sort_values("race_id").reset_index(drop=True)
    market = market.sort_values("race_id").reset_index(drop=True)
    if not v2.race_id.equals(market.race_id):
        raise RuntimeError("Courses non alignées pour bootstrap")
    rng = np.random.default_rng(seed)
    n = len(v2)
    metrics = {
        "avg_hits_top5_in_top7": (v2.hits_top5_in_top7.to_numpy(float), market.hits_top5_in_top7.to_numpy(float)),
        "four_of_five_top7_rate": (v2.four_of_five_top7.to_numpy(float), market.four_of_five_top7.to_numpy(float)),
        "five_of_five_top7_rate": (v2.five_of_five_top7.to_numpy(float), market.five_of_five_top7.to_numpy(float)),
        "ndcg7": (v2.ndcg7.to_numpy(float), market.ndcg7.to_numpy(float)),
    }
    out = {}
    for name, (a, b) in metrics.items():
        diffs = np.empty(samples)
        for i in range(samples):
            idx = rng.integers(0, n, n)
            diffs[i] = np.mean(a[idx] - b[idx])
        out[name] = {
            "observed_difference": float(np.mean(a - b)),
            "ci95_low": float(np.quantile(diffs, 0.025)),
            "ci95_high": float(np.quantile(diffs, 0.975)),
            "probability_difference_positive": float(np.mean(diffs > 0)),
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", required=True)
    ap.add_argument("--supports", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    h, all_supports, supports, q, base, full, y, rel = load_dataset(args.history, args.supports)
    base_features = list(base.columns)
    ranker_features = list(full.columns)

    rows = []
    oof_raw = []
    oof_y = []

    for current_order in range(MIN_TRAIN_SUPPORTS, len(supports)):
        train_mask = q["support_order"].lt(current_order).to_numpy()
        test_mask = q["support_order"].eq(current_order).to_numpy()
        if not test_mask.any():
            continue

        scaler, logit = fit_logit(base.loc[train_mask, base_features], y[train_mask])
        raw_logit = logit.predict_proba(scaler.transform(base.loc[test_mask, base_features]))[:, 1]
        oof_raw.extend(raw_logit.tolist())
        oof_y.extend(y[test_mask].tolist())

        train_groups = q.loc[train_mask].groupby("support_order", sort=True).size().to_numpy()
        ranker = fit_ranker(full.loc[train_mask, ranker_features], rel[train_mask], train_groups)
        rank_score = ranker.predict(full.loc[test_mask, ranker_features])

        g = q.loc[test_mask].copy().reset_index(drop=True)
        btest = base.loc[test_mask].reset_index(drop=True)
        g["market_score"] = btest["market_prob"].to_numpy()
        g["logit_score"] = raw_logit
        g["ranker_score"] = rank_score

        market_order = g.sort_values(["market_score", "horse_number"], ascending=[False, True]).horse_number.astype(int).tolist()
        logit_order = g.sort_values(["logit_score", "horse_number"], ascending=[False, True]).horse_number.astype(int).tolist()
        ranker_order = g.sort_values(["ranker_score", "horse_number"], ascending=[False, True]).horse_number.astype(int).tolist()

        core = market_order[:6]
        challengers = [n for n in logit_order if n not in core][:1]
        remaining = [n for n in logit_order if n not in core and n not in challengers]
        v2_order = core + challengers + remaining

        srow = supports.iloc[current_order]
        meta = {
            "race_id": str(srow.race_id),
            "race_date": pd.Timestamp(srow.race_date).date().isoformat(),
            "meeting_number": int(srow.meeting_number),
            "race_number": int(srow.race_number),
            "hippodrome": str(srow.hippodrome),
            "discipline": str(srow.discipline),
            "field_size": int(len(g)),
            "support_order": int(current_order),
            "common_test": bool(srow.get("common_test", False)),
        }
        for method, order in [
            ("market", market_order),
            ("logistic_v1_expanding", logit_order),
            ("lgbm_ranker", ranker_order),
            ("v2_market6_plus_logit1", v2_order),
        ]:
            rows.append(metric_row(g, order, method, meta))

    wf = pd.DataFrame(rows)
    wf.to_csv(outdir / "quinte_v2_walkforward.csv", index=False)

    # Calibration hors échantillon de la probabilité Top5 logistique.
    raw = np.clip(np.asarray(oof_raw, dtype=float), 1e-6, 1 - 1e-6)
    y_oof = np.asarray(oof_y, dtype=int)
    raw_logit_feature = np.log(raw / (1 - raw)).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e6, max_iter=2000, solver="lbfgs", random_state=42)
    calibrator.fit(raw_logit_feature, y_oof)

    # Modèles finaux sur les 98 supports évaluables.
    final_scaler, final_logit = fit_logit(base[base_features], y)
    groups_all = q.groupby("support_order", sort=True).size().to_numpy()
    final_ranker = fit_ranker(full[ranker_features], rel, groups_all)
    final_ranker.booster_.save_model(str(outdir / "quinte_v2_ranker.txt"))

    # Agrégats et sous-ensembles.
    summary = {
        "walk_forward_start_after_supports": MIN_TRAIN_SUPPORTS,
        "walk_forward_races": int(len(supports) - MIN_TRAIN_SUPPORTS),
        "all": {},
        "holdout_last_28": {},
        "common_test": {},
        "by_discipline_v2": {},
    }
    for method in wf.method.unique():
        summary["all"][method] = aggregate(wf[wf.method.eq(method)])
        tail_ids = supports.iloc[-28:].race_id.astype(str).tolist()
        summary["holdout_last_28"][method] = aggregate(wf[wf.method.eq(method) & wf.race_id.isin(tail_ids)])
        summary["common_test"][method] = aggregate(wf[wf.method.eq(method) & wf.common_test])
    v2wf = wf[wf.method.eq("v2_market6_plus_logit1")]
    for discipline, g in v2wf.groupby("discipline"):
        summary["by_discipline_v2"][str(discipline)] = aggregate(g)
    summary["paired_bootstrap_vs_market"] = paired_bootstrap(
        wf[wf.method.eq("v2_market6_plus_logit1")],
        wf[wf.method.eq("market")],
    )
    holdout_ids = supports.iloc[-28:].race_id.astype(str).tolist()
    summary["paired_bootstrap_holdout_last_28"] = paired_bootstrap(
        wf[wf.method.eq("v2_market6_plus_logit1") & wf.race_id.isin(holdout_ids)],
        wf[wf.method.eq("market") & wf.race_id.isin(holdout_ids)],
        samples=10000,
        seed=43,
    )
    summary["oof_calibration"] = {
        "rows": int(len(y_oof)),
        "prevalence": float(y_oof.mean()),
        "raw_probability_mean": float(raw.mean()),
        "platt_coef": float(calibrator.coef_[0, 0]),
        "platt_intercept": float(calibrator.intercept_[0]),
    }

    artifact = {
        "model_name": "HorseProno_Quinte_V2_market6_top5_challenger",
        "status": "production_candidate_walk_forward",
        "objective": "Maximiser la couverture de l'arrivée Top5 dans un shortlist Top7 avec validation walk-forward",
        "target": "finish_position in 1..5 (Top5 binaire) pour le modèle logistique; pertinence ordinale 5..1 pour le challenger LambdaMART offline",
        "package_version": "2.0.0",
        "training_population": "98 supports Quinté+ officiels évaluables; historique toutes courses utilisé uniquement pour les variables antérieures",
        "training_supports": int(len(supports)),
        "official_supports_source_rows": int(len(all_supports)),
        "training_end": pd.Timestamp(supports.race_date.max()).date().isoformat(),
        "excluded_supports": all_supports.loc[~all_supports.is_evaluable, ["race_date", "meeting_number", "race_number", "exclusion_reason"]].assign(
            race_date=lambda d: d.race_date.dt.strftime("%Y-%m-%d")
        ).to_dict(orient="records"),
        "strategy": {
            "market_core_size": 6,
            "model_challengers": 1,
            "ranking_note": "Le marché normalisé conserve les six premiers; le modèle logistique Top5 calibré sélectionne un challenger hors noyau.",
            "selection_protocol": "Recette choisie sur les 30 premières courses walk-forward puis contrôlée sur les 28 dernières laissées de côté.",
        },
        "top5_logistic": {
            "features": base_features,
            "C": BASE_C,
            "mean": final_scaler.mean_.tolist(),
            "scale": final_scaler.scale_.tolist(),
            "coef": final_logit.coef_[0].tolist(),
            "intercept": float(final_logit.intercept_[0]),
            "platt_calibrator": {
                "input": "logit(raw_top5_probability)",
                "coef": float(calibrator.coef_[0, 0]),
                "intercept": float(calibrator.intercept_[0]),
                "fit_source": "strict walk-forward OOF predictions",
            },
        },
        "ranker": {
            "role": "offline_challenger_only",
            "algorithm": "LightGBM LambdaMART",
            "objective": "lambdarank",
            "target_relevance": "1er=5, 2e=4, 3e=3, 4e=2, 5e=1, autres=0",
            "features": ranker_features,
            "params": RANKER_PARAMS,
            "model_file": "quinte_v2_ranker.txt",
        },
        "backtest": summary,
        "limitations": [
            "Le nombre de supports Quinté reste limité (98 évaluables).",
            "Sur le holdout indépendant de 28 courses, les intervalles bootstrap restent larges; le gain de shortlist n'est pas statistiquement certain à 95%.",
            "Le ROI réel n'est pas validable sans cotes horodatées au moment de la décision.",
            "Le segment PLAT reste le principal axe d'amélioration.",
        ],
    }
    (outdir / "quinte_v2_artifact.json").write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    (outdir / "quinte_v2_backtest_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps({
        "supports_officiels": int(len(all_supports)),
        "supports_evaluables": int(len(supports)),
        "partants": int(len(q)),
        "walk_forward_races": summary["walk_forward_races"],
        "market": summary["all"]["market"],
        "v2": summary["all"]["v2_market6_plus_logit1"],
        "holdout_market": summary["holdout_last_28"]["market"],
        "holdout_v2": summary["holdout_last_28"]["v2_market6_plus_logit1"],
        "calibration": summary["oof_calibration"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
