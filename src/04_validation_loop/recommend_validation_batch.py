#!/usr/bin/env python3
"""
Recommend a wet-lab batch that balances exploitation and exploration.

Default behavior:
- use the active iteration's `05_bo_optimization` candidate files as the
  primary exploitation pool
- supplement them with `03_optimization` candidate files for extra chemistry
  diversity in the exploration bucket
- learn blind-spot chemistry from the last completed validation stage
- filter out already tested formulations
- recommend a mixed batch with high-value exploitation candidates and
  high-uncertainty / under-modeled exploration candidates

This script centers the recommendation around `05`, because that optimizer
already encodes the exploration/exploitation tradeoff and support constraints.
It also borrows from `03` by default because calibration probes often benefit
from chemistry diversity that BO alone will not prioritize.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import sys
from dataclasses import dataclass
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
VALIDATION_PATH = os.path.join(PROJECT_ROOT, "data", "validation", "validation_results.csv")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from filter_tested_candidates import format_formulation  # noqa: E402
from update_model_weighted_prior import CompositeGP  # noqa: F401,E402


EXPERIMENT_ID_PATTERN = re.compile(r"(\d+)")


@dataclass
class StageModel:
    stage: int
    iteration_dir: str
    metadata: Dict
    feature_names: List[str]
    is_composite_model: bool


def round_or_none(value: float, digits: int = 4) -> Optional[float]:
    """Return a JSON-safe rounded number."""
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        return None
    return round(float(value), digits)


def parse_experiment_stage(experiment_id: str) -> Optional[int]:
    """Map EXP IDs to integer validation stages."""
    match = EXPERIMENT_ID_PATTERN.search(str(experiment_id))
    if not match:
        return None
    value = int(match.group(1))
    if value < 1000:
        return 0
    return value // 1000


def load_validation_df() -> pd.DataFrame:
    """Load measured wet-lab rows."""
    df = pd.read_csv(VALIDATION_PATH)
    df = df[df["viability_measured"].notna()].copy()
    df["stage"] = df["experiment_id"].map(parse_experiment_stage)
    return df


def load_active_metadata() -> Dict:
    """Load active model metadata from the root mirror."""
    with open(os.path.join(MODELS_DIR, "model_metadata.json"), "r") as handle:
        return json.load(handle)


def discover_stage_model(stage: int) -> StageModel:
    """Resolve one saved iteration directory by stage number."""
    candidates = []
    prefix = f"iteration_{stage}"
    for entry in os.listdir(MODELS_DIR):
        if not entry.startswith(prefix):
            continue
        metadata_path = os.path.join(MODELS_DIR, entry, "model_metadata.json")
        if not os.path.exists(metadata_path):
            continue
        with open(metadata_path, "r") as handle:
            metadata = json.load(handle)
        candidates.append((entry, metadata))

    if not candidates:
        raise FileNotFoundError(f"No saved model found for stage {stage}")

    candidates.sort(key=lambda item: item[1].get("updated_at", item[1].get("trained_at", "")))
    iteration_dir, metadata = candidates[-1]
    return StageModel(
        stage=stage,
        iteration_dir=iteration_dir,
        metadata=metadata,
        feature_names=list(metadata["feature_names"]),
        is_composite_model=bool(metadata.get("is_composite_model", False)),
    )


def load_stage_predictor(stage_model: StageModel):
    """Load one saved model."""
    model_dir = os.path.join(MODELS_DIR, stage_model.iteration_dir)
    if stage_model.is_composite_model:
        with open(os.path.join(model_dir, "composite_model.pkl"), "rb") as handle:
            return pickle.load(handle), None
    with open(os.path.join(model_dir, "gp_model.pkl"), "rb") as handle:
        gp = pickle.load(handle)
    with open(os.path.join(model_dir, "scaler.pkl"), "rb") as handle:
        scaler = pickle.load(handle)
    return gp, scaler


def predict(model, scaler, X: np.ndarray, is_composite_model: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Predict mean and uncertainty."""
    if is_composite_model:
        return model.predict(X, return_std=True)
    X_scaled = scaler.transform(X)
    return model.predict(X_scaled, return_std=True)


def active_features(row: pd.Series, feature_names: Sequence[str], threshold: float = 1e-6) -> List[str]:
    """Return non-zero features for one formulation."""
    features = []
    for name in feature_names:
        value = row.get(name, 0.0)
        if pd.isna(value):
            continue
        if abs(float(value)) > threshold:
            features.append(name)
    return features


def top_features_by_magnitude(row: pd.Series, feature_names: Sequence[str], limit: int = 2) -> List[str]:
    """Return the dominant active features in descending magnitude."""
    parts = []
    for name in feature_names:
        value = row.get(name, 0.0)
        if pd.isna(value):
            continue
        value = float(value)
        if abs(value) <= 1e-6:
            continue
        parts.append((abs(value), name))
    parts.sort(reverse=True)
    return [name for _, name in parts[:limit]]


def chemistry_family(row: pd.Series, feature_names: Sequence[str]) -> str:
    """Coarse family label used to avoid near-duplicate picks."""
    names = top_features_by_magnitude(row, feature_names, limit=2)
    if not names:
        return "empty"
    return "+".join(sorted(name.replace("_M", "").replace("_pct", "") for name in names))


def build_tested_signatures(validation_df: pd.DataFrame, feature_names: Sequence[str]) -> set[str]:
    """Build a set of already tested formulation signatures."""
    return {
        format_formulation(row, feature_names)
        for _, row in validation_df.iterrows()
    }


def load_candidate_files(target_stage: int, include_random: bool) -> List[str]:
    """Load candidate CSVs for one stage."""
    basenames = [
        f"bo_candidates_general_iteration_{target_stage}_prior_mean.csv",
        f"bo_candidates_dmso_free_iteration_{target_stage}_prior_mean.csv",
    ]
    if include_random:
        basenames.extend(
            [
                f"candidates_general_iteration_{target_stage}_prior_mean.csv",
                f"candidates_dmso_free_iteration_{target_stage}_prior_mean.csv",
            ]
        )
    paths = [os.path.join(RESULTS_DIR, name) for name in basenames if os.path.exists(os.path.join(RESULTS_DIR, name))]
    if not paths:
        raise FileNotFoundError(
            f"No candidate files found for stage {target_stage}. "
            "Run optimization first or pass a different stage."
        )
    return paths


def load_candidate_pool(
    target_stage: int,
    feature_names: Sequence[str],
    tested_signatures: set[str],
    include_random: bool,
) -> pd.DataFrame:
    """Load and align candidate files into one pool."""
    rows: List[dict] = []
    for path in load_candidate_files(target_stage, include_random):
        source_file = os.path.basename(path)
        source_kind = "bo" if source_file.startswith("bo_") else "random"
        df = pd.read_csv(path).fillna(0.0)
        for feature_name in feature_names:
            if feature_name not in df.columns:
                df[feature_name] = 0.0

        for _, row in df.iterrows():
            signature = format_formulation(row, feature_names)
            if signature in tested_signatures:
                continue
            record = {name: row.get(name, 0.0) for name in feature_names}
            record.update(
                {
                    "source_file": source_file,
                    "source_kind": source_kind,
                    "source_rank": int(row["rank"]),
                    "predicted_viability": float(row["predicted_viability"]),
                    "uncertainty": float(row["uncertainty"]),
                    "dmso_percent": float(row.get("dmso_percent", 0.0)),
                    "n_ingredients": int(row.get("n_ingredients", len(active_features(row, feature_names)))),
                    "signature": signature,
                    "chemistry_family": chemistry_family(row, feature_names),
                }
            )
            if "acquisition_value" in row.index:
                record["acquisition_value"] = float(row["acquisition_value"])
            rows.append(record)

    pool = pd.DataFrame(rows)
    if pool.empty:
        raise ValueError("No untested candidate formulations remain in the selected candidate pool.")
    return pool.drop_duplicates(subset=["signature", "source_file"]).reset_index(drop=True)


def build_residual_signals(
    validation_df: pd.DataFrame,
    last_completed_stage: int,
    stage_model: StageModel,
) -> Tuple[Dict[str, float], Dict[Tuple[str, str], float], Dict[str, int], Dict[Tuple[str, str], int]]:
    """Build feature- and pair-level blind-spot signals from the last completed stage."""
    batch = validation_df[validation_df["stage"] == last_completed_stage].copy()
    model, scaler = load_stage_predictor(stage_model)
    X = batch.reindex(columns=stage_model.feature_names, fill_value=0.0).fillna(0.0).to_numpy(float)
    pred, _ = predict(model, scaler, X, stage_model.is_composite_model)
    batch["residual"] = batch["viability_measured"].astype(float) - pred

    feature_signal = {name: 0.0 for name in stage_model.feature_names}
    feature_counts = {name: 0 for name in stage_model.feature_names}
    pair_signal: Dict[Tuple[str, str], float] = {}
    pair_counts: Dict[Tuple[str, str], int] = {}

    for _, row in batch.iterrows():
        active = active_features(row, stage_model.feature_names)
        if not active:
            continue
        residual = float(row["residual"])
        share = residual / len(active)
        for feature_name in active:
            feature_signal[feature_name] += share
            feature_counts[feature_name] += 1
        for pair in combinations(sorted(active), 2):
            pair_signal[pair] = pair_signal.get(pair, 0.0) + residual / max(1, len(active) - 1)
            pair_counts[pair] = pair_counts.get(pair, 0) + 1

    for feature_name, count in feature_counts.items():
        if count:
            feature_signal[feature_name] /= count
    for pair, count in list(pair_counts.items()):
        if count:
            pair_signal[pair] /= count

    wetlab_feature_counts = {name: 0 for name in stage_model.feature_names}
    wetlab_pair_counts: Dict[Tuple[str, str], int] = {}
    for _, row in validation_df.iterrows():
        active = active_features(row, stage_model.feature_names)
        for feature_name in active:
            wetlab_feature_counts[feature_name] += 1
        for pair in combinations(sorted(active), 2):
            wetlab_pair_counts[pair] = wetlab_pair_counts.get(pair, 0) + 1

    return feature_signal, pair_signal, wetlab_feature_counts, wetlab_pair_counts


def normalize(values: pd.Series) -> pd.Series:
    """Min-max normalize a numeric series."""
    if values.empty:
        return values
    min_value = float(values.min())
    max_value = float(values.max())
    if math.isclose(min_value, max_value):
        return pd.Series(np.full(len(values), 0.5), index=values.index)
    return (values - min_value) / (max_value - min_value)


def score_candidates(
    pool: pd.DataFrame,
    feature_names: Sequence[str],
    feature_signal: Dict[str, float],
    pair_signal: Dict[Tuple[str, str], float],
    wetlab_feature_counts: Dict[str, int],
    wetlab_pair_counts: Dict[Tuple[str, str], int],
) -> pd.DataFrame:
    """Attach exploitation and exploration scores to the pool."""
    blindspot_values = []
    pair_blindspot_values = []
    novelty_values = []
    feature_explanations = []

    for _, row in pool.iterrows():
        active = active_features(row, feature_names)
        pair_list = list(combinations(sorted(active), 2))

        per_feature = {name: feature_signal.get(name, 0.0) for name in active}
        positive_feature_names = [
            name for name, value in sorted(per_feature.items(), key=lambda item: item[1], reverse=True)
            if value > 0
        ]
        feature_explanations.append(positive_feature_names[:3])

        blindspot = 0.0
        pair_blindspot = 0.0
        if active:
            blindspot += sum(feature_signal.get(name, 0.0) for name in active) / len(active)
        if pair_list:
            pair_blindspot = sum(pair_signal.get(pair, 0.0) for pair in pair_list) / len(pair_list)
            blindspot += 0.5 * pair_blindspot
        blindspot_values.append(blindspot)
        pair_blindspot_values.append(pair_blindspot)

        novelty_parts = []
        for feature_name in active:
            novelty_parts.append(1.0 / (1 + wetlab_feature_counts.get(feature_name, 0)))
        for pair in pair_list:
            novelty_parts.append(1.0 / (1 + wetlab_pair_counts.get(pair, 0)))
        novelty_values.append(float(np.mean(novelty_parts)) if novelty_parts else 0.0)

    scored = pool.copy()
    scored["blindspot_raw"] = blindspot_values
    scored["pair_blindspot_raw"] = pair_blindspot_values
    scored["novelty_raw"] = novelty_values
    scored["blindspot_features"] = feature_explanations

    scored["pred_norm"] = normalize(scored["predicted_viability"])
    scored["unc_norm"] = normalize(scored["uncertainty"])
    scored["novelty_norm"] = normalize(scored["novelty_raw"])
    scored["blindspot_positive_norm"] = normalize(scored["blindspot_raw"].clip(lower=0.0))
    scored["pair_blindspot_positive_norm"] = normalize(scored["pair_blindspot_raw"].clip(lower=0.0))

    # Favor BO candidates by default; random-search candidates can still surface when requested.
    source_bonus = scored["source_kind"].map({"bo": 1.0, "random": 0.75}).fillna(0.75)

    scored["exploitation_score"] = (
        0.75 * scored["pred_norm"]
        + 0.15 * (1.0 - scored["unc_norm"])
        + 0.10 * source_bonus
    )
    scored["exploration_score"] = (
        0.30 * scored["unc_norm"]
        + 0.30 * scored["blindspot_positive_norm"]
        + 0.25 * scored["pair_blindspot_positive_norm"]
        + 0.05 * scored["pred_norm"]
        + 0.10 * scored["novelty_norm"]
        + 0.10 * source_bonus
    )
    return scored


def select_diverse(
    scored: pd.DataFrame,
    score_column: str,
    n_select: int,
    already_selected: Optional[set[str]] = None,
    family_limit: int = 2,
    min_predicted_viability: float = 0.0,
    min_active_features: int = 1,
    require_positive_pair_blindspot: bool = False,
) -> List[dict]:
    """Select top candidates with simple chemistry diversity constraints."""
    already_selected = already_selected or set()
    family_counts: Dict[str, int] = {}
    selected: List[dict] = []

    ranked = scored.sort_values(
        [score_column, "predicted_viability", "uncertainty"],
        ascending=[False, False, False],
    )
    for _, row in ranked.iterrows():
        if row["signature"] in already_selected:
            continue
        if float(row["predicted_viability"]) < min_predicted_viability:
            continue
        if len(active_features(row, [c for c in row.index if c.endswith("_M") or c.endswith("_pct")])) < min_active_features:
            continue
        if require_positive_pair_blindspot and float(row.get("pair_blindspot_raw", 0.0)) <= 0.0:
            continue
        family = str(row["chemistry_family"])
        if family_counts.get(family, 0) >= family_limit:
            continue
        selected.append(row.to_dict())
        already_selected.add(row["signature"])
        family_counts[family] = family_counts.get(family, 0) + 1
        if len(selected) >= n_select:
            break
    return selected


def rationale_text(candidate: dict, bucket: str) -> str:
    """Generate a short human-readable rationale."""
    features = candidate.get("blindspot_features") or []
    feature_hint = ""
    if features:
        pretty = ", ".join(
            name.replace("_M", "").replace("_pct", "") for name in features[:2]
        )
        feature_hint = f" Blind-spot features: {pretty}."

    if bucket == "exploit":
        return (
            "High predicted viability with comparatively manageable uncertainty."
            + feature_hint
        )
    return (
        "Chosen to improve calibration in a less-certain or under-modeled region."
        + feature_hint
    )


def save_outputs(
    target_stage: int,
    target_iteration_dir: str,
    exploit: List[dict],
    explore: List[dict],
    source_mode: str,
    last_completed_stage: int,
    feature_signal: Dict[str, float],
):
    """Write recommendations to CSV and summary text."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows: List[dict] = []
    for bucket, candidates in [("exploit", exploit), ("explore", explore)]:
        for order, candidate in enumerate(candidates, start=1):
            rows.append(
                {
                    "recommendation_type": bucket,
                    "bucket_rank": order,
                    "source_file": candidate["source_file"],
                    "source_kind": candidate["source_kind"],
                    "source_rank": candidate["source_rank"],
                    "predicted_viability": round_or_none(candidate["predicted_viability"]),
                    "uncertainty": round_or_none(candidate["uncertainty"]),
                    "exploitation_score": round_or_none(candidate["exploitation_score"]),
                    "exploration_score": round_or_none(candidate["exploration_score"]),
                    "blindspot_score": round_or_none(candidate["blindspot_raw"]),
                    "novelty_score": round_or_none(candidate["novelty_raw"]),
                    "dmso_percent": round_or_none(candidate["dmso_percent"]),
                    "n_ingredients": int(candidate["n_ingredients"]),
                    "formulation": candidate["signature"],
                    "rationale": rationale_text(candidate, bucket),
                }
            )

    output_df = pd.DataFrame(rows)
    output_tag = target_iteration_dir
    csv_path = os.path.join(RESULTS_DIR, f"validation_recommendations_{output_tag}.csv")
    txt_path = os.path.join(RESULTS_DIR, f"validation_recommendations_{output_tag}_summary.txt")
    output_df.to_csv(csv_path, index=False)

    top_positive_features = [
        name for name, value in sorted(feature_signal.items(), key=lambda item: item[1], reverse=True)
        if value > 0
    ][:5]
    top_positive_features = [
        name.replace("_M", "").replace("_pct", "") for name in top_positive_features
    ]

    lines = [
        "=" * 80,
        "CryoMN Validation Batch Recommendation",
        "=" * 80,
        f"Target stage: iteration {target_stage} ({target_iteration_dir})",
        f"Candidate source mode: {source_mode}",
        f"Blind-spot feedback source: stage {last_completed_stage}",
        f"Primary blind-spot features: {', '.join(top_positive_features) if top_positive_features else 'none detected'}",
        "",
        "Why this defaults to 05 BO:",
        "  The 05 candidate files already encode exploration/exploitation and stay closer",
        "  to supported chemistry. 03 is useful as a fallback pool, but not the default.",
        "",
        "Recommended exploitation picks:",
    ]

    for candidate in exploit:
        lines.extend(
            [
                f"- {candidate['signature']}",
                f"  source: {candidate['source_file']} rank {candidate['source_rank']}",
                f"  predicted viability: {candidate['predicted_viability']:.1f}% +/- {candidate['uncertainty']:.1f}%",
                f"  rationale: {rationale_text(candidate, 'exploit')}",
            ]
        )

    lines.append("")
    lines.append("Recommended exploration / calibration picks:")
    for candidate in explore:
        lines.extend(
            [
                f"- {candidate['signature']}",
                f"  source: {candidate['source_file']} rank {candidate['source_rank']}",
                f"  predicted viability: {candidate['predicted_viability']:.1f}% +/- {candidate['uncertainty']:.1f}%",
                f"  blindspot score: {candidate['blindspot_raw']:.2f} | novelty score: {candidate['novelty_raw']:.3f}",
                f"  rationale: {rationale_text(candidate, 'explore')}",
            ]
        )

    with open(txt_path, "w") as handle:
        handle.write("\n".join(lines) + "\n")

    print(f"Saved CSV: {csv_path}")
    print(f"Saved summary: {txt_path}")


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Recommend a balanced validation batch from current candidate files."
    )
    parser.add_argument("--stage", type=int, default=None, help="Target iteration stage. Defaults to active stage.")
    parser.add_argument("--total", type=int, default=6, help="Total number of recommendations.")
    parser.add_argument(
        "--exploit-ratio",
        type=float,
        default=2.0 / 3.0,
        help="Fraction of the batch reserved for exploitation picks.",
    )
    parser.add_argument(
        "--bo-only",
        action="store_true",
        help="Use only 05 BO candidate files and skip 03 random-search candidates.",
    )
    args = parser.parse_args()

    include_random = not args.bo_only
    validation_df = load_validation_df()
    active_metadata = load_active_metadata()
    target_stage = args.stage or int(active_metadata.get("iteration", 0))
    target_iteration_dir = str(active_metadata.get("iteration_dir", f"iteration_{target_stage}"))
    last_completed_stage = max(stage for stage in validation_df["stage"].dropna().astype(int) if stage < target_stage)

    feedback_model = discover_stage_model(last_completed_stage)
    feature_signal, pair_signal, wetlab_feature_counts, wetlab_pair_counts = build_residual_signals(
        validation_df, last_completed_stage, feedback_model
    )

    tested_signatures = build_tested_signatures(validation_df, feedback_model.feature_names)
    pool = load_candidate_pool(
        target_stage=target_stage,
        feature_names=feedback_model.feature_names,
        tested_signatures=tested_signatures,
        include_random=include_random,
    )
    scored = score_candidates(
        pool,
        feature_names=feedback_model.feature_names,
        feature_signal=feature_signal,
        pair_signal=pair_signal,
        wetlab_feature_counts=wetlab_feature_counts,
        wetlab_pair_counts=wetlab_pair_counts,
    )

    n_exploit = max(1, min(args.total - 1, int(round(args.total * args.exploit_ratio))))
    n_explore = max(1, args.total - n_exploit)

    selected_signatures: set[str] = set()
    exploit = select_diverse(
        scored,
        score_column="exploitation_score",
        n_select=n_exploit,
        already_selected=selected_signatures,
        family_limit=2,
        min_predicted_viability=45.0,
    )
    explore = select_diverse(
        scored,
        score_column="exploration_score",
        n_select=n_explore,
        already_selected=selected_signatures,
        family_limit=1,
        min_predicted_viability=30.0,
        min_active_features=2,
        require_positive_pair_blindspot=True,
    )

    print("=" * 80)
    print("CryoMN Validation Batch Recommendation")
    print("=" * 80)
    print(f"Target stage: iteration {target_stage} ({target_iteration_dir})")
    print(f"Feedback source: stage {last_completed_stage} residuals")
    print(f"Candidate pool: {'05 BO + 03 random' if include_random else '05 BO only'}")
    print(f"Recommended split: {len(exploit)} exploitation / {len(explore)} exploration")
    print("")

    print("Exploitation picks:")
    for candidate in exploit:
        print(
            f"  - {candidate['signature']} | {candidate['source_file']} rank {candidate['source_rank']} | "
            f"pred {candidate['predicted_viability']:.1f}% +/- {candidate['uncertainty']:.1f}%"
        )

    print("")
    print("Exploration / calibration picks:")
    for candidate in explore:
        print(
            f"  - {candidate['signature']} | {candidate['source_file']} rank {candidate['source_rank']} | "
            f"pred {candidate['predicted_viability']:.1f}% +/- {candidate['uncertainty']:.1f}% | "
            f"blindspot {candidate['blindspot_raw']:.2f}"
        )

    print("")
    save_outputs(
        target_stage=target_stage,
        target_iteration_dir=target_iteration_dir,
        exploit=exploit,
        explore=explore,
        source_mode="bo+random" if include_random else "bo",
        last_completed_stage=last_completed_stage,
        feature_signal=feature_signal,
    )


if __name__ == "__main__":
    main()
