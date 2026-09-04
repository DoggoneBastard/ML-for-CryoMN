#!/usr/bin/env python3
"""Generate five review-only V2 multi-objective visualization concepts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Iterable, Mapping
import warnings

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy.stats import norm


V2_ROOT = Path(__file__).resolve().parents[1]
if str(V2_ROOT) not in sys.path:
    sys.path.insert(0, str(V2_ROOT))

from helper.evaluation_metrics import (  # noqa: E402
    build_feasible_paired_objectives,
    compute_campaign_hypervolume_progress,
    compute_observed_pareto_front,
)
from helper.paths import (  # noqa: E402
    FORMULATIONS_PATH,
    NEXT_ROUND_CANDIDATES_PATH,
    OBSERVATIONS_PATH,
    PROJECT_ROOT,
    RESULTS_V2_DIR,
)
from helper.plot_theme import apply_plot_theme  # noqa: E402
from helper.registry import load_registry  # noqa: E402


DEFAULT_OUTPUT_DIR = RESULTS_V2_DIR / "visualization_concepts"
DEFAULT_PROSPECTIVE_TABLE = (
    RESULTS_V2_DIR
    / "reports"
    / "prospective"
    / "tables"
    / "prospective_evaluation_table.csv"
)
DEFAULT_PROSPECTIVE_METRICS = (
    RESULTS_V2_DIR
    / "reports"
    / "prospective"
    / "tables"
    / "prospective_metrics.csv"
)
REFERENCE_POINT = {
    "viability_percent": 0.0,
    "critical_axial_load_N_per_needle": 0.0,
}
CONCEPTS = {
    "concept_A_feasible_pareto": "Feasible Pareto evidence map",
    "concept_B_campaign_timeline": "Campaign learning timeline",
    "concept_C_surrogate_trust": "Surrogate trust sheet",
    "concept_D_candidate_decision": "Next-round decision sheet",
    "concept_E_publication_triptych": "Publication triptych",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _round_number(value: object) -> int | None:
    text = str(value).strip()
    if text.startswith("ROUND_") and text.removeprefix("ROUND_").isdigit():
        return int(text.removeprefix("ROUND_"))
    return None


def _wilson_interval(passes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return np.nan, np.nan
    proportion = passes / total
    denominator = 1.0 + z**2 / total
    center = (proportion + z**2 / (2.0 * total)) / denominator
    half_width = (
        z
        * np.sqrt(
            proportion * (1.0 - proportion) / total
            + z**2 / (4.0 * total**2)
        )
        / denominator
    )
    return max(0.0, center - half_width), min(1.0, center + half_width)


def _save_figure(fig: plt.Figure, directory: Path, stem: str) -> tuple[Path, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    png = directory / f"{stem}.png"
    pdf = directory / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def _write_bundle(
    directory: Path,
    stem: str,
    table: pd.DataFrame,
    metadata: Mapping[str, object],
    caption: str,
) -> None:
    table.to_csv(directory / f"{stem}_tidy_data.csv", index=False)
    (directory / "metadata.json").write_text(
        json.dumps(dict(metadata), indent=2) + "\n",
        encoding="utf-8",
    )
    (directory / "caption.txt").write_text(caption.strip() + "\n", encoding="utf-8")


def _base_metadata(
    formulations_path: Path,
    observations_path: Path,
    candidates_path: Path,
    prospective_table_path: Path,
    observations: pd.DataFrame,
    cohort: str,
    seed: int,
    synthetic_status: str,
) -> dict[str, object]:
    paths = {
        "formulations": formulations_path,
        "observations": observations_path,
        "candidates": candidates_path,
        "prospective_table": prospective_table_path,
    }
    return {
        "prototype_policy": "v2_visualization_concepts_v1",
        "cohort": cohort,
        "seed": seed,
        "reference_point": REFERENCE_POINT,
        "synthetic_data_status": synthetic_status,
        "endpoint_counts": {
            str(endpoint): int(count)
            for endpoint, count in observations.get(
                "endpoint", pd.Series(dtype=str)
            ).value_counts().items()
        },
        "data_hashes": {
            name: _sha256(path)
            for name, path in paths.items()
            if path.exists()
        },
    }


def _short_formulation(row: pd.Series) -> str:
    registry = load_registry()
    parts: list[str] = []
    for feature in registry.feature_names:
        value = pd.to_numeric(row.get(feature), errors="coerce")
        if pd.isna(value) or float(value) <= 0:
            continue
        name = registry.get_by_feature(feature).display_name
        parts.append(name)
    if not parts:
        return "no active ingredients"
    if len(parts) <= 3:
        return " + ".join(parts)
    return " + ".join(parts[:3]) + f" +{len(parts) - 3}"


def _formal_viability(prospective: pd.DataFrame) -> pd.DataFrame:
    mask = (
        prospective["endpoint"].astype(str).eq("viability_percent")
        & prospective["formal_metric_eligible"].astype(bool)
    )
    result = prospective.loc[mask].copy()
    for column in (
        "prediction_mean",
        "prediction_std",
        "observed_mean",
        "absolute_error",
    ):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.dropna(subset=["prediction_mean", "observed_mean"])


def _concept_a(
    output_dir: Path,
    metadata: Mapping[str, object],
    seed: int,
) -> Path:
    directory = output_dir / "concept_A_feasible_pareto"
    rng = np.random.default_rng(seed)
    observed = pd.DataFrame(
        {
            "point_id": [f"demo_obs_{index:02d}" for index in range(1, 12)],
            "point_type": "feasible_observed",
            "viability_percent": [24, 34, 43, 51, 59, 66, 72, 79, 83, 48, 69],
            "critical_axial_load_N_per_needle": [2.55, 2.35, 2.12, 1.95, 1.72, 1.50, 1.34, 1.10, 0.92, 1.35, 0.85],
            "viability_std": np.nan,
            "load_std": np.nan,
            "intact_status": "pass",
        }
    )
    observed.loc[[3, 9], "point_type"] = "failed_intact"
    observed.loc[[3, 9], "intact_status"] = "fail"
    recommendations = pd.DataFrame(
        {
            "point_id": [f"demo_rec_{index:02d}" for index in range(1, 5)],
            "point_type": "recommended_prediction",
            "viability_percent": [55, 64, 74, 82] + rng.normal(0, 0.5, 4),
            "critical_axial_load_N_per_needle": [2.25, 1.92, 1.53, 1.18] + rng.normal(0, 0.025, 4),
            "viability_std": [5.0, 6.0, 7.0, 8.0],
            "load_std": [0.16, 0.18, 0.20, 0.22],
            "intact_status": "predicted",
        }
    )
    table = pd.concat([observed, recommendations], ignore_index=True)
    feasible = observed.loc[observed["intact_status"].eq("pass")].copy()
    frontier = compute_observed_pareto_front(feasible)
    dominated = frontier.loc[frontier["is_dominated"]]
    pareto = frontier.loc[frontier["is_pareto"]].sort_values("viability_percent")
    failed = observed.loc[observed["intact_status"].eq("fail")]

    theme = apply_plot_theme("legacy_inspired_publication")
    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    ax.scatter(
        dominated["viability_percent"],
        dominated["critical_axial_load_N_per_needle"],
        s=60,
        color=theme.slate,
        alpha=0.55,
        label="Feasible, dominated observation",
    )
    ax.plot(
        pareto["viability_percent"],
        pareto["critical_axial_load_N_per_needle"],
        color=theme.teal,
        linewidth=1.8,
        zorder=2,
    )
    ax.scatter(
        pareto["viability_percent"],
        pareto["critical_axial_load_N_per_needle"],
        s=76,
        color=theme.teal,
        marker="o",
        label="Feasible, nondominated observation",
        zorder=3,
    )
    ax.scatter(
        failed["viability_percent"],
        failed["critical_axial_load_N_per_needle"],
        s=82,
        color=theme.coral,
        marker="x",
        linewidth=2,
        label="Failed intact gate (excluded)",
    )
    ax.errorbar(
        recommendations["viability_percent"],
        recommendations["critical_axial_load_N_per_needle"],
        xerr=1.96 * recommendations["viability_std"],
        yerr=1.96 * recommendations["load_std"],
        fmt="D",
        markerfacecolor="none",
        markeredgecolor=theme.gold,
        ecolor=theme.gold,
        elinewidth=1,
        capsize=2,
        markersize=7,
        label="Recommended prediction ±95% interval",
    )
    for index, row in recommendations.reset_index(drop=True).iterrows():
        ax.annotate(
            f"R9-{index + 1}",
            (row["viability_percent"], row["critical_axial_load_N_per_needle"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            color=theme.gold,
        )
    ax.scatter([0], [0], marker="+", s=80, color=theme.text, label="Fixed HV reference (0, 0)")
    ax.set_xlabel("Viability (%)")
    ax.set_ylabel("Critical axial load (N/needle)")
    ax.set_title("Concept A — Feasible Pareto evidence map")
    ax.set_xlim(-2, 100)
    ax.set_ylim(-0.05, 3.0)
    ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    ax.text(
        0.5,
        0.48,
        "LAYOUT DEMO — SYNTHETIC MECHANICS DATA",
        transform=ax.transAxes,
        ha="center",
        va="center",
        rotation=24,
        fontsize=17,
        color=theme.coral,
        alpha=0.25,
        weight="bold",
    )
    ax.text(
        0.01,
        0.01,
        "Current V2 load observations: 0 • not a scientific result",
        transform=ax.transAxes,
        color=theme.muted,
        fontsize=9,
    )
    fig.tight_layout()
    png, _ = _save_figure(fig, directory, "concept_A_feasible_pareto")
    concept_metadata = dict(metadata)
    concept_metadata.update(
        {
            "cohort": "layout demonstration only",
            "synthetic_data_status": "synthetic_mechanics_layout_demo",
            "excluded_from_scientific_metrics": True,
        }
    )
    _write_bundle(
        directory,
        "concept_A_feasible_pareto",
        table,
        concept_metadata,
        "A structural preview of the future viability–load trade-off map. All mechanics values in this concept are deterministic synthetic layout data because V2 has no measured critical-load observations. Failed-intact points are explicitly excluded from the demonstration frontier, while recommendations remain visually distinct from observations.",
    )
    return png


def _timeline_table(
    observations: pd.DataFrame,
    prospective: pd.DataFrame,
    metrics: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    completed_rounds = sorted(
        {
            str(value)
            for value in observations.get("batch_id", pd.Series(dtype=str))
            if _round_number(value) is not None
        },
        key=lambda value: int(value.removeprefix("ROUND_")),
    )
    proposed_rounds = sorted(
        {
            str(value)
            for value in candidates.get("batch_id", pd.Series(dtype=str))
            if _round_number(value) is not None
        },
        key=lambda value: int(value.removeprefix("ROUND_")),
    )
    rounds = sorted(
        set(completed_rounds) | set(proposed_rounds),
        key=lambda value: int(value.removeprefix("ROUND_")),
    )
    rows: list[dict[str, object]] = []
    round_metrics = metrics.loc[metrics["scope"].astype(str).eq("round")]
    for round_id in rounds:
        number = int(round_id.removeprefix("ROUND_"))
        round_obs = observations.loc[observations["batch_id"].astype(str).eq(round_id)].copy()
        phase_values = prospective.loc[
            prospective["round_id"].astype(str).eq(round_id), "active_phase"
        ].dropna()
        phase = str(phase_values.iloc[0]) if not phase_values.empty else "mechanics_bootstrap"
        provenance_values = prospective.loc[
            prospective["round_id"].astype(str).eq(round_id), "provenance_class"
        ].dropna()
        provenance = str(provenance_values.iloc[0]) if not provenance_values.empty else "current_proposal"
        viability = round_obs.loc[
            round_obs["endpoint"].astype(str).eq("viability_percent")
        ].copy()
        if not viability.empty:
            viability["value"] = pd.to_numeric(viability["value"], errors="coerce")
            grouped = viability.groupby("formulation_id")["value"].mean().dropna()
            for formulation_id, value in grouped.items():
                rows.append(
                    {
                        "round_id": round_id,
                        "round_number": number,
                        "phase": phase,
                        "provenance": provenance,
                        "panel": "observed_viability",
                        "series": "individual_formulation",
                        "formulation_id": formulation_id,
                        "value": float(value),
                        "lower": np.nan,
                        "upper": np.nan,
                        "count_pass": np.nan,
                        "count_fail": np.nan,
                    }
                )
            rows.append(
                {
                    "round_id": round_id,
                    "round_number": number,
                    "phase": phase,
                    "provenance": provenance,
                    "panel": "observed_viability",
                    "series": "median_iqr",
                    "formulation_id": "",
                    "value": float(grouped.median()),
                    "lower": float(grouped.quantile(0.25)),
                    "upper": float(grouped.quantile(0.75)),
                    "count_pass": np.nan,
                    "count_fail": np.nan,
                }
            )
        for metric_name in ("mae", "bias", "interval_95_coverage"):
            match = round_metrics.loc[
                round_metrics["round_id"].astype(str).eq(round_id)
                & round_metrics["endpoint"].astype(str).eq("viability_percent")
            ]
            value = pd.to_numeric(match.iloc[0].get(metric_name), errors="coerce") if not match.empty else np.nan
            rows.append(
                {
                    "round_id": round_id,
                    "round_number": number,
                    "phase": phase,
                    "provenance": provenance,
                    "panel": "prospective_performance",
                    "series": metric_name,
                    "formulation_id": "",
                    "value": value,
                    "lower": np.nan,
                    "upper": np.nan,
                    "count_pass": np.nan,
                    "count_fail": np.nan,
                }
            )
        intact = round_obs.loc[
            round_obs["endpoint"].astype(str).eq("intact_patch_formation_pass")
        ].copy()
        if intact.empty:
            passes = failures = total = 0
            proportion = low = high = np.nan
        else:
            intact["value"] = pd.to_numeric(intact["value"], errors="coerce")
            aggregated = intact.groupby("formulation_id")["value"].min().dropna()
            passes = int(aggregated.ge(0.5).sum())
            failures = int(aggregated.lt(0.5).sum())
            total = passes + failures
            proportion = passes / total if total else np.nan
            low, high = _wilson_interval(passes, total)
        rows.append(
            {
                "round_id": round_id,
                "round_number": number,
                "phase": phase,
                "provenance": provenance,
                "panel": "feasibility_mechanics",
                "series": "intact_pass_proportion",
                "formulation_id": "",
                "value": proportion,
                "lower": low,
                "upper": high,
                "count_pass": passes,
                "count_fail": failures,
            }
        )
        mechanical_count = int(
            round_obs["endpoint"].astype(str).eq(
                "critical_axial_load_N_per_needle"
            ).sum()
        )
        rows.append(
            {
                "round_id": round_id,
                "round_number": number,
                "phase": phase,
                "provenance": provenance,
                "panel": "feasibility_mechanics",
                "series": "mechanical_measurement_count",
                "formulation_id": "",
                "value": mechanical_count,
                "lower": np.nan,
                "upper": np.nan,
                "count_pass": np.nan,
                "count_fail": np.nan,
            }
        )
    return pd.DataFrame(rows)


def _plot_timeline(table: pd.DataFrame, title: str) -> plt.Figure:
    theme = apply_plot_theme("legacy_inspired_publication")
    rounds = sorted(table["round_number"].dropna().astype(int).unique())
    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9.4), sharex=True)
    viability = table.loc[table["panel"].eq("observed_viability")]
    individual = viability.loc[viability["series"].eq("individual_formulation")]
    jitter = ((individual.groupby("round_number").cumcount() % 7) - 3) * 0.025
    axes[0].scatter(
        individual["round_number"] + jitter,
        individual["value"],
        s=24,
        color=theme.slate,
        alpha=0.34,
        edgecolor="none",
        label="Formulation mean",
    )
    median = viability.loc[viability["series"].eq("median_iqr")]
    axes[0].errorbar(
        median["round_number"],
        median["value"],
        yerr=np.vstack(
            [median["value"] - median["lower"], median["upper"] - median["value"]]
        ),
        color=theme.blue,
        marker="o",
        linewidth=1.6,
        capsize=3,
        label="Median and IQR",
    )
    axes[0].set_ylabel("Observed viability (%)")
    axes[0].set_ylim(-3, 105)
    axes[0].legend(frameon=False, ncol=2, fontsize=8.5, loc="upper left")

    performance = table.loc[table["panel"].eq("prospective_performance")]
    formal = performance["provenance"].astype(str).eq("formal_frozen")
    for metric_name, color, marker in (
        ("mae", theme.coral, "o"),
        ("bias", theme.gold, "s"),
    ):
        frame = performance.loc[performance["series"].eq(metric_name)]
        axes[1].plot(
            frame["round_number"],
            frame["value"],
            color=color,
            marker=marker,
            linewidth=1.5,
            label=metric_name.upper() if metric_name == "mae" else "Bias",
        )
    axes[1].axhline(0, color=theme.muted, linewidth=0.8)
    axes[1].set_ylabel("Prospective error\n(viability points)")
    coverage = performance.loc[performance["series"].eq("interval_95_coverage")]
    coverage_axis = axes[1].twinx()
    coverage_axis.plot(
        coverage["round_number"],
        100 * pd.to_numeric(coverage["value"], errors="coerce"),
        color=theme.teal,
        marker="^",
        linestyle="--",
        linewidth=1.3,
        label="95% interval coverage",
    )
    coverage_axis.axhline(95, color=theme.teal, linestyle=":", linewidth=0.9)
    coverage_axis.set_ylabel("Empirical coverage (%)")
    coverage_axis.set_ylim(0, 105)
    handles, labels = axes[1].get_legend_handles_labels()
    handles2, labels2 = coverage_axis.get_legend_handles_labels()
    axes[1].legend(handles + handles2, labels + labels2, frameon=False, ncol=3, fontsize=8.2, loc="upper left")
    formal_rounds = sorted(
        performance.loc[formal, "round_number"].dropna().astype(int).unique()
    )
    if formal_rounds:
        axes[1].axvspan(min(formal_rounds) - 0.45, max(formal_rounds) + 0.45, color=theme.sky, alpha=0.08)
        axes[1].text(
            0.98,
            0.91,
            f"Formal frozen cohort: R{min(formal_rounds)}–R{max(formal_rounds)}",
            transform=axes[1].transAxes,
            ha="right",
            color=theme.blue,
            fontsize=8.5,
        )

    feasibility = table.loc[
        table["series"].eq("intact_pass_proportion")
    ].copy()
    axes[2].errorbar(
        feasibility["round_number"],
        feasibility["value"],
        yerr=np.vstack(
            [feasibility["value"] - feasibility["lower"], feasibility["upper"] - feasibility["value"]]
        ),
        color=theme.teal,
        marker="o",
        linewidth=1.5,
        capsize=3,
        label="Intact pass proportion (Wilson 95% CI)",
    )
    for _, row in feasibility.dropna(subset=["value"]).iterrows():
        axes[2].text(
            row["round_number"],
            min(float(row["upper"]) + 0.04, 1.03),
            f"{int(row['count_pass'])}P/{int(row['count_fail'])}F",
            ha="center",
            fontsize=7.8,
            color=theme.muted,
        )
    axes[2].set_ylabel("Intact pass proportion")
    axes[2].set_ylim(-0.04, 1.08)
    axes[2].set_xlabel("Campaign round")
    axes[2].set_xticks(rounds)
    axes[2].set_xticklabels([f"R{number}" for number in rounds])
    axes[2].legend(frameon=False, fontsize=8.5, loc="lower left")
    mechanics = table.loc[table["series"].eq("mechanical_measurement_count")]
    for axis in axes:
        axis.axvspan(8.5, 9.5, color=theme.gold, alpha=0.06, zorder=0)
    axes[0].text(
        9.0,
        101,
        "mechanics-bootstrap proposal",
        ha="center",
        va="top",
        fontsize=8.2,
        color=theme.gold,
    )
    axes[2].text(
        0.99,
        0.06,
        f"Completed critical-load measurements: {int(mechanics['value'].sum())}\nHypervolume: not estimable",
        transform=axes[2].transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color=theme.coral,
    )
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, y=0.995, fontsize=15)
    fig.tight_layout()
    return fig


def _concept_b(
    output_dir: Path,
    metadata: Mapping[str, object],
    observations: pd.DataFrame,
    prospective: pd.DataFrame,
    metrics: pd.DataFrame,
    candidates: pd.DataFrame,
) -> tuple[Path, pd.DataFrame]:
    directory = output_dir / "concept_B_campaign_timeline"
    table = _timeline_table(observations, prospective, metrics, candidates)
    fig = _plot_timeline(table, "Concept B — Campaign learning timeline")
    png, _ = _save_figure(fig, directory, "concept_B_campaign_timeline")
    concept_metadata = dict(metadata)
    concept_metadata.update({"cohort": "all completed V2 rounds plus current proposal", "synthetic_data_status": "none"})
    _write_bundle(
        directory,
        "concept_B_campaign_timeline",
        table,
        concept_metadata,
        "The campaign timeline separates observed viability, frozen prospective error, and intact-patch feasibility. Formal frozen rounds are distinguished from reconstructed or migration-era evidence. The mechanics lane reports zero completed critical-load measurements, so no hypervolume curve is drawn.",
    )
    return png, table


def _trust_table(prospective: pd.DataFrame) -> pd.DataFrame:
    viability = _formal_viability(prospective)
    rows: list[dict[str, object]] = []
    for _, row in viability.iterrows():
        rows.append(
            {
                "panel": "viability_prediction",
                "candidate_id": row["candidate_id"],
                "round_id": row["round_id"],
                "prediction": row["prediction_mean"],
                "observed": row["observed_mean"],
                "uncertainty_std": row["prediction_std"],
                "absolute_error": row["absolute_error"],
                "nominal_coverage": np.nan,
                "empirical_coverage": np.nan,
                "interval_width": 3.92 * row["prediction_std"] if pd.notna(row["prediction_std"]) else np.nan,
                "coverage_lower": np.nan,
                "coverage_upper": np.nan,
                "binary_outcome": np.nan,
            }
        )
    valid_std = viability.dropna(subset=["prediction_std"])
    for nominal in (0.50, 0.68, 0.80, 0.90, 0.95):
        z_value = norm.ppf((1.0 + nominal) / 2.0)
        covered = (
            (valid_std["observed_mean"] >= valid_std["prediction_mean"] - z_value * valid_std["prediction_std"])
            & (valid_std["observed_mean"] <= valid_std["prediction_mean"] + z_value * valid_std["prediction_std"])
        )
        covered_count = int(covered.sum())
        coverage_lower, coverage_upper = _wilson_interval(
            covered_count,
            int(len(covered)),
        )
        rows.append(
            {
                "panel": "coverage",
                "candidate_id": "",
                "round_id": "FORMAL_COHORT",
                "prediction": np.nan,
                "observed": np.nan,
                "uncertainty_std": np.nan,
                "absolute_error": np.nan,
                "nominal_coverage": nominal,
                "empirical_coverage": float(covered.mean()) if len(covered) else np.nan,
                "interval_width": float((2 * z_value * valid_std["prediction_std"]).mean()) if len(valid_std) else np.nan,
                "coverage_lower": coverage_lower,
                "coverage_upper": coverage_upper,
                "binary_outcome": np.nan,
            }
        )
    intact = prospective.loc[
        prospective["endpoint"].astype(str).eq("intact_patch_formation_pass")
        & prospective["formal_metric_eligible"].astype(bool)
    ].copy()
    for _, row in intact.iterrows():
        rows.append(
            {
                "panel": "intact_probability",
                "candidate_id": row["candidate_id"],
                "round_id": row["round_id"],
                "prediction": pd.to_numeric(row["prediction_mean"], errors="coerce"),
                "observed": np.nan,
                "uncertainty_std": np.nan,
                "absolute_error": np.nan,
                "nominal_coverage": np.nan,
                "empirical_coverage": np.nan,
                "interval_width": np.nan,
                "coverage_lower": np.nan,
                "coverage_upper": np.nan,
                "binary_outcome": pd.to_numeric(row["observed_mean"], errors="coerce"),
            }
        )
    return pd.DataFrame(rows)


def _plot_trust(table: pd.DataFrame, metrics: pd.DataFrame, title: str) -> plt.Figure:
    theme = apply_plot_theme("legacy_inspired_publication")
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.5))
    viability = table.loc[table["panel"].eq("viability_prediction")]
    ax = axes[0, 0]
    ax.errorbar(
        viability["observed"],
        viability["prediction"],
        yerr=1.96 * pd.to_numeric(viability["uncertainty_std"], errors="coerce"),
        fmt="o",
        markersize=4.5,
        alpha=0.55,
        color=theme.blue,
        ecolor=theme.sky,
        elinewidth=0.7,
        capsize=1,
    )
    limits = [-5, max(105.0, float(viability[["prediction", "observed"]].max().max()) + 5)]
    ax.plot(limits, limits, linestyle="--", color=theme.muted, linewidth=1)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_xlabel("Observed viability (%)")
    ax.set_ylabel("Frozen predicted viability (%)")
    formal = metrics.loc[
        metrics["scope"].astype(str).eq("pooled_formal")
        & metrics["endpoint"].astype(str).eq("viability_percent")
    ]
    if not formal.empty:
        row = formal.iloc[0]
        ax.text(
            0.03,
            0.97,
            f"n={int(row['n_evaluated'])}  MAE={row['mae']:.1f}  bias={row['bias']:+.1f}\nR²={row['r2']:.2f}  95% coverage={100*row['interval_95_coverage']:.1f}%",
            transform=ax.transAxes,
            va="top",
            fontsize=8.6,
        )

    ax = axes[0, 1]
    ax.scatter(
        viability["uncertainty_std"],
        viability["absolute_error"],
        color=theme.coral,
        alpha=0.58,
        s=30,
    )
    maximum = max(1.0, float(viability[["uncertainty_std", "absolute_error"]].max().max()))
    ax.plot([0, maximum], [0, maximum], linestyle="--", color=theme.muted, linewidth=1)
    ax.set_xlabel("Predicted standard deviation (viability points)")
    ax.set_ylabel("Absolute prospective error (viability points)")
    ax.set_title("Does stated uncertainty track error?", fontsize=11)

    ax = axes[1, 0]
    coverage = table.loc[table["panel"].eq("coverage")]
    ax.plot([0, 1], [0, 1], linestyle="--", color=theme.muted, linewidth=1)
    ax.errorbar(
        coverage["nominal_coverage"],
        coverage["empirical_coverage"],
        yerr=np.vstack(
            [
                coverage["empirical_coverage"] - coverage["coverage_lower"],
                coverage["coverage_upper"] - coverage["empirical_coverage"],
            ]
        ),
        marker="o",
        color=theme.teal,
        linewidth=1.6,
        capsize=3,
    )
    ax.set_xlabel("Nominal interval coverage")
    ax.set_ylabel("Observed formal-cohort coverage")
    ax.set_xlim(0.45, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Coverage (n ≥ 30 permits a curve)", fontsize=11)

    ax = axes[1, 1]
    intact = table.loc[table["panel"].eq("intact_probability")]
    jitter = ((np.arange(len(intact)) % 9) - 4) * 0.012
    colors = np.where(intact["binary_outcome"].ge(0.5), theme.teal, theme.coral)
    ax.scatter(
        intact["prediction"],
        intact["binary_outcome"] + jitter,
        c=colors,
        s=30,
        alpha=0.62,
    )
    ax.axvline(0.5, color=theme.muted, linestyle="--", linewidth=1)
    ax.set_xlabel("Frozen intact-pass probability")
    ax.set_ylabel("Observed intact outcome (0 fail, 1 pass)")
    ax.set_yticks([0, 1])
    intact_metric = metrics.loc[
        metrics["scope"].astype(str).eq("pooled_formal")
        & metrics["endpoint"].astype(str).eq("intact_patch_formation_pass")
    ]
    brier = float(intact_metric.iloc[0]["brier_score"]) if not intact_metric.empty else np.nan
    passes = int(intact["binary_outcome"].ge(0.5).sum())
    failures = int(intact["binary_outcome"].lt(0.5).sum())
    ax.set_title(f"Intact feasibility • {passes} pass/{failures} fail • Brier {brier:.3f}", fontsize=10.5)
    ax.text(
        0.02,
        0.50,
        "Mechanical surrogate: not trained/evaluable\ncritical-load observations = 0",
        transform=ax.transAxes,
        color=theme.coral,
        fontsize=8.7,
        va="center",
    )
    for axis in axes.flat:
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle(title, y=0.995, fontsize=15)
    fig.tight_layout()
    return fig


def _concept_c(
    output_dir: Path,
    metadata: Mapping[str, object],
    prospective: pd.DataFrame,
    metrics: pd.DataFrame,
) -> tuple[Path, pd.DataFrame]:
    directory = output_dir / "concept_C_surrogate_trust"
    table = _trust_table(prospective)
    fig = _plot_trust(table, metrics, "Concept C — Surrogate trust sheet")
    png, _ = _save_figure(fig, directory, "concept_C_surrogate_trust")
    concept_metadata = dict(metadata)
    concept_metadata.update({"cohort": "formal frozen prospective cohort", "synthetic_data_status": "none"})
    _write_bundle(
        directory,
        "concept_C_surrogate_trust",
        table,
        concept_metadata,
        "Formal frozen predictions are compared with later experiments, with model error, uncertainty magnitude, empirical coverage, and intact feasibility kept distinct. The positive viability bias and sub-nominal 95% coverage remain visible rather than being summarized as recommendation success. Mechanical validation metrics are omitted because no critical-load outcomes exist.",
    )
    return png, table


def _decision_table(candidates: pd.DataFrame) -> pd.DataFrame:
    frame = candidates.copy().sort_values("selection_rank")
    frame["formulation_description"] = frame.apply(_short_formulation, axis=1)
    status = frame["viability_prediction_status"].fillna("").astype(str)
    reportable = ~status.str.startswith("unknown_")
    frame["public_viability_mean"] = pd.to_numeric(
        frame["predicted_viability_percent"], errors="coerce"
    ).where(reportable)
    frame["public_viability_std"] = pd.to_numeric(
        frame["viability_std"], errors="coerce"
    ).where(reportable)
    columns = [
        "selection_rank",
        "candidate_id",
        "formulation_id",
        "formulation_description",
        "selection_role",
        "recommendation_type",
        "viability_prediction_status",
        "public_viability_mean",
        "public_viability_std",
        "empirical_combination_pass_probability",
        "support_status",
        "candidate_origin",
        "mechanical_test_recommended",
        "mechanical_selection_rank",
        "mechanical_transition_role",
    ]
    return frame[columns].reset_index(drop=True)


def _concept_d(
    output_dir: Path,
    metadata: Mapping[str, object],
    candidates: pd.DataFrame,
) -> Path:
    directory = output_dir / "concept_D_candidate_decision"
    table = _decision_table(candidates)
    theme = apply_plot_theme("legacy_inspired_publication")
    fig = plt.figure(figsize=(13.2, 8.6))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 1.45], wspace=0.08)
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    y = np.arange(len(table))[::-1]
    known = table["public_viability_mean"].notna()
    axes[0].errorbar(
        table.loc[known, "public_viability_mean"],
        y[known.to_numpy()],
        xerr=1.96 * pd.to_numeric(table.loc[known, "public_viability_std"], errors="coerce"),
        fmt="o",
        color=theme.blue,
        ecolor=theme.sky,
        elinewidth=1,
        capsize=2,
        markersize=6,
    )
    for position, (_, row) in zip(y, table.iterrows()):
        if pd.isna(row["public_viability_mean"]):
            axes[0].text(3, position, "Unreported", va="center", color=theme.muted, fontsize=8.5)
    axes[0].set_xlim(0, 105)
    axes[0].set_xlabel("Reportable viability prediction (%)")
    labels = [
        f"#{int(row.selection_rank)}  {row.candidate_id}\n{row.formulation_description}"
        for row in table.itertuples()
    ]
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels, fontsize=7.7)

    probabilities = pd.to_numeric(
        table["empirical_combination_pass_probability"], errors="coerce"
    )
    axes[1].hlines(y, 0, probabilities, color=theme.sky, linewidth=4, alpha=0.45)
    axes[1].scatter(probabilities, y, color=theme.teal, marker="s", s=38)
    axes[1].axvline(0.5, color=theme.coral, linestyle="--", linewidth=1)
    axes[1].set_xlim(0, 1.02)
    axes[1].set_xlabel("Empirical combination\nintact probability")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])

    axes[2].set_xlim(0, 1)
    axes[2].set_ylim(-0.8, len(table) - 0.2)
    axes[2].set_yticks(y)
    axes[2].set_yticklabels([])
    axes[2].set_xticks([])
    axes[2].set_xlabel("Decision attributes")
    for position, (_, row) in zip(y, table.iterrows()):
        mechanical = bool(row["mechanical_test_recommended"])
        rank = pd.to_numeric(row["mechanical_selection_rank"], errors="coerce")
        mech_text = f"MECH #{int(rank)}" if mechanical and pd.notna(rank) else "screen only"
        role = str(row["mechanical_transition_role"]).strip()
        if not role or role.lower() == "nan":
            role = str(row["selection_role"])
        color = theme.gold if mechanical else theme.muted
        axes[2].scatter(0.04, position, marker="D" if mechanical else "o", s=45, facecolors="none" if mechanical else color, edgecolors=color)
        axes[2].text(
            0.10,
            position,
            f"{mech_text}  •  {role}\n{row['candidate_origin']}  •  support={row['support_status']}",
            va="center",
            fontsize=8.0,
            color=color if mechanical else theme.text,
        )
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Concept D — Why these 12, and which four receive mechanics?", y=0.99, fontsize=15)
    fig.text(
        0.99,
        0.01,
        "Unknown viability estimates are labelled, not positioned using raw surrogate diagnostics.",
        ha="right",
        color=theme.muted,
        fontsize=8.5,
    )
    png, _ = _save_figure(fig, directory, "concept_D_candidate_decision")
    concept_metadata = dict(metadata)
    concept_metadata.update({"cohort": "active Round 9 frozen proposal", "synthetic_data_status": "none"})
    _write_bundle(
        directory,
        "concept_D_candidate_decision",
        table,
        concept_metadata,
        "The operator sheet exposes the active decision evidence for all twelve candidates in selection order. It uses empirical ingredient-combination intact probability, not the diagnostic classifier, and assigns no numeric plotting position to an unreported viability prediction. Mechanical recommendations and ranks are shown as a separate execution decision.",
    )
    return png


def _concept_e(
    output_dir: Path,
    metadata: Mapping[str, object],
    timeline: pd.DataFrame,
    trust: pd.DataFrame,
    metrics: pd.DataFrame,
) -> Path:
    directory = output_dir / "concept_E_publication_triptych"
    theme = apply_plot_theme("legacy_inspired_publication")
    fig = plt.figure(figsize=(12.0, 7.2))
    grid = fig.add_gridspec(2, 5, width_ratios=[1, 1, 1, 0.92, 0.92], hspace=0.34, wspace=0.55)
    pareto_ax = fig.add_subplot(grid[:, :3])
    parity_ax = fig.add_subplot(grid[0, 3:])
    progress_ax = fig.add_subplot(grid[1, 3:])
    pareto_ax.set_xlabel("Observed viability (%)")
    pareto_ax.set_ylabel("Observed critical axial load (N/needle)")
    pareto_ax.set_xlim(0, 100)
    pareto_ax.set_ylim(0, 1)
    pareto_ax.text(
        0.5,
        0.55,
        "Observed Pareto front not yet estimable",
        transform=pareto_ax.transAxes,
        ha="center",
        va="center",
        fontsize=14,
        color=theme.coral,
    )
    pareto_ax.text(
        0.5,
        0.45,
        "0 paired viability + critical-load measurements\nIntact feasibility is required before frontier entry",
        transform=pareto_ax.transAxes,
        ha="center",
        va="center",
        fontsize=9.5,
        color=theme.muted,
    )
    pareto_ax.set_title("A  Feasible observed trade-off", loc="left", fontsize=11.5)

    viability = trust.loc[trust["panel"].eq("viability_prediction")]
    parity_ax.scatter(
        viability["observed"],
        viability["prediction"],
        color=theme.blue,
        alpha=0.48,
        s=24,
    )
    parity_ax.plot([-5, 110], [-5, 110], color=theme.muted, linestyle="--", linewidth=1)
    parity_ax.set_xlim(-5, 110)
    parity_ax.set_ylim(-5, 110)
    parity_ax.set_xlabel("Observed viability (%)")
    parity_ax.set_ylabel("Frozen prediction (%)")
    parity_ax.set_title("B  Formal prospective validation", loc="left", fontsize=11.5)
    formal = metrics.loc[
        metrics["scope"].astype(str).eq("pooled_formal")
        & metrics["endpoint"].astype(str).eq("viability_percent")
    ]
    if not formal.empty:
        row = formal.iloc[0]
        parity_ax.text(
            0.03,
            0.96,
            f"n={int(row['n_evaluated'])} • MAE {row['mae']:.1f} • bias {row['bias']:+.1f}\n95% coverage {100*row['interval_95_coverage']:.1f}%",
            transform=parity_ax.transAxes,
            va="top",
            fontsize=8,
        )

    viability_summary = timeline.loc[
        timeline["panel"].eq("observed_viability")
        & timeline["series"].eq("median_iqr")
    ]
    feasibility = timeline.loc[timeline["series"].eq("intact_pass_proportion")]
    progress_ax.plot(
        viability_summary["round_number"],
        viability_summary["value"],
        color=theme.blue,
        marker="o",
        linewidth=1.5,
        label="Median viability (%)",
    )
    feasibility_axis = progress_ax.twinx()
    feasibility_axis.plot(
        feasibility["round_number"],
        feasibility["value"],
        color=theme.teal,
        marker="s",
        linestyle="--",
        linewidth=1.3,
        label="Intact pass proportion",
    )
    progress_ax.set_xlabel("Campaign round")
    progress_ax.set_ylabel("Median viability (%)")
    feasibility_axis.set_ylabel("Intact pass proportion")
    feasibility_axis.set_ylim(0, 1.05)
    progress_ax.set_title("C  Campaign progress; HV withheld", loc="left", fontsize=11.5)
    handles, labels = progress_ax.get_legend_handles_labels()
    handles2, labels2 = feasibility_axis.get_legend_handles_labels()
    progress_ax.legend(handles + handles2, labels + labels2, frameon=False, fontsize=7.8, loc="lower left")
    progress_ax.text(
        0.98,
        0.82,
        "Fixed-reference HV: not estimable",
        transform=progress_ax.transAxes,
        ha="right",
        va="top",
        color=theme.coral,
        fontsize=8,
    )
    for axis in (pareto_ax, parity_ax, progress_ax):
        axis.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Concept E — Publication triptych (current evidence state)", y=0.995, fontsize=15)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fig.tight_layout()
    png, _ = _save_figure(fig, directory, "concept_E_publication_triptych")
    table = pd.concat(
        [
            trust.assign(source_concept="surrogate_trust"),
            timeline.assign(source_concept="campaign_timeline"),
        ],
        ignore_index=True,
        sort=False,
    )
    concept_metadata = dict(metadata)
    concept_metadata.update({"cohort": "current real V2 evidence only", "synthetic_data_status": "none"})
    _write_bundle(
        directory,
        "concept_E_publication_triptych",
        table,
        concept_metadata,
        "The compact triptych reserves the largest panel for the eventual feasible observed Pareto front, while currently stating why it cannot be estimated. Formal prospective viability performance and real campaign progress occupy the supporting panels. Hypervolume is withheld rather than displayed as zero.",
    )
    return png


def _comparison_sheet(output_dir: Path, images: Iterable[Path]) -> Path:
    theme = apply_plot_theme("legacy_inspired_publication")
    image_paths = list(images)
    fig, axes = plt.subplots(3, 2, figsize=(12.5, 14.5))
    axes = axes.flatten()
    for axis, path in zip(axes, image_paths):
        axis.imshow(plt.imread(path))
        axis.set_title(CONCEPTS[path.parent.name], fontsize=11)
        axis.axis("off")
    axes[-1].axis("off")
    axes[-1].text(
        0.05,
        0.78,
        "Present dataset: Concept B\nCampaign learning timeline",
        fontsize=14,
        color=theme.blue,
        va="top",
    )
    axes[-1].text(
        0.05,
        0.48,
        "Mature campaign: Concept E\nPublication triptych",
        fontsize=14,
        color=theme.teal,
        va="top",
    )
    axes[-1].text(
        0.05,
        0.18,
        "Concept A is a watermarked\nsynthetic layout demonstration.",
        fontsize=11,
        color=theme.coral,
        va="top",
    )
    fig.suptitle("V2 visualization concept comparison", fontsize=16, y=0.995)
    fig.tight_layout()
    path = output_dir / "concept_comparison.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def _write_gallery_readme(output_dir: Path) -> None:
    text = """# V2 Visualization Concept Gallery

These figures are experimental V2-only concepts. They are not imported by the
production reporting commands and do not change any optimizer or campaign
metric.

## Concept A — Feasible Pareto evidence map

This is the clearest mature-campaign display of the two-objective trade-off and
the intact feasibility gate. The current V2 database has no load measurements,
so this concept uses deterministic synthetic mechanics values solely to test
layout. Its watermark, file metadata, and caption prevent it from being
mistaken for scientific evidence.

## Concept B — Campaign learning timeline

This is the strongest design for the present dataset because every scientific
panel can be populated honestly. It separates experimental viability,
prospective error, intact feasibility, and the absence of mechanics evidence.
The reader can see both experimental progress and the model's persistent
positive bias without inferring an unavailable Pareto result.

## Concept C — Surrogate trust sheet

This is the strongest internal diagnostic view. It makes predictive accuracy,
uncertainty calibration, interval sharpness, and feasibility classification
separate questions. The formal frozen cohort is primary, and the mechanical
panel reports an evidence count rather than a zero-valued performance metric.

## Concept D — Next-round decision sheet

This is the strongest operator-facing view because it explains all twelve
selected rows and the four mechanical assignments. Unknown public viability
predictions are labelled without plotting their raw diagnostic means. The
active empirical ingredient-combination probability is shown instead of the
diagnostic intact classifier.

## Concept E — Publication triptych

This is the recommended mature-campaign manuscript or presentation design.
The Pareto map receives the dominant visual area, while prospective validation
and fixed-reference progress provide supporting evidence. In the current state,
the Pareto and hypervolume positions are explicitly withheld because paired
mechanics evidence does not yet exist.

## Recommendation

Use Concept B for current campaign review, Concept C for model diagnostics, and
Concept D at the wet-lab handoff. Revisit Concept E after enough feasible paired
mechanical observations exist. Do not integrate any concept into production
reports until a direction is explicitly selected.
"""
    (output_dir / "README.md").write_text(text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formulations", default=str(FORMULATIONS_PATH))
    parser.add_argument("--observations", default=str(OBSERVATIONS_PATH))
    parser.add_argument("--candidates", default=str(NEXT_ROUND_CANDIDATES_PATH))
    parser.add_argument("--prospective-table", default=str(DEFAULT_PROSPECTIVE_TABLE))
    parser.add_argument("--prospective-metrics", default=str(DEFAULT_PROSPECTIVE_METRICS))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    formulations_path = Path(args.formulations)
    observations_path = Path(args.observations)
    candidates_path = Path(args.candidates)
    prospective_table_path = Path(args.prospective_table)
    prospective_metrics_path = Path(args.prospective_metrics)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    formulations = pd.read_csv(formulations_path)
    observations = pd.read_csv(observations_path)
    candidates = pd.read_csv(candidates_path)
    prospective = pd.read_csv(prospective_table_path)
    metrics = pd.read_csv(prospective_metrics_path)
    feasible_pairs, _ = build_feasible_paired_objectives(formulations, observations)
    hypervolume = compute_campaign_hypervolume_progress(feasible_pairs, REFERENCE_POINT)
    if not hypervolume.empty:
        hypervolume.to_csv(output_dir / "scientific_hypervolume_progress.csv", index=False)

    base_metadata = _base_metadata(
        formulations_path,
        observations_path,
        candidates_path,
        prospective_table_path,
        observations,
        cohort="concept-specific",
        seed=args.seed,
        synthetic_status="none",
    )
    image_a = _concept_a(output_dir, base_metadata, args.seed)
    image_b, timeline = _concept_b(
        output_dir,
        base_metadata,
        observations,
        prospective,
        metrics,
        candidates,
    )
    image_c, trust = _concept_c(output_dir, base_metadata, prospective, metrics)
    image_d = _concept_d(output_dir, base_metadata, candidates)
    image_e = _concept_e(output_dir, base_metadata, timeline, trust, metrics)
    comparison = _comparison_sheet(output_dir, [image_a, image_b, image_c, image_d, image_e])
    _write_gallery_readme(output_dir)
    print(f"Generated five V2 visualization concepts: {output_dir.resolve()}")
    print(f"Comparison sheet: {comparison.resolve()}")


if __name__ == "__main__":
    main()
