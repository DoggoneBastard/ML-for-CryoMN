"""Matplotlib-only rendering functions for V2 evaluation reports."""

from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, brier_score_loss, mean_absolute_error, r2_score

from .config import load_optimization_config
from .evaluation_metrics import (
    _build_model_evaluation_frames,
    _cross_validated_predictions,
    _normalize_frame,
    _pareto_frontier_mask,
    _round_metrics,
    _round_sort_key,
    _observed_endpoint_frame,
)
from .models import build_training_frame
from .plot_theme import get_plot_theme
from .registry import IngredientRegistry


_COMPAT = get_plot_theme("v2_compat")
PAGE_BG = _COMPAT.page_background
AX_BG = _COMPAT.axes_background
GRID = _COMPAT.grid
TEXT = _COMPAT.text
MUTED = _COMPAT.muted
BLUE = _COMPAT.blue
TEAL = _COMPAT.teal
GOLD = _COMPAT.gold
CORAL = _COMPAT.coral
SLATE = _COMPAT.slate

def _artifact_path(
    output_dir: Path,
    base_name: str,
    suffix: str,
    artifact_prefix: str = "",
) -> Path:
    prefix = f"{artifact_prefix}_" if str(artifact_prefix).strip() else ""
    return output_dir / f"{prefix}{base_name}{suffix}"


def _apply_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "axes.facecolor": AX_BG,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "figure.facecolor": PAGE_BG,
            "savefig.facecolor": PAGE_BG,
            "grid.color": GRID,
            "grid.alpha": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
        }
    )


def _format_metric(value: float | None, fmt: str = "{:.2f}") -> str:
    if value is None or not np.isfinite(value):
        return "n/a"
    return fmt.format(float(value))


def _save_endpoint_counts(
    observations: pd.DataFrame,
    output_dir: Path,
    artifact_prefix: str = "",
) -> Path | None:
    if observations.empty or "endpoint" not in observations.columns:
        return None
    counts = observations["endpoint"].value_counts().sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    fig.patch.set_facecolor(PAGE_BG)
    bars = ax.barh(counts.index, counts.values, color=[BLUE, GOLD, TEAL, CORAL][: len(counts)])
    ax.set_xlabel("Observation rows")
    ax.set_title("Where the v2 evidence currently lives", pad=14)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    for bar, value in zip(bars, counts.values):
        ax.text(bar.get_width() + max(counts.values) * 0.02, bar.get_y() + bar.get_height() / 2, str(int(value)), va="center")
    fig.tight_layout()
    path = _artifact_path(output_dir, "endpoint_observation_counts", ".png", artifact_prefix)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_observed_performance_landscape(
    formulations: pd.DataFrame,
    observations: pd.DataFrame,
    output_dir: Path,
    artifact_prefix: str = "",
) -> Path | None:
    frame = _observed_endpoint_frame(formulations, observations)
    required = {"viability_percent", "critical_axial_load_N_per_needle"}
    if frame.empty or not required.issubset(frame.columns):
        return None
    plot_data = frame.dropna(subset=list(required)).copy()
    if plot_data.empty:
        return None

    frontier_mask = _pareto_frontier_mask(
        plot_data,
        "viability_percent",
        "critical_axial_load_N_per_needle",
    )
    frontier = plot_data.loc[frontier_mask].sort_values("viability_percent")

    fig, ax = plt.subplots(figsize=(8.2, 6))
    fig.patch.set_facecolor(PAGE_BG)
    ax.axvspan(plot_data["viability_percent"].quantile(0.75), plot_data["viability_percent"].max() + 2, color=TEAL, alpha=0.08)
    ax.scatter(
        plot_data["viability_percent"],
        plot_data["critical_axial_load_N_per_needle"],
        s=70,
        color=SLATE,
        alpha=0.35,
        edgecolor="white",
        linewidth=0.6,
        label="Observed formulations",
    )
    ax.scatter(
        frontier["viability_percent"],
        frontier["critical_axial_load_N_per_needle"],
        s=90,
        color=TEAL,
        edgecolor="white",
        linewidth=0.8,
        label="Pareto frontier",
        zorder=3,
    )
    if len(frontier) > 1:
        ax.plot(
            frontier["viability_percent"],
            frontier["critical_axial_load_N_per_needle"],
            color=TEAL,
            linewidth=2,
            alpha=0.8,
            zorder=2,
        )
    ax.set_xlabel("Observed viability (%)")
    ax.set_ylabel("Observed critical axial load (N/needle)")
    ax.set_title("Observed performance landscape", pad=14)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    path = _artifact_path(output_dir, "observed_performance_landscape", ".png", artifact_prefix)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_placeholder(
    output_dir: Path,
    base_name: str,
    title: str,
    message: str,
    artifact_prefix: str = "",
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(PAGE_BG)
    ax.axis("off")
    ax.set_title(title, pad=16)
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=14, color=MUTED, transform=ax.transAxes)
    fig.tight_layout()
    path = _artifact_path(output_dir, base_name, ".png", artifact_prefix)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _plot_parity_axis(
    ax: plt.Axes,
    frame: pd.DataFrame,
    title: str,
    unit_label: str,
    color: str,
) -> dict[str, float | int | None]:
    if frame.empty:
        ax.text(0.5, 0.5, "Not enough data yet", ha="center", va="center", color=MUTED, transform=ax.transAxes)
        ax.set_title(title)
        ax.set_xlabel(f"Observed {unit_label}")
        ax.set_ylabel(f"Predicted {unit_label}")
        return {"n": 0, "mae": None, "r2": None}

    actual = frame["actual"].to_numpy(dtype=float)
    predicted = frame["predicted"].to_numpy(dtype=float)
    low = float(np.nanmin(np.concatenate([actual, predicted])))
    high = float(np.nanmax(np.concatenate([actual, predicted])))
    pad = max((high - low) * 0.08, 1e-6)

    ax.scatter(actual, predicted, s=75, color=color, alpha=0.85, edgecolor="white", linewidth=0.7)
    ax.plot([low - pad, high + pad], [low - pad, high + pad], color=MUTED, linestyle="--", linewidth=1.5)
    ax.set_xlim(low - pad, high + pad)
    ax.set_ylim(low - pad, high + pad)
    ax.set_title(title)
    ax.set_xlabel(f"Observed {unit_label}")
    ax.set_ylabel(f"Predicted {unit_label}")

    mae = float(mean_absolute_error(actual, predicted))
    r2: float | None
    if len(frame) >= 2 and np.nanstd(actual) > 0:
        r2 = float(r2_score(actual, predicted))
    else:
        r2 = None
    ax.text(
        0.03,
        0.97,
        f"n={len(frame)}\nMAE={_format_metric(mae)}\nR²={_format_metric(r2)}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": AX_BG, "edgecolor": GRID},
    )
    return {"n": len(frame), "mae": mae, "r2": r2}


def _plot_probability_axis(ax: plt.Axes, frame: pd.DataFrame) -> dict[str, float | int | None]:
    if frame.empty:
        ax.text(0.5, 0.5, "Not enough pass/fail data yet", ha="center", va="center", color=MUTED, transform=ax.transAxes)
        ax.set_title("Intact-patch pass probabilities")
        ax.set_xlabel("Predicted pass probability")
        return {"n": 0, "brier": None, "accuracy": None}

    plot_data = frame.copy()
    jitter = np.linspace(-0.08, 0.08, len(plot_data)) if len(plot_data) > 1 else np.array([0.0])
    plot_data["y"] = plot_data["actual"] + jitter
    colors = plot_data["actual"].map({0.0: CORAL, 1.0: TEAL}).fillna(BLUE)
    ax.scatter(
        plot_data["predicted"],
        plot_data["y"],
        s=75,
        c=colors,
        alpha=0.85,
        edgecolor="white",
        linewidth=0.7,
    )
    ax.set_title("Intact-patch pass probabilities")
    ax.set_xlabel("Predicted pass probability")
    ax.set_yticks([0, 1], labels=["Observed fail", "Observed pass"])
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.4, 1.4)

    actual = plot_data["actual"].to_numpy(dtype=float)
    predicted = plot_data["predicted"].to_numpy(dtype=float)
    accuracy = float(accuracy_score(actual, predicted >= 0.5))
    brier = float(brier_score_loss(actual, predicted))
    ax.text(
        0.03,
        0.97,
        f"n={len(plot_data)}\nAccuracy={_format_metric(accuracy)}\nBrier={_format_metric(brier)}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": AX_BG, "edgecolor": GRID},
    )
    return {"n": len(plot_data), "brier": brier, "accuracy": accuracy}


def _save_model_evaluation_overview(
    formulations: pd.DataFrame,
    observations: pd.DataFrame,
    output_dir: Path,
    registry: IngredientRegistry,
    artifact_prefix: str = "",
) -> Path | None:
    if formulations.empty or observations.empty:
        return None

    evaluation_frames = _build_model_evaluation_frames(formulations, observations, registry)
    viability_cv = evaluation_frames["viability_percent"]
    load_cv = evaluation_frames["critical_axial_load_N_per_needle"]
    intact_cv = evaluation_frames["intact_patch_formation_pass"]
    if viability_cv.empty and load_cv.empty and intact_cv.empty:
        return None

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10))
    fig.patch.set_facecolor(PAGE_BG)
    ax_viability, ax_load, ax_intact, ax_scorecard = axes.flatten()

    viability_metrics = _plot_parity_axis(
        ax_viability,
        viability_cv,
        "Formulation-grouped cross-validation: viability",
        "viability (%)",
        BLUE,
    )
    load_metrics = _plot_parity_axis(
        ax_load,
        load_cv,
        "Formulation-grouped cross-validation: mechanics",
        "critical load (N/needle)",
        GOLD,
    )
    intact_metrics = _plot_probability_axis(ax_intact, intact_cv)

    ax_scorecard.axis("off")
    score_lines = [
        "Reader notes",
        "",
        "These panels use cross-validated predictions",
        "built from the current v2 database.",
        "",
        f"Viability rows evaluated: {viability_metrics['n']}",
        f"Mechanical rows evaluated: {load_metrics['n']}",
        f"Intact pass/fail rows evaluated: {intact_metrics['n']}",
        "",
        f"Viability MAE: {_format_metric(viability_metrics['mae'])}",
        f"Mechanical MAE: {_format_metric(load_metrics['mae'])}",
        f"Intact accuracy: {_format_metric(intact_metrics['accuracy'])}",
        "",
        "Use this figure to judge whether the model",
        "is tracking the current data cleanly enough",
        "to support the next round of experiments.",
    ]
    ax_scorecard.text(
        0.02,
        0.98,
        "\n".join(score_lines),
        va="top",
        ha="left",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.5", "facecolor": AX_BG, "edgecolor": GRID},
    )
    fig.suptitle("V2 cross-validated model evaluation overview", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    path = _artifact_path(output_dir, "model_evaluation_overview", ".png", artifact_prefix)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_multiobjective_parity_plot(
    viability_predictions: pd.DataFrame,
    load_predictions: pd.DataFrame,
    output_dir: Path,
    artifact_prefix: str = "",
) -> Path:
    if viability_predictions.empty and load_predictions.empty:
        return _save_placeholder(
            output_dir,
            "multiobjective_paired_parity",
            "Multi-objective parity overview",
            "Not enough real paired data yet.",
            artifact_prefix=artifact_prefix,
        )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.patch.set_facecolor(PAGE_BG)
    panels = [
        (axes[0], viability_predictions, "Viability parity", "Observed viability (%)", "Predicted viability (%)", BLUE),
        (axes[1], load_predictions, "Critical-load parity", "Observed critical load (N/needle)", "Predicted critical load (N/needle)", GOLD),
    ]
    for ax, frame, title, xlabel, ylabel, color in panels:
        if frame.empty:
            ax.text(0.5, 0.5, "Not enough data yet", ha="center", va="center", color=MUTED, transform=ax.transAxes)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            continue
        actual = frame["actual"].to_numpy(dtype=float)
        predicted = frame["predicted"].to_numpy(dtype=float)
        low = float(np.nanmin(np.concatenate([actual, predicted])))
        high = float(np.nanmax(np.concatenate([actual, predicted])))
        pad = max((high - low) * 0.08, 1e-6)
        ax.scatter(actual, predicted, s=70, color=color, alpha=0.85, edgecolor="white", linewidth=0.7)
        ax.plot([low - pad, high + pad], [low - pad, high + pad], linestyle="--", color=MUTED, linewidth=1.4)
        r2 = float(r2_score(actual, predicted)) if len(frame) >= 2 and np.nanstd(actual) > 0 else np.nan
        ax.text(
            0.03,
            0.97,
            f"n={len(frame)}\nR²={r2:.3f}" if np.isfinite(r2) else f"n={len(frame)}\nR²=n/a",
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox={"boxstyle": "round,pad=0.35", "facecolor": AX_BG, "edgecolor": GRID},
        )
        ax.set_xlim(low - pad, high + pad)
        ax.set_ylim(low - pad, high + pad)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    fig.suptitle("Observed vs predicted fits from the real multi-objective database", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = _artifact_path(output_dir, "multiobjective_paired_parity", ".png", artifact_prefix)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_candidate_plot(
    candidates: pd.DataFrame,
    output_dir: Path,
    artifact_prefix: str = "",
) -> Path | None:
    if candidates.empty or "intact_patch_pass_probability" not in candidates.columns:
        return None
    frame = candidates.copy()
    frame["predicted_viability_percent"] = pd.to_numeric(
        frame.get("predicted_viability_percent"),
        errors="coerce",
    )
    frame["raw_surrogate_viability_mean"] = pd.to_numeric(
        frame.get("raw_surrogate_viability_mean"),
        errors="coerce",
    )
    frame["_plot_viability"] = frame["predicted_viability_percent"].fillna(
        frame["raw_surrogate_viability_mean"]
    )
    frame["_unknown_viability"] = frame.get(
        "viability_prediction_status",
        pd.Series("", index=frame.index, dtype=object),
    ).astype(str).str.startswith("unknown_")
    frame["intact_patch_pass_probability"] = pd.to_numeric(frame["intact_patch_pass_probability"], errors="coerce")
    frame["predicted_critical_axial_load_N_per_needle"] = pd.to_numeric(
        frame.get("predicted_critical_axial_load_N_per_needle"),
        errors="coerce",
    )
    frame["selection_rank"] = pd.to_numeric(frame.get("selection_rank"), errors="coerce")
    frame = frame.dropna(subset=["_plot_viability", "intact_patch_pass_probability"])
    if frame.empty:
        return None

    mechanical_flags = (
        frame["mechanical_test_recommended"].astype(bool)
        if "mechanical_test_recommended" in frame.columns
        else pd.Series(False, index=frame.index)
    )
    sizes = frame["predicted_critical_axial_load_N_per_needle"].fillna(0.0).to_numpy(dtype=float)
    if np.nanmax(sizes) > 0:
        sizes = 80 + 160 * (sizes / max(np.nanmax(sizes), 1e-9))
    else:
        sizes = np.full(len(frame), 100.0)

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(PAGE_BG)
    high_viability = float(frame["_plot_viability"].quantile(0.75))
    high_pass = float(frame["intact_patch_pass_probability"].quantile(0.75))
    ax.axvspan(high_viability, frame["_plot_viability"].max() + 1, color=GOLD, alpha=0.08)
    ax.axhspan(high_pass, 1.03, color=TEAL, alpha=0.08)

    known = ~frame["_unknown_viability"]
    screen_only = frame.loc[~mechanical_flags & known]
    mechanical = frame.loc[mechanical_flags & known]
    unknown = frame.loc[~known]
    if not screen_only.empty:
        ax.scatter(
            screen_only["_plot_viability"],
            screen_only["intact_patch_pass_probability"],
            s=sizes[(~mechanical_flags & known).to_numpy()],
            color=BLUE,
            alpha=0.82,
            edgecolor="white",
            linewidth=0.7,
            label="Viability screen",
        )
    if not mechanical.empty:
        ax.scatter(
            mechanical["_plot_viability"],
            mechanical["intact_patch_pass_probability"],
            s=sizes[(mechanical_flags & known).to_numpy()],
            color=CORAL,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.8,
            label="Mechanical follow-up",
        )
    if not unknown.empty:
        ax.scatter(
            unknown["_plot_viability"],
            unknown["intact_patch_pass_probability"],
            s=sizes[(~known).to_numpy()],
            marker="X",
            color=SLATE,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.8,
            label="Unknown viability (raw GP diagnostic)",
        )
    for _, row in frame.nsmallest(5, "selection_rank").iterrows():
        if pd.isna(row["selection_rank"]):
            continue
        ax.annotate(
            f"#{int(row['selection_rank'])}",
            (row["_plot_viability"], row["intact_patch_pass_probability"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
            color=TEXT,
        )
    ax.set_xlabel("Viability estimate; raw GP diagnostic for unknown rows (%)")
    ax.set_ylabel("Predicted intact-patch probability")
    ax.set_ylim(-0.03, 1.03)
    ax.set_title("Next-round candidate screen", pad=14)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    path = _artifact_path(output_dir, "next_round_candidate_screen", ".png", artifact_prefix)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _save_hv_igd_plot(metrics: pd.DataFrame, output_dir: Path, artifact_prefix: str = "") -> Path:
    if metrics.empty:
        return _save_placeholder(
            output_dir,
            "normalized_hypervolume_igd_vs_round",
            "Normalized hypervolume and IGD vs round",
            "No paired viability/load rounds yet.",
            artifact_prefix=artifact_prefix,
        )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    fig.patch.set_facecolor(PAGE_BG)
    x = np.arange(1, len(metrics) + 1)
    axes[0].plot(x, metrics["normalized_hypervolume"], marker="o", color=TEAL, linewidth=2)
    axes[0].set_title("Normalized hypervolume vs round")
    axes[0].set_xlabel("Round")
    axes[0].set_ylabel("Normalized hypervolume")
    axes[0].set_xticks(x, metrics["batch_id"], rotation=30, ha="right")

    axes[1].plot(x, metrics["igd"], marker="o", color=CORAL, linewidth=2)
    axes[1].set_title("IGD vs round")
    axes[1].set_xlabel("Round")
    axes[1].set_ylabel("IGD")
    axes[1].set_xticks(x, metrics["batch_id"], rotation=30, ha="right")
    fig.tight_layout()
    path = _artifact_path(output_dir, "normalized_hypervolume_igd_vs_round", ".png", artifact_prefix)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_pareto_progression_plot(paired: pd.DataFrame, output_dir: Path, artifact_prefix: str = "") -> Path:
    if paired.empty:
        return _save_placeholder(
            output_dir,
            "pareto_front_progression",
            "Pareto-front progression",
            "No paired viability/load observations yet.",
            artifact_prefix=artifact_prefix,
        )
    rounds = sorted(paired["batch_id"].unique(), key=_round_sort_key)
    colors = plt.cm.viridis(np.linspace(0.15, 0.95, len(rounds)))

    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(PAGE_BG)
    ax.scatter(
        paired["viability_percent"],
        paired["critical_axial_load_N_per_needle"],
        s=40,
        color=SLATE,
        alpha=0.18,
        edgecolor="none",
        label="All paired observations",
    )
    for color, round_id in zip(colors, rounds):
        cumulative = paired.loc[paired["batch_id"].map(_round_sort_key) <= _round_sort_key(round_id)].copy()
        frontier = cumulative.loc[
            _pareto_frontier_mask(cumulative, "viability_percent", "critical_axial_load_N_per_needle")
        ].sort_values("viability_percent")
        if frontier.empty:
            continue
        ax.plot(
            frontier["viability_percent"],
            frontier["critical_axial_load_N_per_needle"],
            color=color,
            linewidth=2,
            marker="o",
            label=round_id,
        )
    ax.set_title("Pareto-front progression across real multi-objective rounds")
    ax.set_xlabel("Observed viability (%)")
    ax.set_ylabel("Observed critical load (N/needle)")
    ax.legend(frameon=False, loc="best", fontsize=8)
    fig.tight_layout()
    path = _artifact_path(output_dir, "pareto_front_progression", ".png", artifact_prefix)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _save_endpoint_r2_plot(
    formulations: pd.DataFrame,
    observations: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: Path,
    registry: IngredientRegistry,
    artifact_prefix: str = "",
) -> Path:
    if metrics.empty:
        return _save_placeholder(
            output_dir,
            "endpoint_r2_vs_round",
            "Endpoint R² vs round",
            "No paired viability/load rounds yet.",
            artifact_prefix=artifact_prefix,
        )

    rounds = metrics["batch_id"].tolist()
    rows = []
    for round_id in rounds:
        cumulative_obs = observations.loc[observations["batch_id"].map(_round_sort_key) <= _round_sort_key(round_id)].copy()
        cumulative_frame = build_training_frame(formulations, cumulative_obs, registry)
        paired_keys = cumulative_frame.dropna(subset=["viability_percent", "critical_axial_load_N_per_needle"])[["formulation_id", "batch_id"]]
        if paired_keys.empty:
            rows.append({"batch_id": round_id, "viability_r2": np.nan, "load_r2": np.nan})
            continue
        allowed = set((str(row.formulation_id), str(row.batch_id)) for row in paired_keys.itertuples())
        filtered_obs = cumulative_obs.loc[
            cumulative_obs.apply(lambda row: (str(row.get("formulation_id", "")), str(row.get("batch_id", ""))) in allowed, axis=1)
        ].copy()
        viability_predictions = _cross_validated_predictions(formulations, filtered_obs, registry, "viability_percent")
        load_predictions = _cross_validated_predictions(
            formulations,
            filtered_obs,
            registry,
            "critical_axial_load_N_per_needle",
        )
        viability_r2 = (
            float(r2_score(viability_predictions["actual"], viability_predictions["predicted"]))
            if len(viability_predictions) >= 2 and np.nanstd(viability_predictions["actual"]) > 0
            else np.nan
        )
        load_r2 = (
            float(r2_score(load_predictions["actual"], load_predictions["predicted"]))
            if len(load_predictions) >= 2 and np.nanstd(load_predictions["actual"]) > 0
            else np.nan
        )
        rows.append({"batch_id": round_id, "viability_r2": viability_r2, "load_r2": load_r2})
    r2_frame = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(8.5, 5))
    fig.patch.set_facecolor(PAGE_BG)
    x = np.arange(1, len(r2_frame) + 1)
    ax.plot(x, r2_frame["viability_r2"], marker="o", linewidth=2, color=BLUE, label="Viability R²")
    ax.plot(x, r2_frame["load_r2"], marker="o", linewidth=2, color=GOLD, label="Critical-load R²")
    ax.set_title("Formulation-grouped cross-validation R² vs cumulative round")
    ax.set_xlabel("Round")
    ax.set_ylabel("R²")
    ax.set_xticks(x, r2_frame["batch_id"], rotation=30, ha="right")
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    path = _artifact_path(output_dir, "endpoint_r2_vs_round", ".png", artifact_prefix)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.facecolor": AX_BG,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT,
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "figure.facecolor": PAGE_BG,
            "savefig.facecolor": PAGE_BG,
            "grid.color": GRID,
            "grid.alpha": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
        }
    )


def _placeholder_plot(path: Path, title: str, message: str) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axis("off")
    ax.set_title(title)
    ax.text(0.5, 0.5, message, ha="center", va="center", color=MUTED)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _prediction_plot(table: pd.DataFrame, path: Path) -> Path:
    continuous = table[
        table["evaluation_eligible"].astype(bool)
        & (table["metric_type"].astype(str) == "continuous")
    ].copy()
    endpoints = [
        endpoint
        for endpoint in [
            "viability_percent",
            "critical_axial_load_N_per_needle",
        ]
        if endpoint in set(continuous["endpoint"].astype(str))
    ]
    if not endpoints:
        return _placeholder_plot(
            path,
            "Frozen predictions vs observed results",
            "No eligible continuous prospective observations yet.",
        )
    fig, axes = plt.subplots(1, len(endpoints), figsize=(7 * len(endpoints), 5.5))
    axes_array = np.atleast_1d(axes)
    labels = {
        "viability_percent": "Viability (%)",
        "critical_axial_load_N_per_needle": "Critical load (N/needle)",
    }
    for ax, endpoint in zip(axes_array, endpoints):
        frame = continuous[continuous["endpoint"].astype(str) == endpoint]
        actual = pd.to_numeric(frame["observed_mean"], errors="coerce").to_numpy(dtype=float)
        predicted = pd.to_numeric(frame["prediction_mean"], errors="coerce").to_numpy(dtype=float)
        std = pd.to_numeric(frame["prediction_std"], errors="coerce").to_numpy(dtype=float)
        valid_std = np.isfinite(std) & (std >= 0)
        yerr = np.where(valid_std, 1.96 * std, 0.0)
        colors = [TEAL if bool(value) else BLUE for value in frame["formal_cohort"]]
        ax.errorbar(
            actual,
            predicted,
            yerr=yerr,
            fmt="none",
            ecolor=GRID,
            alpha=0.8,
            capsize=3,
        )
        ax.scatter(
            actual,
            predicted,
            s=70,
            c=colors,
            edgecolor="white",
            linewidth=0.7,
            alpha=0.9,
        )
        low = float(np.nanmin(np.concatenate([actual, predicted])))
        high = float(np.nanmax(np.concatenate([actual, predicted])))
        pad = max((high - low) * 0.08, 1e-6)
        ax.plot([low - pad, high + pad], [low - pad, high + pad], "--", color=MUTED)
        ax.set_xlim(low - pad, high + pad)
        ax.set_ylim(low - pad, high + pad)
        ax.set_xlabel(f"Observed {labels[endpoint]}")
        ax.set_ylabel(f"Frozen prediction {labels[endpoint]}")
        ax.set_title(labels[endpoint])
    fig.suptitle("Proposal-time predictions vs observed results", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _gate_plot(table: pd.DataFrame, path: Path) -> Path:
    frame = table[
        table["evaluation_eligible"].astype(bool)
        & (table["endpoint"].astype(str) == "intact_patch_formation_pass")
    ].copy()
    if frame.empty:
        return _placeholder_plot(
            path,
            "Prospective intact-gate calibration",
            "No eligible intact-gate observations yet.",
        )
    predicted = pd.to_numeric(frame["prediction_mean"], errors="coerce")
    actual = pd.to_numeric(frame["observed_mean"], errors="coerce")
    jitter = np.linspace(-0.06, 0.06, len(frame)) if len(frame) > 1 else np.array([0.0])
    colors = [TEAL if bool(value) else BLUE for value in frame["formal_cohort"]]
    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.scatter(
        predicted,
        actual + jitter,
        s=75,
        c=colors,
        edgecolor="white",
        linewidth=0.7,
        alpha=0.9,
    )
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.25, 1.25)
    ax.set_yticks([0, 1], labels=["Observed fail", "Observed pass"])
    ax.set_xlabel("Frozen predicted intact-pass probability")
    ax.set_title("Proposal-time intact-gate predictions")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def _error_by_round_plot(metrics: pd.DataFrame, path: Path) -> Path:
    frame = metrics[
        (metrics["scope"].astype(str) == "round")
        & (metrics["endpoint"].astype(str) == "viability_percent")
        & pd.to_numeric(metrics["mae"], errors="coerce").notna()
    ].copy()
    if frame.empty:
        return _placeholder_plot(
            path,
            "Prospective viability error by round",
            "No round-level prospective viability metrics yet.",
        )
    frame["_sort"] = frame["round_id"].map(_round_sort_key)
    frame = frame.sort_values("_sort")
    x = np.arange(len(frame))
    colors = [TEAL if bool(value) else BLUE for value in frame["formal_cohort"]]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(x, frame["mae"], color=GRID, linewidth=2)
    ax.scatter(x, frame["mae"], c=colors, s=90, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x, labels=frame["round_id"], rotation=30, ha="right")
    ax.set_ylabel("Viability MAE (percentage points)")
    ax.set_title("Frozen-prediction error by completed round")
    for index, row in frame.reset_index(drop=True).iterrows():
        ax.annotate(
            str(row["provenance_class"]),
            (index, float(row["mae"])),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=MUTED,
        )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
