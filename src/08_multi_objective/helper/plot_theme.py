"""V2-owned plotting themes; no runtime dependency on the legacy project."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt


@dataclass(frozen=True)
class PlotTheme:
    page_background: str
    axes_background: str
    grid: str
    text: str
    muted: str
    blue: str
    gold: str
    teal: str
    coral: str
    purple: str
    sky: str
    slate: str


THEMES = {
    "v2_compat": PlotTheme(
        page_background="#f7f2e8",
        axes_background="#fffdf8",
        grid="#d8d0c1",
        text="#2d2a26",
        muted="#8a8175",
        blue="#4c78a8",
        gold="#d7a44c",
        teal="#4f8f6b",
        coral="#d96c5f",
        purple="#8e6c8a",
        sky="#76a5c5",
        slate="#6c7a89",
    ),
    "legacy_inspired_publication": PlotTheme(
        page_background="#ffffff",
        axes_background="#ffffff",
        grid="#d7d7d7",
        text="#242424",
        muted="#6b6b6b",
        blue="#0072B2",
        gold="#E69F00",
        teal="#009E73",
        coral="#D55E00",
        purple="#CC79A7",
        sky="#56B4E9",
        slate="#777777",
    ),
}


def get_plot_theme(profile: str = "v2_compat") -> PlotTheme:
    try:
        return THEMES[profile]
    except KeyError as error:
        raise ValueError(
            f"Unknown V2 plot theme {profile!r}; choose one of {sorted(THEMES)}."
        ) from error


def apply_plot_theme(profile: str = "v2_compat") -> PlotTheme:
    """Apply a named theme and return its immutable color definition."""

    theme = get_plot_theme(profile)
    plt.rcParams.update(
        {
            "font.size": 10.5,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "axes.facecolor": theme.axes_background,
            "axes.edgecolor": theme.grid,
            "axes.labelcolor": theme.text,
            "xtick.color": theme.text,
            "ytick.color": theme.text,
            "text.color": theme.text,
            "figure.facecolor": theme.page_background,
            "savefig.facecolor": theme.page_background,
            "grid.color": theme.grid,
            "grid.alpha": 0.70,
            "axes.grid": True,
            "axes.axisbelow": True,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return theme
