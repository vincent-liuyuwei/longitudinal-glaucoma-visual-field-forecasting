"""Generate publication-ready visual-field layout figures.

The script creates two vector figures:

1. The 52 valid Humphrey 24-2 locations from the public UWHVF coordinate
   table (the two blind-spot locations are excluded).
2. The native 59-point Bern/GRAPE layout coloured by the fixed six-region
   mapping used in the thesis.

PDF and SVG outputs are vector graphics; PNG files are exported at 600 dpi
for previews or systems that do not accept PDF/SVG uploads.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


REGION_ORDER = [
    "Supero-Nasal",
    "Supero-Temporal",
    "Macular",
    "Infero-Nasal",
    "Infero-Temporal",
    "Temporal",
]

# Fixed one-based point membership from the implementation used in the
# thesis.  The corresponding Python indices are obtained by subtracting one.
REGION_POINTS = {
    "Supero-Nasal": [2, 15, 16, 22, 25, 29, 35, 39, 43, 47, 51, 54, 57],
    "Supero-Temporal": [1, 6, 21, 31, 34, 38, 46, 50, 56],
    "Macular": [5, 8, 11, 14, 17, 18, 19, 20, 33],
    "Infero-Nasal": [3, 12, 13, 23, 26, 30, 36, 40, 44, 48, 52, 55, 58],
    "Infero-Temporal": [4, 9, 24, 32, 37, 41, 49, 53, 59],
    "Temporal": [7, 10, 27, 28, 42, 45],
}

# Colour-blind-friendly, print-safe palette.
REGION_COLOURS = {
    "Supero-Nasal": "#D55E00",
    "Supero-Temporal": "#0072B2",
    "Macular": "#CC79A7",
    "Infero-Nasal": "#009E73",
    "Infero-Temporal": "#E69F00",
    "Temporal": "#56B4E9",
}


def _base_axis(ax, xlim=(-31, 31), ylim=(-31, 31), show_fixation=True):
    """Apply a consistent right-eye visual-field coordinate style."""
    ax.axhline(0, color="#A0A0A0", linewidth=0.7, zorder=0)
    ax.axvline(0, color="#A0A0A0", linewidth=0.7, zorder=0)
    if show_fixation:
        ax.add_patch(Circle((0, 0), 1.7, facecolor="white", edgecolor="#606060",
                            linewidth=0.8, zorder=1))
        ax.text(0, 0, "fixation", ha="center", va="center", fontsize=6,
                color="#606060", zorder=2)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xticks([-30, -20, -10, 0, 10, 20, 30])
    ax.set_yticks([-30, -20, -10, 0, 10, 20, 30])
    ax.set_xlabel("Horizontal eccentricity (degrees)")
    ax.set_ylabel("Vertical eccentricity (degrees)")
    ax.grid(True, linewidth=0.35, alpha=0.35, zorder=0)
    ax.set_facecolor("#FAFAFA")


def _save(fig, stem: str):
    """Save vector and high-resolution raster versions."""
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def make_uw_hfa_52():
    """Plot the 52 non-blind-spot HFA 24-2 locations from UWHVF."""
    path = ROOT / "data" / "uwhvf-master" / "CSV" / "Coord_242.csv"
    coords = pd.read_csv(path)
    retained = coords.loc[coords["Cluster"] != 0].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(6.6, 6.3))
    _base_axis(ax)

    ax.scatter(retained["X"], retained["Y"], s=135, facecolor="#2C7FB8",
               edgecolor="white", linewidth=0.8, zorder=3)
    for i, row in retained.iterrows():
        # The displayed index is the effective 1--52 order after blind-spot
        # exclusion, not the original 1--54 LocID.
        ax.text(row["X"], row["Y"], str(i + 1), ha="center", va="center",
                fontsize=6.2, color="white", zorder=4)

    ax.set_title("Humphrey 24-2 layout: 52 valid TD locations",
                 pad=10, fontsize=12, weight="bold")
    ax.text(0.02, 0.02,
            "Two physiological blind-spot locations excluded\n"
            "Right-eye coordinate convention",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7.2,
            color="#404040",
            bbox={"facecolor": "white", "edgecolor": "#B0B0B0",
                  "linewidth": 0.5, "alpha": 0.9})
    _save(fig, "uw_hfa_24_2_52_points")


def make_native_59_regions():
    """Plot the native 59-point Bern/GRAPE layout and six fixed regions."""
    coords = pd.read_excel(ROOT / "data" / "Bern" / "x_y 1.xlsx")
    if list(coords.columns) != ["V1", "V2"] or len(coords) != 59:
        raise ValueError("Unexpected native Bern/GRAPE coordinate table")

    point_to_region = {}
    for region, points in REGION_POINTS.items():
        for point in points:
            if point in point_to_region:
                raise ValueError(f"Point {point} appears in multiple regions")
            point_to_region[point] = region
    if set(point_to_region) != set(range(1, 60)):
        raise ValueError("Six-region mapping does not cover exactly 59 points")

    fig, ax = plt.subplots(figsize=(8.0, 6.8))
    # The native 59-point table contains a central point at (0, 0), so a
    # separate fixation marker would obscure TD point 33.
    _base_axis(ax, xlim=(-31, 31), ylim=(-31, 31), show_fixation=False)

    for region in REGION_ORDER:
        points = REGION_POINTS[region]
        idx = [point - 1 for point in points]
        ax.scatter(coords.loc[idx, "V1"], coords.loc[idx, "V2"],
                   s=250, facecolor=REGION_COLOURS[region], edgecolor="white",
                   linewidth=0.85, zorder=3)

    for point, row in coords.iterrows():
        ax.text(row["V1"], row["V2"], str(point + 1), ha="center",
                va="center", fontsize=5.5, color="black", zorder=4)

    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=8,
               markerfacecolor=REGION_COLOURS[r], markeredgecolor="white",
               label=f"{r} (n={len(REGION_POINTS[r])})")
        for r in REGION_ORDER
    ]
    ax.legend(handles=handles, title="Fixed native 59-point regions",
              loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=True,
              fontsize=8, title_fontsize=8.5, borderaxespad=0)
    ax.set_title("Native Bern/GRAPE 59-point visual-field representation",
                 pad=10, fontsize=12, weight="bold")
    ax.text(0.02, 0.02,
            "Point labels are one-based TD indices\n"
            "Region assignment is fixed before model training",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=7.2,
            color="#404040",
            bbox={"facecolor": "white", "edgecolor": "#B0B0B0",
                  "linewidth": 0.5, "alpha": 0.9})
    _save(fig, "native_59_six_region_mapping")


if __name__ == "__main__":
    make_uw_hfa_52()
    make_native_59_regions()
    print(f"Wrote figures to {OUT}")
