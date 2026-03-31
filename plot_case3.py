"""Plotting script for case3 parameter study results.

Produces six figures saved to case3/plots/:
    inflow_rate.png        — absolute inflow vs time
    outflow_rate.png       — absolute outflow vs time
    flow_anomaly.png       — outflow anomaly vs no-damage baseline
    damage_evolution.png   — average dilation and friction damage vs time
    slip_tendency.png      — slip tendency vs damage scatter with time trajectory
    summary_bars.png       — final-time summary bar chart for all combinations

Usage::

    python plot_case3.py            # saves PNGs only
    python plot_case3.py --show     # also opens interactive windows
"""

from __future__ import annotations

import argparse
from pathlib import Path


import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import porepy as pp
from case3 import damage_combinations, folder_name, INITIALIZATION_TIME

CELL_SIZE = 32  # [m], used in plot labels but not for loading results (which may include multiple cell sizes)
RESULTS_FILE = "results.csv"
MODEL_NAME = "thermo"
PLOT_DIR = Path(f"case3/plots/cell_size_{CELL_SIZE}/{MODEL_NAME}/")

FONTSIZE = 16  # Global font size for all plot text

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Pick the most readable time unit for the post-init time span.
_TIME_UNITS = [
    (pp.YEAR, "years"),
    (pp.WEEK, "weeks"),
    (pp.DAY, "days"),
    (pp.HOUR, "hours"),
    (pp.MINUTE, "minutes"),
    (pp.SECOND, "seconds"),
]


def _pick_time_unit(max_seconds: float) -> tuple[float, str]:
    """Return (divisor_in_seconds, unit_label) for the largest unit where
    max_seconds / divisor >= 2, so the axis shows at least two ticks."""
    for divisor, name in _TIME_UNITS:
        if max_seconds / divisor >= 2:
            return divisor, name
    return pp.SECOND, "seconds"


# Set to False (or pass --hide-iso) to omit the iso/aniso prefix in labels.
_SHOW_ISO_TAG: bool = False


def _label(isotropic: bool, damages: list[str]) -> str:
    dmg = " + ".join(damages) if damages else "no damage"
    if _SHOW_ISO_TAG:
        tag = "iso" if isotropic else "aniso"
        return f"{tag}, {dmg}"
    return dmg


def _is_baseline(damages: list[str]) -> bool:
    return len(damages) == 0


def load_results() -> dict[tuple, pd.DataFrame]:
    """Load all available CSV results, keyed by (isotropic, tuple(damages))."""
    data: dict[tuple, pd.DataFrame] = {}
    for isotropic, damages in damage_combinations:
        path = (
            Path(folder_name(isotropic, damages, CELL_SIZE, MODEL_NAME)) / RESULTS_FILE
        )
        if path.exists():
            df = pd.read_csv(path)
            df = df[df["time"] >= INITIALIZATION_TIME].reset_index(drop=True)
            # time_plot is set after all data is loaded so units are consistent
            df["_time_offset_s"] = df["time"] - INITIALIZATION_TIME
            data[(isotropic, tuple(damages))] = df
        else:
            print(f"  [missing] {path}")
    return data


def _add_time_plot_column(data: dict) -> str:
    """Add 'time_plot' column to every dataframe and return the unit label."""
    if not data:
        return "seconds"
    max_offset = max(df["_time_offset_s"].max() for df in data.values())
    divisor, unit_label = _pick_time_unit(float(max_offset))
    for df in data.values():
        df["time_plot"] = df["_time_offset_s"] / divisor
    return unit_label


# Consistent color/marker per combination
_COMBOS = [(iso, tuple(dmg)) for iso, dmg in damage_combinations]
_CMAP = plt.colormaps["tab10"].resampled(len(_COMBOS))

STYLES: dict[tuple, dict] = {
    key: {"color": _CMAP(i), "marker": ["o", "s", "^", "D"][i % 4], "lw": 1.8}
    for i, key in enumerate(_COMBOS)
}
# Override baseline (no-damage) to black so it stands out as the reference.
for _key in list(STYLES):
    if _is_baseline(list(_key[1])):
        STYLES[_key]["color"] = "black"

# ---------------------------------------------------------------------------
# Plot 1+2 — Flow rates and anomaly
# ---------------------------------------------------------------------------


def plot_flow_rates(
    data: dict, ax_in: plt.Axes, ax_out: plt.Axes, time_unit: str = "days"
) -> None:
    for key, df in data.items():
        isotropic, damages = key
        s = STYLES[key]
        ls = "-"  # "--" if _is_baseline(list(damages)) else "-"
        kw = dict(
            color=s["color"],
            marker=s["marker"],
            ls=ls,
            lw=s["lw"],
            ms=5,
            label=_label(isotropic, list(damages)),
        )
        ax_in.plot(df["time_plot"], df["inflow_rate"], **kw)
        ax_out.plot(df["time_plot"], df["outflow_rate"], **kw)

    for ax, title in [(ax_in, "Inflow"), (ax_out, "Outflow")]:
        ax.set_xlabel(f"Time [{time_unit}]", fontsize=FONTSIZE)
        ax.set_ylabel("Flow rate [m³/s]", fontsize=FONTSIZE)
        ax.set_title(title, fontsize=FONTSIZE)
        ax.tick_params(labelsize=FONTSIZE)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=FONTSIZE)


def plot_flow_anomaly(data: dict, ax: plt.Axes, time_unit: str = "days") -> None:
    """(Q_damaged - Q_base) / Q_base × 100 [%] for each damaged case."""
    baselines: dict[bool, pd.DataFrame] = {
        iso: df for (iso, dmg), df in data.items() if _is_baseline(list(dmg))
    }

    plotted = False
    for key, df in data.items():
        isotropic, damages = key
        if _is_baseline(list(damages)) or isotropic not in baselines:
            continue
        base = baselines[isotropic]
        # Interpolate baseline outflow onto this run's time points (adaptive
        # time stepping means time_index values differ between combinations).
        base_ref = np.interp(
            df["time"].values, base["time"].values, base["outflow_rate"].values
        )
        nonzero = np.abs(base_ref) > 1e-20
        anomaly = np.where(
            nonzero, (df["outflow_rate"].values - base_ref) / base_ref * 100, np.nan
        )

        s = STYLES[key]
        ax.plot(
            df.loc[nonzero, "time_plot"],
            anomaly[nonzero],
            color=s["color"],
            marker=s["marker"],
            lw=s["lw"],
            ms=5,
            label=_label(isotropic, list(damages)),
        )
        plotted = True

    ax.axhline(0, color="k", lw=0.8, ls=":")
    ax.set_xlabel(f"Time [{time_unit}]", fontsize=FONTSIZE)
    ax.set_ylabel("Outflow anomaly [%]", fontsize=FONTSIZE)
    ax.set_title("Outflow change vs no-damage baseline", fontsize=FONTSIZE)
    ax.tick_params(labelsize=FONTSIZE)
    ax.grid(True, alpha=0.3)
    if plotted:
        ax.legend(fontsize=FONTSIZE)


# ---------------------------------------------------------------------------
# Plot 3 — Damage evolution
# ---------------------------------------------------------------------------


def plot_damage(
    data: dict, ax_dil: plt.Axes, ax_fric: plt.Axes, time_unit: str = "days"
) -> None:
    has_dil = has_fric = False
    for key, df in data.items():
        isotropic, damages = key
        s = STYLES[key]
        kw = dict(
            color=s["color"],
            marker=s["marker"],
            lw=s["lw"],
            ms=5,
            label=_label(isotropic, list(damages)),
        )

        col = "average_dilation_damage"
        if col in df.columns and not df[col].isna().all():
            ax_dil.plot(df["time_plot"], df[col], **kw)
            has_dil = True

        col = "average_friction_damage"
        if col in df.columns and not df[col].isna().all():
            ax_fric.plot(df["time_plot"], df[col], **{**kw, "label": "_"})
            has_fric = True

    for ax, title, active in [
        (ax_dil, "Average dilation damage", has_dil),
        (ax_fric, "Average friction damage", has_fric),
    ]:
        ax.set_xlabel(f"Time [{time_unit}]")
        ax.set_ylabel("Average damage [-]")
        ax.set_title(title, fontsize=FONTSIZE)
        ax.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        if active and ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=FONTSIZE)


# ---------------------------------------------------------------------------
# Plot 3b — Combined damage evolution (both types on one axes)
# ---------------------------------------------------------------------------

# Fixed line style and marker per damage type so the visual encoding is
# independent of which combination is shown.
_DAMAGE_STYLE = {
    "average_dilation_damage": {"ls": "-", "marker": "o", "label_suffix": "$d^d$"},
    "average_friction_damage": {"ls": "--", "marker": "s", "label_suffix": "$d^f$"},
}


def plot_damage_combined(data: dict, ax: plt.Axes, time_unit: str = "days") -> None:
    """Both damage types on one axes.

    Color encodes the damage combination; line style + marker encode the damage
    type (solid/circle = dilation, dashed/square = friction).
    """
    plotted = False
    for key, df in data.items():
        isotropic, damages = key
        s = STYLES[key]
        for col, ds in _DAMAGE_STYLE.items():
            if col not in df.columns or df[col].isna().all():
                continue
            combo_label = _label(isotropic, list(damages))
            ax.plot(
                df["time_plot"],
                df[col],
                color=s["color"],
                ls=ds["ls"],
                marker=ds["marker"],
                lw=s["lw"],
                ms=5,
                label=f"{combo_label} ({ds['label_suffix']})",
            )
            plotted = True

    ax.set_xlabel(f"Time [{time_unit}]", fontsize=FONTSIZE)
    ax.set_ylabel("Average damage [-]", fontsize=FONTSIZE)
    ax.set_title("Damage evolution", fontsize=FONTSIZE)
    ax.tick_params(labelsize=FONTSIZE)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    if plotted and ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=FONTSIZE)


# ---------------------------------------------------------------------------
# Plot 4 — Slip tendency vs damage scatter
# ---------------------------------------------------------------------------


def plot_scatter(data: dict, axes: list[plt.Axes], time_unit: str = "days") -> None:
    damage_cols = ["average_dilation_damage", "average_friction_damage"]
    titles = ["Dilation damage", "Friction damage"]

    # Collect global time range for consistent colormap
    all_times = np.concatenate([df["time_plot"].values for df in data.values()])
    t_min, t_max = all_times.min(), all_times.max()
    norm = plt.Normalize(t_min, t_max)

    for ax, dcol, dtitle in zip(axes, damage_cols, titles):
        for key, df in data.items():
            isotropic, damages = key
            if dcol not in df.columns or df[dcol].isna().all():
                continue

            x = df[dcol].values
            y = df["average_slip_tendency"].values
            t = df["time_plot"].values
            color = STYLES[key]["color"]

            ax.scatter(
                x,
                y,
                c=t,
                cmap="viridis",
                norm=norm,
                s=50,
                zorder=5,
                edgecolors=color,
                linewidths=1.2,
            )

            # Trajectory arrows
            for i in range(len(x) - 1):
                dx, dy = x[i + 1] - x[i], y[i + 1] - y[i]
                if abs(dx) + abs(dy) < 1e-12:
                    continue
                ax.annotate(
                    "",
                    xy=(x[i + 1], y[i + 1]),
                    xytext=(x[i], y[i]),
                    arrowprops=dict(arrowstyle="->", color=color, lw=1.2),
                )

            # Label each trajectory with one legend entry (invisible scatter)
            ax.scatter(
                [],
                [],
                c=[color],
                s=40,
                edgecolors=color,
                label=_label(isotropic, list(damages)),
            )

        ax.set_xlabel(f"{dtitle} [-]")
        ax.set_ylabel("Average slip tendency [-]")
        ax.set_title(f"Slip tendency vs {dtitle.lower()}", fontsize=FONTSIZE)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=FONTSIZE)

    fig = axes[0].get_figure()
    sm = cm.ScalarMappable(cmap="viridis", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=axes, label=f"Time [{time_unit}]", shrink=0.8)


# ---------------------------------------------------------------------------
# Plot 7 — Final-time summary bar chart
# ---------------------------------------------------------------------------


# Columns whose values are already dimensionless and O(1): keep unscaled by
# default so their true magnitude is visible without normalisation.
_SUMMARY_UNSCALED_COLS = {
    "average_slip_tendency",
    "average_dilation_damage",
    "average_friction_damage",
}


def plot_summary_bars(
    data: dict,
    ax: plt.Axes,
    normalize: bool = True,
    skip_empty_cols: bool = True,
) -> None:
    """Bar chart of final-time metrics.

    Parameters
    ----------
    normalize:
        When *True* (default) every metric is divided by its own cross-
        combination maximum so all bars sit on a common [0, 1] axis.  Columns
        listed in ``_SUMMARY_UNSCALED_COLS`` (slip tendency and damage, which
        are already dimensionless and O(1)) are **always** left unscaled;
        only outflow and tangential jump are normalised.
        When *False* those two metrics are also left unscaled (their raw
        display value is used), which makes the bar heights directly
        comparable only if the magnitudes happen to be similar.
    skip_empty_cols:
        When *True* (default) any metric column that is NaN or zero for **all**
        combinations is omitted from the chart entirely.
    """
    # (column, display_label, unit_scale, unit_string)
    # unit_scale converts the raw SI value to a human-readable unit (e.g.
    # m → mm) before any optional normalisation.
    metrics = [
        ("outflow_rate", "Outflow", 1.0, "m³/s"),
        ("average_slip_tendency", "Slip tend.", 1.0, "-"),
        ("average_dilation_damage", "Dil. damage", 1.0, "-"),
        ("average_friction_damage", "Fric. damage", 1.0, "-"),
        ("average_tangential_jump", "Tang. jump", 1e3, "mm"),
    ]

    labels, metric_vals = [], {m[0]: [] for m in metrics}
    for key, df in data.items():
        isotropic, damages = key
        labels.append(_label(isotropic, list(damages)))
        final = df.iloc[-1]
        for col, *_ in metrics:
            v = final.get(col, np.nan)
            metric_vals[col].append(float(v) if not pd.isna(v) else 0.0)

    # Per-metric normalisation divisor (1.0 means no normalisation).
    norm_maxes: dict[str, float] = {}
    for col, *_ in metrics:
        raw = np.array(metric_vals[col])
        if normalize and col not in _SUMMARY_UNSCALED_COLS:
            norm_maxes[col] = float(np.nanmax(np.abs(raw))) or 1.0
        else:
            norm_maxes[col] = 1.0

    # Drop metrics that are entirely absent (all NaN or zero) if requested.
    if skip_empty_cols:
        metrics = [
            m
            for m in metrics
            if np.any(np.array(metric_vals[m[0]]) != 0)
            and not np.all(np.isnan(metric_vals[m[0]]))
        ]

    n_combos, n_metrics = len(labels), len(metrics)
    x = np.arange(n_combos)
    width = 0.75 / n_metrics

    for i, (col, base_label, unit_scale, unit_str) in enumerate(metrics):
        raw = np.array(metric_vals[col]) * unit_scale
        divisor = norm_maxes[col] * unit_scale
        vals = raw / divisor if divisor != 0 else raw

        if normalize and col not in _SUMMARY_UNSCALED_COLS:
            # Show what the bar height was normalised by
            max_display = norm_maxes[col] * unit_scale
            if max_display != 0:
                exp = int(np.floor(np.log10(abs(max_display))))
                divisor_str = (
                    f"{max_display:.3g}" if -3 <= exp <= 3 else f"{max_display:.2e}"
                )
            else:
                divisor_str = "1"
            legend_label = f"{base_label} [{unit_str}] (÷{divisor_str})"
        else:
            legend_label = f"{base_label} [{unit_str}]"

        offset = (i - n_metrics / 2 + 0.5) * width
        ax.bar(x + offset, vals, width * 0.75, label=legend_label)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=FONTSIZE)
    if normalize:
        ax.set_ylabel("Value (dimensionless metrics unscaled; others ÷ max)")
    else:
        ax.set_ylabel("Value (unscaled, mixed units)")
    ax.set_title("Final-time summary", fontsize=FONTSIZE)
    ax.legend(fontsize=FONTSIZE, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main(show: bool = False, normalize: bool = True, show_iso_tag: bool = True) -> None:
    print("Loading results...")
    data = load_results()

    if not data:
        print("No results found. Run case3.py first.")
        return

    global _SHOW_ISO_TAG
    _SHOW_ISO_TAG = show_iso_tag

    found = [_label(k[0], list(k[1])) for k in data]
    print(f"Loaded {len(data)} result(s): {', '.join(found)}")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    time_unit = _add_time_plot_column(data)
    print(f"  Time unit: {time_unit}")
    suptitle = False
    # Figure A: inflow rate
    fig_in, ax_in = plt.subplots(figsize=(6, 5))
    if suptitle:
        fig_in.suptitle("Case 3 — Inflow Rate", fontsize=FONTSIZE + 1)
    # Figure B: outflow rate
    fig_out, ax_out = plt.subplots(figsize=(6, 5))
    if suptitle:
        fig_out.suptitle("Case 3 — Outflow Rate", fontsize=FONTSIZE + 1)
    plot_flow_rates(data, ax_in, ax_out, time_unit)
    fig_in.tight_layout()
    fig_in.savefig(PLOT_DIR / "inflow_rate.png", dpi=150)
    print(f"  → {PLOT_DIR / 'inflow_rate.png'}")
    fig_out.tight_layout()
    fig_out.savefig(PLOT_DIR / "outflow_rate.png", dpi=150)
    print(f"  → {PLOT_DIR / 'outflow_rate.png'}")

    # Figure C: outflow anomaly
    fig_anom, ax_anom = plt.subplots(figsize=(6, 5))
    if suptitle:
        fig_anom.suptitle("Case 3 — Outflow Anomaly", fontsize=FONTSIZE + 1)
    plot_flow_anomaly(data, ax_anom, time_unit)
    fig_anom.tight_layout()
    fig_anom.savefig(PLOT_DIR / "flow_anomaly.png", dpi=150)
    print(f"  → {PLOT_DIR / 'flow_anomaly.png'}")

    # Figure 3: damage evolution (side-by-side)
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 4))
    if suptitle:
        fig3.suptitle("Case 3 — Damage Evolution", fontsize=FONTSIZE + 1)
    plot_damage(data, axes3[0], axes3[1], time_unit)
    fig3.tight_layout()
    fig3.savefig(PLOT_DIR / "damage_evolution.png", dpi=150)
    print(f"  → {PLOT_DIR / 'damage_evolution.png'}")

    # Figure 3b: combined damage evolution (single axes)
    fig3b, ax3b = plt.subplots(figsize=(6, 5))
    if suptitle:
        fig3b.suptitle("Case 3 — Damage Evolution (combined)", fontsize=FONTSIZE + 1)
    plot_damage_combined(data, ax3b, time_unit)
    fig3b.tight_layout()
    fig3b.savefig(PLOT_DIR / "damage_combined.png", dpi=150)
    print(f"  → {PLOT_DIR / 'damage_combined.png'}")

    # Figure 4: slip tendency vs damage scatter
    fig4, axes4 = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    if suptitle:
        fig4.suptitle(
            "Case 3 — Slip Tendency vs Damage (arrows = time direction)",
            fontsize=FONTSIZE + 1,
        )
    plot_scatter(data, list(axes4), time_unit)
    fig4.savefig(PLOT_DIR / "slip_tendency.png", dpi=150)
    print(f"  → {PLOT_DIR / 'slip_tendency.png'}")

    # Figure 7: summary bars
    fig7, ax7 = plt.subplots(figsize=(10, 5))
    if suptitle:
        fig7.suptitle("Case 3 — Final-Time Summary", fontsize=FONTSIZE + 1)
    plot_summary_bars(data, ax7, normalize=normalize, skip_empty_cols=True)
    fig7.tight_layout()
    fig7.savefig(PLOT_DIR / "summary_bars.png", dpi=150)
    print(f"  → {PLOT_DIR / 'summary_bars.png'}")

    if show:
        plt.show()
    else:
        plt.close("all")


def watch(
    interval: int = 60, normalize: bool = True, show_iso_tag: bool = True
) -> None:
    """Re-plot every *interval* seconds until interrupted."""
    import time

    print(f"Watch mode — refreshing every {interval}s. Ctrl+C to stop.")
    while True:
        print(f"\n[{time.strftime('%H:%M:%S')}] Updating plots...")
        try:
            main(show=False, normalize=normalize, show_iso_tag=show_iso_tag)
        except Exception as exc:
            print(f"  (error: {exc})")
        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--show", action="store_true", help="Open interactive windows")
    parser.add_argument(
        "--hide-iso",
        dest="show_iso_tag",
        action="store_false",
        default=True,
        help="Omit the iso/aniso prefix from all labels",
    )
    parser.add_argument(
        "--no-normalize",
        dest="normalize",
        action="store_false",
        default=True,
        help="Leave slip tendency and damage unscaled in the summary bar chart",
    )
    parser.add_argument(
        "--watch",
        metavar="SECONDS",
        nargs="?",
        const=60,
        type=int,
        help="Re-plot every SECONDS seconds (default: 60)",
    )
    args = parser.parse_args()
    if args.watch is not None:
        watch(
            interval=args.watch,
            normalize=args.normalize,
            show_iso_tag=args.show_iso_tag,
        )
    else:
        main(show=args.show, normalize=args.normalize, show_iso_tag=args.show_iso_tag)
