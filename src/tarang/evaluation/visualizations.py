"""Visualization routines for TARANG Stage 3 model evaluation.

Generates publication-quality figures matching the established reference aesthetics:
- `performance.png`: Error boxplots comparing all benchmark models
- `best_model.png`: Histogram density of HGBR error (all sessions vs per-vehicle)
- `performance_over_features.png`: Dependence of error on user charging characteristics with OLS fits
- `model_agreement.png`: Pairwise model MAE agreement with inset zoom
- `ae_over_time_hist.png`: Temporal degradation histogram (relative change in MAE)
- `seasonality.png`: Monthly average MAPE time series
- `shap_values.png`: TreeSHAP global and participant-level feature importances
- `same_day_overnight_comparison.png`: Subgroup performance comparison
- `heatmap-best-{0,1,2}.png`, `heatmap-worst-{0,1,2}.png`: Charging occupancy heatmaps
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm

from tarang.config import TARGET_DURATION, TARGET_ENERGY
from tarang.evaluation.explainability import FEATURE_DISPLAY_MAP

logger = logging.getLogger("tarang.evaluation.visualizations")

PAGE_WIDTH_FULL: float = 6.84386  # inches
PAGE_WIDTH_COLUMN: float = 3.29736  # inches
GOLDEN_RATIO: float = 1.618
COLOR_MARKER: str = "C3"


def save_figure(fig: plt.Figure, save_path: Path, dpi: int = 300) -> None:
    """Save matplotlib figure and close handle cleanly."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure to %s", save_path)


def _make_metric_legend(colors: List[str], labels: List[str]) -> List[Line2D]:
    """Construct custom legend handles matching reference style."""
    handles = []
    for c, l in zip(colors, labels):
        handles.append(
            Line2D(
                [],
                [],
                marker="s",
                label=l,
                color=c,
                markerfacecolor=c,
                markeredgewidth=0,
                linewidth=0,
                markersize=7,
            )
        )
    handles.append(
        Line2D(
            [],
            [],
            marker="^",
            label="Mean",
            color=COLOR_MARKER,
            markerfacecolor=COLOR_MARKER,
            markeredgewidth=1,
            linewidth=0,
            markersize=5,
        )
    )
    handles.append(
        Line2D(
            [],
            [],
            marker="",
            label="Median",
            color=COLOR_MARKER,
            markerfacecolor=COLOR_MARKER,
            markeredgewidth=0,
            linewidth=1.2,
            markersize=5,
        )
    )
    handles.append(
        Line2D(
            [],
            [],
            marker="x",
            label="SMAPE",
            color=COLOR_MARKER,
            markerfacecolor=COLOR_MARKER,
            markeredgewidth=1.2,
            linewidth=0,
            markersize=5,
        )
    )
    return handles


def plot_performance_boxplots(
    performance_participants: pd.DataFrame,
    performance_all: pd.Series,
    target: str,
    save_path: Path,
) -> None:
    """Generate side-by-side error boxplots comparing all benchmark models."""
    unit = "h" if target == TARGET_DURATION else "kWh"
    width = 0.25
    step = width + 0.1
    dataset = "Test"

    # Identify models
    all_models = performance_all.index.get_level_values(0).unique().tolist()
    models = [m for m in all_models if m != "number_sessions"]

    fig, ax = plt.subplots(
        2,
        1,
        sharex="col",
        figsize=(PAGE_WIDTH_COLUMN, PAGE_WIDTH_COLUMN * 1.3),
    )

    y_labels = [f"Absolute Error ({unit})", "Absolute Percentage Error (%)"]

    for i, model in enumerate(models):
        pos_all = i - 0.5 * step
        pos_veh = pos_all + step

        # Panel 0: Absolute Error (ae)
        ax[0].bxp(
            [
                {
                    "label": model,
                    "whislo": performance_all.get((model, dataset, "wl-ae"), np.nan),
                    "q1": performance_all.get((model, dataset, "q1-ae"), np.nan),
                    "med": performance_all.get((model, dataset, "q2-ae"), np.nan),
                    "q3": performance_all.get((model, dataset, "q3-ae"), np.nan),
                    "whishi": performance_all.get((model, dataset, "wh-ae"), np.nan),
                    "mean": performance_all.get((model, dataset, "mae"), np.nan),
                    "fliers": [],
                }
            ],
            positions=[pos_all],
            widths=width,
            showfliers=False,
            showmeans=True,
            patch_artist=True,
            boxprops={"facecolor": "C0", "edgecolor": "C0"},
            medianprops={"color": COLOR_MARKER},
            meanprops={"markerfacecolor": COLOR_MARKER, "markeredgecolor": "none", "markersize": 5},
            whiskerprops={"color": "C0"},
            capprops={"color": "C0"},
            zorder=3,
        )

        # Vehicle-level MAE distribution
        if (model, dataset, "mae") in performance_participants.columns:
            veh_mae = performance_participants[(model, dataset, "mae")].dropna()
            ax[0].boxplot(
                veh_mae,
                positions=[pos_veh],
                widths=width,
                showfliers=False,
                showmeans=True,
                patch_artist=True,
                boxprops={"facecolor": "C1", "edgecolor": "C1"},
                medianprops={"color": COLOR_MARKER},
                meanprops={"markerfacecolor": COLOR_MARKER, "markeredgecolor": "none", "markersize": 5},
                whiskerprops={"color": "C1"},
                capprops={"color": "C1"},
                zorder=3,
            )

        # Panel 1: Percentage Error (ape)
        ax[1].bxp(
            [
                {
                    "label": model,
                    "whislo": performance_all.get((model, dataset, "wl-ape"), np.nan),
                    "q1": performance_all.get((model, dataset, "q1-ape"), np.nan),
                    "med": performance_all.get((model, dataset, "q2-ape"), np.nan),
                    "q3": performance_all.get((model, dataset, "q3-ape"), np.nan),
                    "whishi": performance_all.get((model, dataset, "wh-ape"), np.nan),
                    "mean": performance_all.get((model, dataset, "mape"), np.nan),
                    "fliers": [],
                }
            ],
            positions=[pos_all],
            widths=width,
            showfliers=False,
            showmeans=True,
            patch_artist=True,
            boxprops={"facecolor": "C0", "edgecolor": "C0"},
            medianprops={"color": COLOR_MARKER},
            meanprops={"markerfacecolor": COLOR_MARKER, "markeredgecolor": "none", "markersize": 5},
            whiskerprops={"color": "C0"},
            capprops={"color": "C0"},
            zorder=3,
        )

        # SMAPE marker
        smape_val = performance_all.get((model, dataset, "smape"), np.nan)
        if not pd.isna(smape_val):
            ax[1].plot(
                [pos_all],
                [smape_val],
                marker="x",
                color=COLOR_MARKER,
                zorder=9,
                linewidth=0,
                markersize=5,
                markeredgewidth=1.2,
            )

        # Vehicle-level MAPE distribution
        if (model, dataset, "mape") in performance_participants.columns:
            veh_mape = performance_participants[(model, dataset, "mape")].dropna()
            ax[1].boxplot(
                veh_mape,
                positions=[pos_veh],
                widths=width,
                showfliers=False,
                showmeans=True,
                patch_artist=True,
                boxprops={"facecolor": "C1", "edgecolor": "C1"},
                medianprops={"color": COLOR_MARKER},
                meanprops={"markerfacecolor": COLOR_MARKER, "markeredgecolor": "none", "markersize": 5},
                whiskerprops={"color": "C1"},
                capprops={"color": "C1"},
                zorder=3,
            )

    # Styling and legends
    legend_handles = _make_metric_legend(["C0", "C1"], ["Error on all sessions", "Mean on each vehicle"])
    ax[0].legend(
        handles=legend_handles,
        frameon=False,
        fontsize=7,
        loc="upper left",
        bbox_to_anchor=(-0.01, 1.05),
    )

    for i, a in enumerate(ax):
        a.set_ylabel(y_labels[i], fontsize=8)
        ylim_top = a.get_ylim()[1]
        a.set_ylim(-0.02 * ylim_top, ylim_top)
        a.grid(axis="y", linestyle="--", alpha=0.4, zorder=0)

    tick_labels = [
        m.removesuffix("Regressor").removesuffix("Regression").replace("HistGradientBoosting", "HGBR")
        for m in models
    ]
    ax[-1].set_xticks(range(len(models)))
    ax[-1].set_xticklabels(tick_labels, rotation=15, fontsize=7)
    ax[-1].set_xlim(-0.5, len(models) - 0.5)

    fig.tight_layout()
    save_figure(fig, save_path)


def plot_best_model_distribution(
    predictions: pd.DataFrame,
    performance_participants: pd.DataFrame,
    target: str,
    save_path: Path,
) -> None:
    """Plot histograms of best model error density per session and per vehicle."""
    unit = "h" if target == TARGET_DURATION else "kWh"
    model_name = "HistGradientBoostingRegressor"
    score_name = "MAE"
    alpha = 0.8

    preds = predictions[predictions["dataset"] == "Test"].copy()
    ae = np.abs(preds[model_name] - preds[target]).dropna().values

    veh_mae = performance_participants.loc[:, (model_name, "Test", "mae")].dropna().values

    bins = np.arange(0, 15, 0.66)

    fig, ax = plt.subplots(
        figsize=(PAGE_WIDTH_COLUMN, PAGE_WIDTH_COLUMN / GOLDEN_RATIO)
    )

    hist_ae, _ = np.histogram(ae, bins=bins)
    if hist_ae.sum() > 0:
        hist_ae = hist_ae / hist_ae.sum()
        ax.bar(
            bins,
            np.append(hist_ae, hist_ae[-1]),
            color="C0",
            label="AE on all sessions",
            alpha=alpha,
            width=bins[1] - bins[0],
            align="edge",
            linewidth=0.5,
            edgecolor="white",
            zorder=0,
        )
        ax.axvline(np.mean(ae), color="C0", linestyle="--", linewidth=1.2, zorder=2)
        ax.axvline(np.median(ae), color="C0", linestyle=":", linewidth=1.2, zorder=2)

    dist_veh, _ = np.histogram(veh_mae, bins=bins)
    if dist_veh.sum() > 0:
        dist_veh = dist_veh / dist_veh.sum()
        ax.bar(
            bins,
            np.append(dist_veh, dist_veh[-1]),
            color="C1",
            label=f"{score_name} for each vehicle",
            alpha=alpha,
            width=bins[1] - bins[0],
            align="edge",
            linewidth=0.5,
            edgecolor="white",
            zorder=1,
        )
        ax.axvline(np.mean(veh_mae), color="C1", linestyle="--", linewidth=1.2, zorder=2)
        ax.axvline(np.median(veh_mae), color="C1", linestyle=":", linewidth=1.2, zorder=2)

    handles, labels = ax.get_legend_handles_labels()
    handles += [
        Line2D([], [], linestyle="--", label="Mean", color="gray"),
        Line2D([], [], linestyle=":", label="Median", color="gray"),
    ]
    labels += ["Mean", "Median"]

    legend = ax.legend(
        handles,
        labels,
        frameon=False,
        fontsize=6.5,
        labelspacing=0.4,
        loc="upper right",
        bbox_to_anchor=(1.01, 1.02),
        markerfirst=False,
    )
    for t in legend.get_texts():
        t.set_ha("right")

    ax.set_xlabel(f"Absolute Error ({unit})", fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.set_xlim(0, bins[-1])
    ax.set_ylim(0, None)

    fig.tight_layout()
    save_figure(fig, save_path)


def plot_performance_over_features(
    vehicle_characteristics: Dict[str, Any],
    target: str,
    save_path: Path,
) -> None:
    """Plot dependence of HGBR MdAPE on participant characteristics with OLS fits."""
    keys = ["duration_variability", "battery_size", "start_hour"]
    titles = ["(a)", "(b)", "(c)"] if target == TARGET_DURATION else ["(d)", "(e)", "(f)"]
    target_label = "Session duration" if target == TARGET_DURATION else "Charged energy"

    fig, ax = plt.subplots(
        1,
        len(keys),
        figsize=(PAGE_WIDTH_FULL, PAGE_WIDTH_FULL / GOLDEN_RATIO / 2),
        sharey=True,
        tight_layout=True,
    )

    ax[0].set_ylabel(f"{target_label}\nMdAPE (%)", fontsize=8)

    for i, key in enumerate(keys):
        info = vehicle_characteristics[key]
        x = info["x"]
        y = info["y"]
        name = info["name"]

        ax[i].set_title(titles[i], weight="bold", fontsize=9)
        ax[i].scatter(x, y, s=2, color="C0", alpha=0.6)
        ax[i].set_xlabel(name, fontsize=8)
        ax[i].set_ylim(0, 150)

        if i == 2:
            ax[i].set_xlim(10, 22)

        # Plot OLS fit line if valid
        if not np.isnan(info["slope"]):
            slope = info["slope"]
            intercept = info["intercept"]
            r2 = info["r2"]
            p_val = info["p_value"]
            line_color = COLOR_MARKER if info["significant"] else "gray"

            x_smooth = np.linspace(x.min(), x.max(), 100)
            y_smooth = slope * x_smooth + intercept
            label_text = f"a = {slope:.3f}\nb = {intercept:.3f}\n$r^2$ = {r2:.3f}\np = {p_val:.3f}"
            ax[i].plot(x_smooth, y_smooth, color=line_color, label=label_text, linewidth=1.2)
            ax[i].legend(loc="upper left", fontsize=6.5, frameon=False)

        # 2D KDE overlay
        if len(x) > 5:
            try:
                sns.kdeplot(x=x, y=y, ax=ax[i], color="grey", linewidths=0.5, levels=4)
            except Exception as e:
                logger.debug("KDE plot skipped: %s", e)

    save_figure(fig, save_path)


def plot_model_agreement(
    performance_participants: pd.DataFrame,
    target: str,
    save_path: Path,
) -> None:
    """Plot pairwise agreement of participant MAE across benchmark models."""
    unit = "h" if target == TARGET_DURATION else "kWh"
    score = "mae"
    dataset = "Test"
    models = ["HistGradientBoostingRegressor", "LinearRegression", "QuantileRegressor"]

    fig, ax = plt.subplots(
        1, 1, figsize=(PAGE_WIDTH_COLUMN, PAGE_WIDTH_COLUMN / GOLDEN_RATIO)
    )

    axins = ax.inset_axes(
        [0.4, 0.08, 0.57, 0.57],
        xlim=(0, 20),
        ylim=(0, 20),
        xticklabels=[],
        yticklabels=[],
        xticks=[0, 10, 20],
        yticks=[0, 10, 20],
    )
    axins.patch.set_alpha(0.5)

    color_matrix = [[], ["C0"], ["C3", "C1"]]

    for i, model_1 in enumerate(models):
        for j, model_2 in enumerate(models):
            if i <= j:
                continue

            col1 = (model_1, dataset, score)
            col2 = (model_2, dataset, score)
            if col1 in performance_participants.columns and col2 in performance_participants.columns:
                valid = performance_participants[col1].notna() & performance_participants[col2].notna()
                x = performance_participants.loc[valid, col1]
                y = performance_participants.loc[valid, col2]

                m1 = model_1.removesuffix("Regressor").removesuffix("Regression").replace("HistGradientBoosting", "HGBR")
                m2 = model_2.removesuffix("Regressor").removesuffix("Regression").replace("HistGradientBoosting", "HGBR")

                c = color_matrix[i][j]
                ax.scatter(x, y, s=2, c=c, label=f"{m1} vs {m2}", alpha=0.9, linewidth=0)
                axins.scatter(x, y, s=2, c=c, alpha=0.9, linewidth=0)

    # Reference diagonal lines
    ax.plot([0, 1], [0, 1], transform=ax.transAxes, color="gray", linestyle="--", alpha=0.5, zorder=0)
    axins.plot([0, 1], [0, 1], transform=axins.transAxes, color="gray", linestyle="--", alpha=0.5)
    ax.indicate_inset_zoom(axins, edgecolor="black", alpha=0.8)

    ax.set_xlabel(f"Model 1 MAE ({unit})", fontsize=8)
    ax.set_ylabel(f"Model 2 MAE ({unit})", fontsize=8)
    ax.legend(
        markerscale=3,
        frameon=False,
        fontsize=6.5,
        labelspacing=0.4,
        loc="upper left",
        bbox_to_anchor=(-0.02, 1.02),
    )

    fig.tight_layout()
    save_figure(fig, save_path)


def plot_error_over_time_histogram(
    error_over_time_dict: Dict[str, Any],
    save_path: Path,
) -> None:
    """Plot histogram of relative change in MAE from first to second half of test split."""
    data = error_over_time_dict["relative_changes"]
    bins = np.linspace(-2, 2, 21)

    fig, ax = plt.subplots(
        1, 1, figsize=(PAGE_WIDTH_COLUMN, PAGE_WIDTH_COLUMN / GOLDEN_RATIO)
    )

    hist, _ = np.histogram(data, bins=bins)
    ax.bar(
        bins,
        np.append(hist, hist[-1]),
        color="C0",
        label="AE on all sessions",
        width=bins[1] - bins[0],
        align="edge",
        linewidth=0.5,
        edgecolor="white",
        zorder=0,
    )

    ax.set_xlabel("Relative change in MAE from first to second half of test set", fontsize=7.5)
    ax.set_ylabel("Vehicles", fontsize=8)
    ax.set_xlim(-2, 2)

    fig.tight_layout()
    save_figure(fig, save_path)


def plot_seasonality(
    seasonality_dict: Dict[str, Dict[str, Any]],
    save_path: Path,
) -> None:
    """Plot monthly average MAPE across Train and Test splits."""
    fig, ax = plt.subplots(
        1, 1, figsize=(PAGE_WIDTH_COLUMN, PAGE_WIDTH_COLUMN / GOLDEN_RATIO)
    )

    for dataset in ["Train", "Test"]:
        dates = seasonality_dict[dataset]["dates"]
        means = seasonality_dict[dataset]["means"]
        if dates:
            ax.plot(dates, means, label=f"{dataset} data", zorder=4, linewidth=1.5)

    ax.set_ylim(0, 200)
    ax.legend(frameon=False, fontsize=7)
    ax.set_ylabel("MAPE (%)", fontsize=8)
    ax.set_xlabel("Date", fontsize=8)

    ax.xaxis.set_major_locator(mdates.MonthLocator((6, 12)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator((3, 6, 9, 12)))
    ax.grid(which="both", axis="x", linestyle="--", alpha=0.5, zorder=0)

    fig.tight_layout()
    save_figure(fig, save_path)


def plot_shap_summary(
    df_global: pd.DataFrame,
    df_vehicle: pd.DataFrame,
    save_path: Path,
) -> None:
    """Plot horizontal boxplots of SHAP importance ordered by global mean(|SHAP|)."""
    # Order ascending so most important feature is at the top of the horizontal plot
    df_global_asc = df_global.sort_values(by="importance_mean_abs_shap", ascending=True)
    sorted_features = df_global_asc["feature"].tolist()
    global_importances = df_global_asc["importance_mean_abs_shap"].values
    y_range = range(1, len(sorted_features) + 1)

    fig, ax = plt.subplots(
        1, 1, figsize=(PAGE_WIDTH_COLUMN, PAGE_WIDTH_COLUMN / GOLDEN_RATIO * 1.1)
    )

    # Plot global mean(|SHAP|) marker 'x'
    ax.plot(
        global_importances,
        y_range,
        marker="x",
        color="C0",
        zorder=9,
        linewidth=0,
        markersize=4.5,
        markeredgewidth=1.2,
    )

    # Boxplot of participant distributions
    veh_reordered = df_vehicle[sorted_features].rename(columns=FEATURE_DISPLAY_MAP)
    veh_reordered.plot.box(
        ax=ax,
        vert=False,
        showfliers=False,
        showmeans=True,
        patch_artist=True,
        boxprops={"facecolor": "C1", "edgecolor": "C1"},
        medianprops={"color": COLOR_MARKER},
        meanprops={
            "markerfacecolor": COLOR_MARKER,
            "markeredgecolor": "none",
            "markersize": 4,
        },
        whiskerprops={"color": "C1"},
        capprops={"color": "C1"},
    )

    # Legend
    legend_elements = [
        Line2D([], [], marker="x", label="Mean of all sessions", color="C0", linewidth=0, markersize=4.5, markeredgewidth=1.2),
        Line2D([], [], marker="s", label="Mean on each vehicle", color="C1", markerfacecolor="C1", linewidth=0, markersize=7),
        Line2D([], [], marker="^", label="Mean", color=COLOR_MARKER, markerfacecolor=COLOR_MARKER, linewidth=0, markersize=4),
        Line2D([], [], marker="", label="Median", color=COLOR_MARKER, linewidth=1.2),
    ]
    ax.legend(
        handles=legend_elements,
        frameon=False,
        fontsize=6.5,
        loc="lower right",
        bbox_to_anchor=(1.01, 0),
    )

    ax.set_xlabel("mean(|SHAP value|)", fontsize=8)
    fig.tight_layout()
    save_figure(fig, save_path)


def plot_subgroup_comparison(
    subgroup_metrics_df: pd.DataFrame,
    target: str,
    save_path: Path,
) -> None:
    """Plot comparison bar chart for All, Same-day, and Overnight subgroups."""
    unit = "h" if target == TARGET_DURATION else "kWh"
    subgroups = subgroup_metrics_df["subgroup"].tolist()

    x = np.arange(len(subgroups))
    width = 0.25

    fig, ax1 = plt.subplots(
        figsize=(PAGE_WIDTH_COLUMN * 1.1, PAGE_WIDTH_COLUMN / GOLDEN_RATIO)
    )

    mae_vals = subgroup_metrics_df["mae"].values
    medae_vals = subgroup_metrics_df["medae"].values
    mape_vals = subgroup_metrics_df["mape"].values

    rects1 = ax1.bar(x - width / 2, mae_vals, width, label=f"MAE ({unit})", color="C0", alpha=0.85)
    rects2 = ax1.bar(x + width / 2, medae_vals, width, label=f"MdAE ({unit})", color="C1", alpha=0.85)

    ax1.set_ylabel(f"Error ({unit})", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.replace(" Sessions", "") for s in subgroups], fontsize=8)

    # Secondary axis for MAPE %
    ax2 = ax1.twinx()
    ax2.plot(x, mape_vals, color=COLOR_MARKER, marker="o", linewidth=1.5, markersize=5, label="MAPE (%)")
    ax2.set_ylabel("MAPE (%)", color=COLOR_MARKER, fontsize=8)
    ax2.tick_params(axis="y", labelcolor=COLOR_MARKER)

    # Combined legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", frameon=False, fontsize=7)

    fig.tight_layout()
    save_figure(fig, save_path)


def heatmap_data_from_series(series: pd.Series, interval_minutes: int = 15) -> Tuple[np.ndarray, pd.DatetimeIndex, pd.DatetimeIndex]:
    """Transform time series into day/hour matrix for heatmap plotting."""
    data = series.copy()
    timezone = data.index.tz
    df = data.to_frame(name="values")
    df["date"] = df.index.date
    df["time"] = df.index.time
    pivoted = df.pivot_table(
        index="time",
        columns="date",
        values="values",
        aggfunc=lambda x: x.iloc[0],
        dropna=False,
    )
    daterange = pivoted.columns.astype("datetime64[ns]").tz_localize(timezone)
    full_daterange = pd.date_range(
        start=daterange.min(),
        end=daterange.max() + dt.timedelta(days=1),
        tz=timezone,
    )
    timerange = pd.date_range(
        start="1970-01-01T00:00:00",
        end="1970-01-02T00:00:00",
        freq=f"{interval_minutes}min",
        tz=timezone,
    )
    return pivoted.to_numpy(), full_daterange, timerange


def plot_pcolormesh_grid(
    ax: plt.Axes,
    daterange: pd.DatetimeIndex,
    timerange: pd.DatetimeIndex,
    data: np.ndarray,
    **kwargs,
) -> Any:
    """Plot pcolormesh with datetime x-axis and hour-of-day y-axis."""
    mesh = ax.pcolormesh(daterange, timerange, data, **kwargs)
    ax.set_xlim(daterange[0], daterange[-1])
    ax.invert_yaxis()

    ax.xaxis.set_major_locator(mdates.MonthLocator((1, 4, 7, 10)))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())

    ax.yaxis.set_major_formatter(mdates.DateFormatter("%H"))
    ax.yaxis.set_major_locator(mdates.AutoDateLocator())
    ax.yaxis.set_minor_locator(mdates.HourLocator())

    ax.set_xlabel("Date", fontsize=8)
    ax.set_ylabel("Hour", fontsize=8)

    for pos in ["left", "bottom", "top", "right"]:
        ax.spines[pos].set_color("#999999")
    ax.tick_params(axis="both", which="both", color="#999999")
    for pos in ["top", "right"]:
        ax.spines[pos].set_visible(False)

    return mesh


def plot_duration_heatmaps(
    predictions: pd.DataFrame,
    performance_participants: pd.DataFrame,
    output_dir: Path,
) -> List[Path]:
    """Generate 4-state charging occupancy heatmaps for top 3 best and worst vehicles.

    Encodes:
    0: True Negative (Neither real nor predicted plugged in - White #ffffff)
    1: False Negative (Real plugged in, predicted unplugged - Orange C1)
    2: False Positive (Real unplugged, predicted plugged in - Red C3)
    3: True Positive (Both real and predicted plugged in - Blue C0)

    Args:
        predictions: Predictions DataFrame.
        performance_participants: MultiIndex performance table by participant.
        output_dir: Output folder to persist heatmap-{best,worst}-{0,1,2}.png.

    Returns:
        List of generated figure Paths.
    """
    model_name = "HistGradientBoostingRegressor"
    dataset = "Test"
    score = "mae"
    resolution = "15min"
    n_sample = 3

    preds = predictions.copy()
    preds["end_predicted"] = preds["StartTime"] + pd.to_timedelta(preds[model_name], unit="h")

    perf = performance_participants.loc[:, (model_name, dataset, score)].dropna().sort_values()
    if len(perf) < n_sample * 2:
        logger.warning("Fewer than %d participants for heatmap generation.", n_sample * 2)
        n_sample = max(1, len(perf) // 2)

    colormap = ListedColormap(["#ffffff", "C1", "C3", "C0"])
    quantile_labels = [
        ["Best", "Second best", "Third best"],
        ["Worst", "Second worst", "Third worst"],
    ]

    saved_paths = []
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sample_groups = [
        ("best", perf.iloc[:n_sample]),
        ("worst", perf.iloc[-n_sample:].iloc[::-1]),
    ]

    for g_idx, (group_name, group_data) in enumerate(sample_groups):
        for i, (pid, mae_val) in enumerate(group_data.items()):
            sessions = preds[preds["ParticipantID"] == pid]
            if sessions.empty:
                continue

            start = sessions["StartTime"].min().round(resolution)
            end = sessions[["StopTime", "end_predicted"]].max().max().round(resolution)

            ts = pd.Series(data=0, index=pd.date_range(start, end, freq=resolution))
            for _, session in sessions.iterrows():
                start_idx = session["StartTime"].round(resolution)
                stop_idx = session["StopTime"].round(resolution)
                pred_idx = session["end_predicted"].round(resolution)

                ts.loc[start_idx:stop_idx] += 1
                ts.loc[start_idx:pred_idx] += 2

            ts.clip(0, 3, inplace=True)

            mat_data, daterange, timerange = heatmap_data_from_series(ts, interval_minutes=15)

            fig, ax = plt.subplots(
                1, 1, figsize=(PAGE_WIDTH_COLUMN, PAGE_WIDTH_COLUMN / GOLDEN_RATIO)
            )
            plot_pcolormesh_grid(ax, daterange, timerange, mat_data, cmap=colormap, rasterized=True)

            title_text = f"{quantile_labels[g_idx][i]} prediction performance (MAE={mae_val:.2f}h)"
            ax.set_title(title_text, fontsize=8.5)
            ax.tick_params(direction="out", which="both")
            for pos in ["top", "right"]:
                ax.spines[pos].set_visible(True)

            file_path = output_dir / f"heatmap-{group_name}-{i}.png"
            save_figure(fig, file_path)
            saved_paths.append(file_path)

    return saved_paths

