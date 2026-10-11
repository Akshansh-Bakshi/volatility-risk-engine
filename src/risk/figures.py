"""Figures for out-of-sample VaR forecasts and breach summaries."""

from __future__ import annotations

import pandas as pd


def plot_var_forecasts(
    forecasts: pd.DataFrame,
    *,
    title: str = "Realised losses and one-day-ahead VaR",
    figsize: tuple[float, float] = (14, 8),
) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]
    """Plot realised losses, all method thresholds and breaches at 95%/99%."""
    import matplotlib.figure as _mfig

    methods = {
        "GARCH Normal": "garch",
        "EGARCH Normal": "egarch",
        "250-return HS": "historical",
    }
    required = ["realised_loss_pct"] + [
        f"{prefix}_var{level}_pct"
        for prefix in methods.values()
        for level in (95, 99)
    ] + [
        f"{prefix}_breach{level}"
        for prefix in methods.values()
        for level in (95, 99)
    ]
    missing = [column for column in required if column not in forecasts]
    if missing or forecasts.empty:
        message = f"Missing VaR plot columns: {', '.join(missing)}." if missing else "Forecast frame is empty."
        raise ValueError(message)

    colors = {"garch": "#1f77b4", "egarch": "#d62728", "historical": "#2ca02c"}
    fig = _mfig.Figure(figsize=figsize, tight_layout=True)
    for axis_number, level in enumerate((95, 99), start=1):
        ax = fig.add_subplot(2, 1, axis_number)
        ax.plot(
            forecasts.index,
            forecasts["realised_loss_pct"],
            color="#555555",
            linewidth=0.9,
            alpha=0.7,
            label="Realised loss",
        )
        for method, prefix in methods.items():
            threshold = forecasts[f"{prefix}_var{level}_pct"]
            ax.plot(
                forecasts.index,
                threshold,
                color=colors[prefix],
                linewidth=1.2,
                label=f"{method} VaR{level}",
            )
            breach_mask = forecasts[f"{prefix}_breach{level}"].astype(bool)
            if breach_mask.any():
                ax.scatter(
                    forecasts.index[breach_mask],
                    forecasts.loc[breach_mask, "realised_loss_pct"],
                    color=colors[prefix],
                    marker="x",
                    s=34,
                    zorder=4,
                    label=f"{method} breach",
                )
        ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.5)
        ax.set_ylabel("Loss / VaR (% return points)")
        ax.set_title(f"{level}% confidence")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        if axis_number == 2:
            ax.set_xlabel("Date")
    fig.suptitle(title)
    return fig


def plot_var_breach_counts(
    summary: pd.DataFrame,
    *,
    title: str = "VaR breach counts by method",
    figsize: tuple[float, float] = (10, 5),
) -> "matplotlib.figure.Figure":  # type: ignore[name-defined]
    """Plot observed breach counts and expected counts for each method/level."""
    import matplotlib.figure as _mfig

    required = {"method", "confidence", "breach_count", "sample_size", "expected_breach_rate"}
    if not required.issubset(summary.columns):
        missing = sorted(required - set(summary.columns))
        raise ValueError(f"Summary is missing required column(s): {', '.join(missing)}.")
    methods = list(dict.fromkeys(summary["method"].tolist()))
    levels = sorted(summary["confidence"].unique())
    fig = _mfig.Figure(figsize=figsize, tight_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    width = 0.8 / max(len(levels), 1)
    base = pd.Series(range(len(methods)), index=methods, dtype=float)
    for level_index, confidence in enumerate(levels):
        rows = summary.loc[summary["confidence"] == confidence].set_index("method")
        positions = base.to_numpy() - 0.4 + width / 2 + level_index * width
        observed = [float(rows.loc[method, "breach_count"]) for method in methods]
        expected = [
            float(rows.loc[method, "sample_size"] * rows.loc[method, "expected_breach_rate"])
            for method in methods
        ]
        ax.bar(positions, observed, width=width, label=f"Observed {confidence:.0%}")
        ax.scatter(positions, expected, marker="_", color="black", s=120, zorder=4,
                   label=f"Expected {confidence:.0%}")
    ax.set_xticks(base.to_numpy(), methods, rotation=12, ha="right")
    ax.set_ylabel("Breach count")
    ax.set_title(title)
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.25)
    return fig
