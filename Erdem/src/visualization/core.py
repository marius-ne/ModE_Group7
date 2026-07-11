# src/visualization/core.py
import numpy as np
from src.visualization.style import *
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from typing import Optional, Sequence


def plot_unit_on_off_status(
    sol_df,
    ax: Optional[plt.Axes] = None,
    unit_cols: Optional[Sequence[str]] = None,
    unit_names: Optional[Sequence[str]] = None,
    show_legend: bool = True,
) -> plt.Axes:
    """
    Zeichnet On/Off-Status jeder Anlage als gefüllte schmale Balken (imshow-like).
    Wenn `ax` übergeben wird, wird dort gezeichnet; sonst wird eine neue Figure/Axes erzeugt.
    show_legend steuert, ob eine interne Legende für diese Achse gezeichnet wird.
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots()
        created_fig = True

    # Default column names
    if unit_cols is None:
        unit_cols = ['delta_B1', 'delta_B2', 'delta_CHP1', 'delta_CHP2']
    if unit_names is None:
        unit_names = ['Boiler 1', 'Boiler 2', 'CHP 1', 'CHP 2']

    # Time index
    if 'k' in sol_df.columns:
        k = sol_df['k'].to_numpy()
        time_steps = np.asarray(k)
    else:
        time_steps = np.arange(len(sol_df))

    n_units = len(unit_cols)
    status_matrix = np.zeros((n_units, len(sol_df)))
    for i, col in enumerate(unit_cols):
        if col in sol_df.columns:
            status_matrix[i, :] = (sol_df[col].to_numpy() > 0.5).astype(float)
        else:
            status_matrix[i, :] = 0.0

    cmap = ListedColormap(['#F3E7D3', '#238443'])  # Off=beige, On=green

    ax.imshow(
        status_matrix,
        aspect='auto',
        interpolation='nearest',
        cmap=cmap,
        vmin=0,
        vmax=1,
        extent=[time_steps[0] - 0.5, time_steps[-1] + 0.5, -0.5, n_units - 0.5],
        origin='lower',
    )

    ax.set_yticks(np.arange(n_units))
    ax.set_yticklabels(unit_names)
    ax.set_ylabel('Unit')

    # no minor ticks on y-axis
    ax.yaxis.set_minor_locator(plt.NullLocator())

    # horizontal dashed lines to separate components
    for i in range(1, n_units):
        ax.axhline(y=i - 0.5, color='gray', linestyle='--', linewidth=1.0, alpha=0.7)

    # Legend direkt unterhalb der Achse
    if show_legend:
        on_patch = mpatches.Patch(facecolor='#238443', edgecolor='black', label='On')
        off_patch = mpatches.Patch(facecolor='#F3E7D3', edgecolor='black', label='Off')

        ax.legend(
            handles=[on_patch, off_patch],
            loc='upper center',
            bbox_to_anchor=(0.5, -0.04),
            ncol=2,
            framealpha=0.95,
            frameon=True,
        )

    if created_fig:
        plt.tight_layout()

    return ax


def plot_tes_operation(
    sol_df,
    ax: Optional[plt.Axes] = None,
    show_legend: bool = True,
) -> plt.Axes:
    """
    Zeichnet TES-Lade-/Entladeleistung (Balken) und SOC (Linie).
    Wenn `ax` übergeben wird, wird dort gezeichnet; sonst wird eine neue Figure/Axes erzeugt.
    Power [kW] auf der linken Achse, Energy [kWh] auf der rechten Achse.
    show_legend steuert, ob diese Achse ihre interne Legende zeichnet.
    Erwartete Spalten in sol_df: 'k', 'Q_in_TES', 'Q_out_TES', 'E_TES'
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots()
        created_fig = True

    # Zeitindex
    if 'k' in sol_df.columns:
        k = sol_df['k'].to_numpy()
    else:
        k = np.arange(len(sol_df))

    Q_in = sol_df['Q_in_TES'].to_numpy() if 'Q_in_TES' in sol_df.columns else np.zeros_like(k)
    Q_out = sol_df['Q_out_TES'].to_numpy() if 'Q_out_TES' in sol_df.columns else np.zeros_like(k)
    E_tes = sol_df['E_TES'].to_numpy() if 'E_TES' in sol_df.columns else np.zeros_like(k)

    # Primary axis (Power kW)
    width = 0.8
    ax.bar(k, Q_in, width=width, color='#2166AC', alpha=0.7, label='Charging')
    ax.bar(k, -Q_out, width=width, color='#B2182B', alpha=0.7, label='Discharging')

    ax.set_xlabel('Time step [-]')
    ax.set_ylabel('(Dis-)Charging Power [kW]')
    ax.axhline(0, color='black', linewidth=0.8)
    ax.grid(True, axis='y', linestyle=':', linewidth=0.8, alpha=0.5)

    # Symmetrische y-Limits um Null
    max_power = max(np.max(Q_in) if Q_in.size else 0.0, np.max(Q_out) if Q_out.size else 0.0)
    if np.isfinite(max_power) and max_power > 0:
        pad = 1.1
        ax.set_ylim(-max_power * pad, max_power * pad)

    # Secondary axis (Energy kWh) - SOC als Linie
    ax2 = ax.twinx()
    soc_color = '#1B9E77'
    ax2.step(k, E_tes, where='mid', color=soc_color, linewidth=0.8, label='State of Charge')

    ax2.set_ylabel('State of Charge [kWh]')#, color=soc_color)
    ax2.tick_params(axis='y', labelcolor=soc_color)
    ax2.spines['right'].set_color(soc_color)

    # keine minor ticks
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax2.yaxis.set_minor_locator(plt.NullLocator())

    # Legend direkt unterhalb der Achse
    if show_legend:
        handles = [
            mpatches.Patch(facecolor='#2166AC', edgecolor='black', label='Charging'),
            mpatches.Patch(facecolor='#B2182B', edgecolor='black', label='Discharging'),
            plt.Line2D([0], [0], color='#1B9E77', linewidth=0.8, label='State of Charge'),
        ]
        labels = ['Charging', 'Discharging', 'State of Charge']
        ax.legend(
            handles=handles,
            labels=labels,
            loc='upper center',
            bbox_to_anchor=(0.5, 1.2),
            ncol=3,
            framealpha=0.95,
            frameon=True,
        )

    if created_fig:
        plt.tight_layout()

    return ax


def plot_dispatch_stacked(
    sol_df,
    unit_cols: Optional[Sequence[str]] = None,
    unit_names: Optional[Sequence[str]] = None,
    height_ratios=(1, 2),
    figsize: tuple[float, float] | None = None,
) -> plt.Figure:
    """
    Wrapper: zeichnet Unit-On/Off (oben) und TES-Operation (unten) untereinander in einer Figure.

    Parameters:
    -----------
    figsize : tuple[float, float] | None
        Figure size (width, height) in inches. Falls None, wird die aktuelle Style-Figsize verwendet.
    """
    apply_style(nrows=2)

    # Falls figsize nicht übergeben, nimm die aktuelle Figsize aus rcParams
    if figsize is None:
        figsize = plt.rcParams['figure.figsize']

    fig, axes = plt.subplots(2, 1, sharex=True, figsize=figsize, gridspec_kw={'height_ratios': height_ratios})
    plt.subplots_adjust(hspace=1.05)

    ax_top = axes[0]
    ax_bot = axes[1]

    plot_unit_on_off_status(sol_df, ax=ax_top, unit_cols=unit_cols, unit_names=unit_names, show_legend=True)
    plot_tes_operation(sol_df, ax=ax_bot, show_legend=True)

    plt.tight_layout()
    return fig
