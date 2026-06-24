"""Standalone dispatch dashboard for LP upper-bound results."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

_MODE_LABELS = {
    "boilers_on": "boilers always ON, CHPs OFF",
    "chp_on":     "CHPs always ON, boilers OFF",
    "rounded":    "LP-lower rounded commitment",
    "min":        "best of boilers_on / chp_on",
}


def plot_lp_upper_dispatch(
    dispatch,
    c_G: float,
    c_el: float,
    opex: float | None = None,
    mode: str = "min",
    output_path: str | None = None,
    fontsize: int = 10,
) -> Path:
    """4-panel dispatch dashboard for an LP upper-bound solution.

    Returns the path to the saved PNG.
    """
    if output_path is None:
        ratio = c_G / c_el if c_el != 0 else 0.0
        output_path = f"Marius/visualization/dispatch_lp_upper_ratio{ratio:.2f}.png"

    fs_tick     = fontsize
    fs_label    = fontsize
    fs_title    = round(fontsize * 1.1)
    fs_legend   = max(fontsize - 1, 7)
    fs_suptitle = round(fontsize * 1.4)

    k = dispatch["k"].to_numpy()

    fig, axes = plt.subplots(4, 1, figsize=(18, 18), sharex=True)

    # 1) Unit commitment (heuristic schedule)
    delta_matrix = np.vstack([
        np.clip(dispatch["dB1"].to_numpy(),   0.0, 1.0),
        np.clip(dispatch["dB2"].to_numpy(),   0.0, 1.0),
        np.clip(dispatch["dCHP1"].to_numpy(), 0.0, 1.0),
        np.clip(dispatch["dCHP2"].to_numpy(), 0.0, 1.0),
    ])
    cmap_uc = mcolors.LinearSegmentedColormap.from_list("uc", ["#ffffe5", "#238443"])
    im = axes[0].imshow(
        delta_matrix, aspect="auto", interpolation="nearest",
        cmap=cmap_uc, vmin=0, vmax=1,
        extent=[k[0] - 0.5, k[-1] + 0.5, -0.5, 3.5], origin="lower",
    )
    axes[0].set_yticks([0, 1, 2, 3])
    axes[0].set_yticklabels(["Boiler 1", "Boiler 2", "CHP 1", "CHP 2"], fontsize=fs_tick)
    axes[0].set_title("Unit Commitment δ (fixed heuristic schedule)", fontsize=fs_title)
    axes[0].set_ylabel("Units", fontsize=fs_label)
    fig.colorbar(im, ax=axes[0], label="δ value [0–1]", fraction=0.015, pad=0.01)

    # 2) TES operation
    axes[1].bar(k,  dispatch["Qout_TES"], width=0.9, label="TES discharge [kW]", color="#2C7FB8", alpha=0.9)
    axes[1].bar(k, -dispatch["Qin_TES"],  width=0.9, label="TES charge [kW]",    color="#EF3B2C", alpha=0.7)
    axes[1].plot(k, dispatch["E_TES"], color="#6A3D9A", linewidth=2.0, label="TES stored energy [kWh]")
    axes[1].axhline(0.0, color="black", linewidth=0.9)
    axes[1].set_ylabel("TES power [kW] / energy [kWh]", fontsize=fs_label)
    axes[1].set_title("TES Operation", fontsize=fs_title)
    axes[1].grid(True, axis="y", linestyle=":", linewidth=0.8, alpha=0.7)
    axes[1].legend(loc="upper right", fontsize=fs_legend)

    # 3) Electrical supply mix
    p_chp = dispatch["Pout_CHP1"] + dispatch["Pout_CHP2"]
    axes[2].fill_between(k, 0, dispatch["Pgrid"], step="mid", alpha=0.45, color="#FDAE61", label="Grid import")
    axes[2].plot(k, p_chp, color="#1B9E77", linewidth=2, label="CHP electric output")
    axes[2].plot(k, dispatch["P_D"], color="#111111", linewidth=1.7, linestyle="--", label="Electric demand")
    axes[2].set_title("Electrical Supply Mix", fontsize=fs_title)
    axes[2].set_ylabel("Power [kW]", fontsize=fs_label)
    axes[2].grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    axes[2].legend(loc="upper right", fontsize=fs_legend)

    # 4) Heat supply and gas purchase
    q_boiler = dispatch["Qout_B1"] + dispatch["Qout_B2"]
    q_chp    = dispatch["Qout_CHP1"] + dispatch["Qout_CHP2"]
    q_tes    = dispatch["Qout_TES"] - dispatch["Qin_TES"]
    q_gas    = dispatch["Qin_B1"] + dispatch["Qin_B2"] + dispatch["Qin_CHP1"] + dispatch["Qin_CHP2"]
    axes[3].plot(k, q_boiler, color="#E31A1C", linewidth=1.8, label="Boiler heat output")
    axes[3].plot(k, q_chp,    color="#FF7F00", linewidth=1.8, label="CHP heat output")
    axes[3].plot(k, q_tes,    color="#2C7FB8", linewidth=1.8, label="TES net heat")
    axes[3].plot(k, dispatch["Q_D"], color="#111111", linewidth=1.7, linestyle="--", label="Heat demand")
    axes[3].bar(k, q_gas, width=0.85, alpha=0.22, color="#33A02C", label="Gas purchased (fuel input)")
    axes[3].set_title("Heat Supply and Gas Purchase", fontsize=fs_title)
    axes[3].set_ylabel("Heat flow / Gas input [kW]", fontsize=fs_label)
    axes[3].set_xlabel("Time step k [-]", fontsize=fs_label)
    axes[3].grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    axes[3].legend(loc="upper right", fontsize=fs_legend)

    mode_label = _MODE_LABELS.get(mode, mode)
    ratio_str  = f"  |  c_G/c_el = {c_G/c_el:.3f}" if c_el != 0 else ""
    opex_str   = f"  |  OPEX = {opex:,.2f} €" if opex is not None else ""
    fig.suptitle(
        f"LP Upper Bound  ({mode_label})\n"
        f"gas = {c_G:.3f} €/kWh,  el = {c_el:.3f} €/kWh{ratio_str}{opex_str}",
        fontsize=fs_suptitle,
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out
