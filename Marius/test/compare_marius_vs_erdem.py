"""
Cross-validation: Marius standalone formulations vs Erdem core.py wrappers.

For each of the 4 formulations (MILP, LP lower, LP upper, LP approx) this
script solves every price ratio in the log-training-sample range using both
implementations, then plots the relative OPEX discrepancy.

Expected outcome: all discrepancies are below ~1e-3 (MIPGap for MILP; purely
numerical for the LPs).

Run from repo root:
    python Marius/test/compare_marius_vs_erdem.py
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
REWRITE    = False
CACHE_PATH = Path("Marius/results/compare_marius_vs_erdem.csv")
RATIOS_CSV = Path("Erdem/results/Sampling/log_training_samples.csv")
MIP_GAP    = 1e-2

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
sys.path.insert(0, "Marius")
import formulation_MILP          as _marius_milp
import formulation_LP_lower      as _marius_lp_lower
import formulation_LP_upper      as _marius_lp_upper
import formulation_LP_approximated as _marius_lp_approx

sys.path.append("Erdem")
from src.optimization.core import (
    solve_milp, solve_lp_lower, solve_lp_upper, solve_lp_approximated,
)

import pandas as _pd
_demand_df = _pd.read_csv(Path("energy_demands.csv"))
_Q_D = _demand_df["hourly heat demand [kW]"].to_numpy()
_P_D = _demand_df["hourly electricity demand [kW]"].to_numpy()

ratios = pd.read_csv(RATIOS_CSV)["ratios"].values
c_el   = 1.0

# ---------------------------------------------------------------------------
# Solve or load cache
# ---------------------------------------------------------------------------
if not REWRITE and CACHE_PATH.exists():
    print(f"Loading cached results from {CACHE_PATH}")
    df = pd.read_csv(CACHE_PATH)
else:
    rows = []
    n = len(ratios)
    for i, ratio in enumerate(ratios):
        c_G = ratio * c_el
        print(f"[{i+1}/{n}] ratio={ratio:.6f}  c_G={c_G:.6f}")

        m_milp,   _ = _marius_milp.solve(c_G, c_el, mip_gap=MIP_GAP)
        m_lower,  _ = _marius_lp_lower.solve(c_G, c_el)
        m_upper,  _ = _marius_lp_upper.solve(c_G, c_el)
        m_approx, _ = _marius_lp_approx.solve(c_G, c_el)

        e_milp,   _ = solve_milp(_Q_D, _P_D, c_G, c_el, mip_gap=MIP_GAP)
        e_lower,  _ = solve_lp_lower(_Q_D, _P_D, c_G, c_el)
        e_upper,  _ = solve_lp_upper(_Q_D, _P_D, c_G, c_el)
        e_approx, _ = solve_lp_approximated(_Q_D, _P_D, c_G, c_el, mode="mean_efficiency")

        rows.append({
            "ratio":         ratio,
            "marius_milp":   m_milp,   "erdem_milp":   e_milp,
            "marius_lower":  m_lower,  "erdem_lower":  e_lower,
            "marius_upper":  m_upper,  "erdem_upper":  e_upper,
            "marius_approx": m_approx, "erdem_approx": e_approx,
        })
        print(f"  MILP   marius={m_milp:,.0f}  erdem={e_milp:,.0f}  "
              f"rel={abs(m_milp-e_milp)/max(abs(m_milp),1):.2e}")
        print(f"  lower  marius={m_lower:,.0f}  erdem={e_lower:,.0f}  "
              f"rel={abs(m_lower-e_lower)/max(abs(m_lower),1):.2e}")
        print(f"  upper  marius={m_upper:,.0f}  erdem={e_upper:,.0f}  "
              f"rel={abs(m_upper-e_upper)/max(abs(m_upper),1):.2e}")
        print(f"  approx marius={m_approx:,.0f}  erdem={e_approx:,.0f}  "
              f"rel={abs(m_approx-e_approx)/max(abs(m_approx),1):.2e}")

    df = pd.DataFrame(rows)
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)
    print(f"\nSaved to {CACHE_PATH}")

# ---------------------------------------------------------------------------
# Relative discrepancies
# ---------------------------------------------------------------------------
pairs = [
    ("MILP",    "marius_milp",   "erdem_milp"),
    ("LP lower","marius_lower",  "erdem_lower"),
    ("LP upper","marius_upper",  "erdem_upper"),
    ("LP approx","marius_approx","erdem_approx"),
]

for label, mc, ec in pairs:
    rel = np.abs(df[mc].values - df[ec].values) / np.maximum(np.abs(df[mc].values), 1.0)
    print(f"{label:12s}  max_rel={rel.max():.2e}  mean_rel={rel.mean():.2e}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
COLORS = ["#2166AC", "#4DAC26", "#D6604D", "#35978F"]
fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
fig.suptitle("Relative OPEX discrepancy: Marius vs Erdem core.py", fontsize=13)

for ax, (label, mc, ec), color in zip(axes.flatten(), pairs, COLORS):
    rel = np.abs(df[mc].values - df[ec].values) / np.maximum(np.abs(df[mc].values), 1.0)
    ax.plot(df["ratio"], rel, color=color, marker="o", markersize=3, linewidth=1.2)
    ax.axhline(MIP_GAP, color="gray", linestyle="--", linewidth=0.9, label=f"MIPGap={MIP_GAP:.0e}")
    ax.set_xscale("log")
    ax.set_title(label, fontsize=11)
    ax.set_ylabel(r"$|OPEX_M - OPEX_E|\,/\,|OPEX_M|$", fontsize=8)
    ax.set_xlabel(r"Price ratio $c_G/c_{\rm el}$", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(True, which="both", ls="--", alpha=0.4)

plt.tight_layout()
out_path = Path("Marius/visualization/compare_marius_vs_erdem.png")
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out_path, dpi=150)
print(f"\nPlot saved to {out_path}")
plt.show()
plt.close()
