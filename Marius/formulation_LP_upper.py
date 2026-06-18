"""
ModE Project 5 -- LP upper bound of the MILP dispatch formulation.

Fixes unit commitment deltas (dB, dCHP) to one of two extreme modes, giving
a feasible MILP solution whose cost is an upper bound on the MILP optimum:
  - "boilers_on": all boilers always committed (dB=1), all CHPs off (dCHP=0)
  - "chp_on":     all CHPs always committed (dCHP=1), all boilers off (dB=0)

TES commitment variables (din_TES, dout_TES) are removed entirely; the TES
can charge and discharge freely within its physical capacity bounds.

Conventions: same as formulation_MILP.py.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pyomo.environ import (
    ConcreteModel, Set, RangeSet, Param, Var, Expression, Objective, Constraint,
    Reals, NonNegativeReals, SolverFactory, value, minimize,
)

# ---------------------------------------------------------------------------
# 0. Fixed inputs
# ---------------------------------------------------------------------------
DEMAND_CSV = Path(__file__).resolve().parent.parent / "energy_demands.csv"
PGRID_DOMAIN = NonNegativeReals       # import only; use Reals to allow export

# ---------------------------------------------------------------------------
# 1. Parameters
# ---------------------------------------------------------------------------
dt = 1.0

# TES
tau_loss = 200.0
tau_in = tau_out = 1.0
eta_in_TES = eta_out_TES = 0.95
E_nom_TES = 1000.0
E_min_TES = 0.0

# Boiler
Qout_nom_B = 530.0
eta_nom_B = 0.8
lam_in_min_B = 0.173
lam_out_min_B = 0.2

# CHP
Qout_nom_CHP = 470.0
Pout_nom_CHP = 380.0
eta_nom_CHP_th = 0.481
eta_nom_CHP_el = 0.389
lam_in_min_CHP = 0.582
lam_out_min_CHP_th = 0.622
lam_out_min_CHP_el = 0.5

# Part-load slopes
beta_B = (1.0 - lam_in_min_B) / (1.0 - lam_out_min_B)
beta_CHP_th = (1.0 - lam_in_min_CHP) / (1.0 - lam_out_min_CHP_th)
beta_CHP_el = (1.0 - lam_in_min_CHP) / (1.0 - lam_out_min_CHP_el)

# ---------------------------------------------------------------------------
# 2. Exact discretization of the TES ODE  (section 1.1)
# ---------------------------------------------------------------------------
a  = 0.9950124791926823   # exp(-dt / tau_loss)
b1 = 0.9476289533903605   # eta_in_TES  * tau_loss * (1 - exp(-dt / tau_loss))
b2 = -1.050004380487934   # -(1/eta_out_TES) * tau_loss * (1 - exp(-dt / tau_loss))

# ---------------------------------------------------------------------------
# 3. Demand time series from CSV
# ---------------------------------------------------------------------------
_df = pd.read_csv(DEMAND_CSV)
P_D_series = _df["hourly electricity demand [kW]"].to_numpy()
Q_D_series = _df["hourly heat demand [kW]"].to_numpy()
N = len(_df)


# ---------------------------------------------------------------------------
# 4. Callable solve function
# ---------------------------------------------------------------------------
def solve(
    c_G: float,
    c_el: float,
    *,
    mode: str = "boilers_on",
    strict_demand_satisfaction: bool = True,
    tee: bool = False,
) -> tuple[float, pd.DataFrame]:
    """
    Build and solve the LP upper-bound dispatch model.

    mode: "boilers_on" -- dB=1 for all boilers, dCHP=0 for all CHPs
          "chp_on"     -- dCHP=1 for all CHPs, dB=0 for all boilers

    TES deltas are dropped; Qin_TES and Qout_TES are bounded only by the
    physical capacity limits (E_nom_TES / tau_in and E_nom_TES / tau_out).
    """
    if mode not in ("boilers_on", "chp_on"):
        raise ValueError("mode must be 'boilers_on' or 'chp_on'")

    dB_val   = 1.0 if mode == "boilers_on" else 0.0
    dCHP_val = 1.0 if mode == "chp_on"     else 0.0

    m = ConcreteModel("ModE_P5_TES_dispatch_LP_upper")

    m.K   = RangeSet(0, N - 1)
    m.B   = Set(initialize=[1, 2])
    m.CHP = Set(initialize=[1, 2])

    m.Q_D = Param(m.K, initialize={k: float(Q_D_series[k]) for k in range(N)})
    m.P_D = Param(m.K, initialize={k: float(P_D_series[k]) for k in range(N)})

    # --- Continuous decision variables ---
    m.Qin_B   = Var(m.B,   m.K, domain=NonNegativeReals)
    m.Qin_CHP = Var(m.CHP, m.K, domain=NonNegativeReals)
    # TES: free within physical capacity (no commitment delta gating)
    m.Qin_TES  = Var(m.K, domain=NonNegativeReals, bounds=(0.0, E_nom_TES / tau_in))
    m.Qout_TES = Var(m.K, domain=NonNegativeReals, bounds=(0.0, E_nom_TES / tau_out))
    m.Pgrid    = Var(m.K, domain=PGRID_DOMAIN)

    # --- Auxiliary variables ---
    m.E_TES    = Var(m.K,   domain=NonNegativeReals, bounds=(E_min_TES, E_nom_TES))
    m.Qout_B   = Var(m.B,   m.K, domain=Reals)
    m.Qout_CHP = Var(m.CHP, m.K, domain=Reals)
    m.Pout_CHP = Var(m.CHP, m.K, domain=Reals)

    @m.Expression(m.K)
    def E_G(m, k):
        return dt * (sum(m.Qin_B[i, k] for i in m.B) + sum(m.Qin_CHP[i, k] for i in m.CHP))

    @m.Expression(m.K)
    def E_el(m, k):
        return dt * m.Pgrid[k]

    # ---------------------------------------------------------------------------
    # 5. Constraints
    # ---------------------------------------------------------------------------

    # --- TES dynamics (no delta gating, no simultaneous-use constraint) ---
    @m.Constraint(m.K)
    def tes_dynamics(m, k):
        k_next = (k + 1) % N
        return m.E_TES[k_next] == a * m.E_TES[k] + b1 * m.Qin_TES[k] + b2 * m.Qout_TES[k]

    # --- Boilers (dB fixed to dB_val) ---
    @m.Constraint(m.B, m.K)
    def boiler_output(m, i, k):
        return m.Qout_B[i, k] == Qout_nom_B * (
            dB_val * lam_out_min_B
            + (1.0 / beta_B) * (m.Qin_B[i, k] * eta_nom_B / Qout_nom_B - dB_val * lam_in_min_B)
        )

    @m.Constraint(m.B, m.K)
    def boiler_fuel_ub(m, i, k):
        return m.Qin_B[i, k] <= dB_val * Qout_nom_B / eta_nom_B

    @m.Constraint(m.B, m.K)
    def boiler_fuel_lb(m, i, k):
        return m.Qin_B[i, k] >= dB_val * lam_in_min_B * Qout_nom_B / eta_nom_B

    # --- CHPs (dCHP fixed to dCHP_val) ---
    @m.Constraint(m.CHP, m.K)
    def chp_heat(m, i, k):
        return m.Qout_CHP[i, k] == Qout_nom_CHP * (
            dCHP_val * lam_out_min_CHP_th
            + (1.0 / beta_CHP_th) * (m.Qin_CHP[i, k] * eta_nom_CHP_th / Qout_nom_CHP - dCHP_val * lam_in_min_CHP)
        )

    @m.Constraint(m.CHP, m.K)
    def chp_power(m, i, k):
        return m.Pout_CHP[i, k] == Pout_nom_CHP * (
            dCHP_val * lam_out_min_CHP_el
            + (1.0 / beta_CHP_el) * (m.Qin_CHP[i, k] * eta_nom_CHP_el / Pout_nom_CHP - dCHP_val * lam_in_min_CHP)
        )

    @m.Constraint(m.CHP, m.K)
    def chp_fuel_ub(m, i, k):
        return m.Qin_CHP[i, k] <= dCHP_val * Qout_nom_CHP / eta_nom_CHP_th

    @m.Constraint(m.CHP, m.K)
    def chp_fuel_lb(m, i, k):
        return m.Qin_CHP[i, k] >= dCHP_val * lam_in_min_CHP * Pout_nom_CHP / eta_nom_CHP_el

    _op = (lambda a, b: a == b) if strict_demand_satisfaction else (lambda a, b: a >= b)

    # --- Balances ---
    @m.Constraint(m.K)
    def heat_balance(m, k):
        return _op(
            sum(m.Qout_CHP[i, k] for i in m.CHP)
            + sum(m.Qout_B[i, k] for i in m.B)
            + m.Qout_TES[k] - m.Qin_TES[k],
            m.Q_D[k],
        )

    @m.Constraint(m.K)
    def power_balance(m, k):
        return _op(sum(m.Pout_CHP[i, k] for i in m.CHP) + m.Pgrid[k], m.P_D[k])

    # ---------------------------------------------------------------------------
    # 6. Objective
    # ---------------------------------------------------------------------------
    @m.Objective(sense=minimize)
    def total_cost(m):
        return c_G * sum(m.E_G[k] for k in m.K) + c_el * sum(m.E_el[k] for k in m.K)

    # ---------------------------------------------------------------------------
    # 7. Solve
    # ---------------------------------------------------------------------------
    solver = SolverFactory("gurobi")
    solver.options["TimeLimit"] = 300
    solver.solve(m, tee=tee)

    opex = value(m.total_cost)

    rows = []
    for k in m.K:
        rows.append({
            "k": k,
            "Q_D": value(m.Q_D[k]),
            "P_D": value(m.P_D[k]),
            "E_TES": value(m.E_TES[k]),
            "Qin_TES": value(m.Qin_TES[k]),
            "Qout_TES": value(m.Qout_TES[k]),
            "Pgrid": value(m.Pgrid[k]),
            **{f"dB{i}": dB_val for i in m.B},
            **{f"dCHP{i}": dCHP_val for i in m.CHP},
            **{f"Qin_B{i}": value(m.Qin_B[i, k]) for i in m.B},
            **{f"Qout_B{i}": value(m.Qout_B[i, k]) for i in m.B},
            **{f"Qin_CHP{i}": value(m.Qin_CHP[i, k]) for i in m.CHP},
            **{f"Qout_CHP{i}": value(m.Qout_CHP[i, k]) for i in m.CHP},
            **{f"Pout_CHP{i}": value(m.Pout_CHP[i, k]) for i in m.CHP},
        })

    return opex, pd.DataFrame(rows)


def plot_dispatch_results(
    dispatch: pd.DataFrame,
    output_path: str = "Marius/visualization/dispatch_overview_LP_upper.png",
    gas_price: float | None = None,
    el_price: float | None = None,
    opex: float | None = None,
    mode: str = "boilers_on",
    fontsize: int = 10,
):
    """Dashboard for the LP upper-bound dispatch (fixed unit commitment)."""
    fs_tick     = fontsize
    fs_label    = fontsize
    fs_title    = round(fontsize * 1.1)
    fs_legend   = max(fontsize - 1, 7)
    fs_suptitle = round(fontsize * 1.4)

    gas_value = gas_price if gas_price is not None else 0.0
    el_value  = el_price  if el_price  is not None else 0.0

    from matplotlib.colors import LinearSegmentedColormap
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes as _inset_axes

    k = dispatch["k"].to_numpy()

    fig, axes = plt.subplots(4, 1, figsize=(18, 18), sharex=True)

    # 1) Unit commitment (fixed constants -- sanity check)
    delta_matrix = np.vstack([
        np.clip(dispatch["dB1"].to_numpy(),   0.0, 1.0),
        np.clip(dispatch["dB2"].to_numpy(),   0.0, 1.0),
        np.clip(dispatch["dCHP1"].to_numpy(), 0.0, 1.0),
        np.clip(dispatch["dCHP2"].to_numpy(), 0.0, 1.0),
    ])
    cmap_uc = LinearSegmentedColormap.from_list("uc_cmap", ["#ffffe5", "#238443"])
    im = axes[0].imshow(
        delta_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap_uc,
        vmin=0,
        vmax=1,
        extent=[k[0] - 0.5, k[-1] + 0.5, -0.5, 3.5],
        origin="lower",
    )
    axes[0].set_yticks([0, 1, 2, 3])
    axes[0].set_yticklabels(["Boiler 1", "Boiler 2", "CHP 1", "CHP 2"], fontsize=fs_tick)
    axes[0].tick_params(labelsize=fs_tick)
    axes[0].set_title("Unit Commitment δ (fixed — sanity check)", fontsize=fs_title)
    axes[0].set_ylabel("Units", fontsize=fs_label)
    cax = _inset_axes(axes[0], width="3%", height="100%", loc="lower left",
                      bbox_to_anchor=(1.01, 0., 1, 1), bbox_transform=axes[0].transAxes,
                      borderpad=0)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("δ value [0–1]", fontsize=fs_label)
    cbar.ax.tick_params(labelsize=fs_tick)

    # 2) TES charging/discharging and state of charge
    axes[1].bar(k,  dispatch["Qout_TES"], width=0.9, label="TES discharge [kW]", color="#2C7FB8", alpha=0.9)
    axes[1].bar(k, -dispatch["Qin_TES"],  width=0.9, label="TES charge [kW]",    color="#EF3B2C", alpha=0.7)
    axes[1].plot(k, dispatch["E_TES"], color="#6A3D9A", linewidth=2.0, label="TES stored energy [kWh]")
    axes[1].axhline(0.0, color="black", linewidth=0.9)
    axes[1].set_ylabel("TES output power [kW]\n/ stored energy [kWh]", fontsize=fs_label)
    axes[1].set_title("TES Operation (free charge/discharge)", fontsize=fs_title)
    axes[1].tick_params(labelsize=fs_tick)
    axes[1].grid(True, axis="y", linestyle=":", linewidth=0.8, alpha=0.7)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=fs_legend)

    # 3) Electrical supply mix
    p_chp_total = dispatch["Pout_CHP1"] + dispatch["Pout_CHP2"]
    axes[2].fill_between(k, 0, dispatch["Pgrid"], step="mid", alpha=0.45, color="#FDAE61", label="Grid import")
    axes[2].plot(k, p_chp_total, color="#1B9E77", linewidth=2, label="CHP electric output")
    axes[2].plot(k, dispatch["P_D"], color="#111111", linewidth=1.7, linestyle="--", label="Electric demand")
    axes[2].set_title("Electrical Supply Mix", fontsize=fs_title)
    axes[2].set_ylabel("Power [kW]", fontsize=fs_label)
    axes[2].tick_params(labelsize=fs_tick)
    axes[2].grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    axes[2].legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=fs_legend)

    # 4) Heat supply mix and gas purchase
    q_boiler_total = dispatch["Qout_B1"] + dispatch["Qout_B2"]
    q_chp_total    = dispatch["Qout_CHP1"] + dispatch["Qout_CHP2"]
    q_tes_net      = dispatch["Qout_TES"] - dispatch["Qin_TES"]
    q_gas_total    = dispatch["Qin_B1"] + dispatch["Qin_B2"] + dispatch["Qin_CHP1"] + dispatch["Qin_CHP2"]

    axes[3].plot(k, q_boiler_total, color="#E31A1C", linewidth=1.8, label="Boiler heat output")
    axes[3].plot(k, q_chp_total,   color="#FF7F00", linewidth=1.8, label="CHP heat output")
    axes[3].plot(k, q_tes_net,     color="#2C7FB8", linewidth=1.8, label="TES net heat (discharge-charge)")
    axes[3].plot(k, dispatch["Q_D"], color="#111111", linewidth=1.7, linestyle="--", label="Heat demand")
    axes[3].bar(k, q_gas_total, width=0.85, alpha=0.22, color="#33A02C", label="Gas purchased (fuel input)")
    axes[3].set_title("Heat Supply and Gas Purchase", fontsize=fs_title)
    axes[3].set_ylabel("Heat flow / Gas input [kW]", fontsize=fs_label)
    axes[3].set_xlabel("Time step k [-]", fontsize=fs_label)
    axes[3].tick_params(labelsize=fs_tick)
    axes[3].grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    axes[3].legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0, fontsize=fs_legend)

    mode_label = "boilers always ON, CHPs OFF" if mode == "boilers_on" else "CHPs always ON, boilers OFF"
    ratio_str = f"   |   c_G/c_el = {gas_value/el_value:.3f}" if el_value != 0 else ""
    opex_str = f"   |   Total OPEX: {opex:,.2f} €{ratio_str}" if opex is not None else ""
    _title = (
        f"LP Upper Bound  ({mode_label})\n"
        f"gas={gas_value:.3f} €/kWh, el={el_value:.3f} €/kWh{opex_str}"
    )
    fig.suptitle(_title, fontsize=fs_suptitle)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.subplots_adjust(right=0.78, top=0.86)
    fig.savefig(output_file, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    _cG  = 0.16
    _cel = 0.1
    _mode = sys.argv[1] if len(sys.argv) > 1 else "boilers_on"

    print(f"N = {N} intervals, dt = {dt} h  ->  discretization a={a:.6f}, b1={b1:.6f}, b2={b2:.6f}")
    print(f"beta_B={beta_B:.5f}, beta_CHP_th={beta_CHP_th:.5f}, beta_CHP_el={beta_CHP_el:.5f}")
    print(f"Mode: {_mode}  (dB={'1' if _mode == 'boilers_on' else '0'}, dCHP={'1' if _mode == 'chp_on' else '0'})")
    print("TES: free charge/discharge within physical capacity limits")

    opex, dispatch = solve(_cG, _cel, mode=_mode, tee=True)

    print("\nSolver status: done")
    print(f"Total cost    : {opex:.2f}")

    dispatch.to_csv("Marius/results/dispatch_result_LP_upper.csv", index=False)
    print("Dispatch written to Marius/results/dispatch_result_LP_upper.csv")

    plot_dispatch_results(dispatch, output_path="Marius/visualization/dispatch_overview_LP_upper.png",
                          gas_price=_cG, el_price=_cel, opex=opex, mode=_mode, fontsize=18)
    print("Dispatch visualization written to Marius/visualization/dispatch_overview_LP_upper.png")
