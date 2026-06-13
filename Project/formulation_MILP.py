"""
ModE Project 5 -- MILP operational dispatch of a district energy system
(2 boilers + 2 CHPs + thermal energy storage), exact ZOH discretization.

Implements the sparse problem formulation from `problem_formulation_Marius.pdf`
in Pyomo and solves it with Gurobi.

Conventions / modeling decisions (read these -- some affect the optimum):
  * Time index k in {0, ..., N-1}, i.e. N intervals of length dt. The state
    E_TES[k] is the storage content at the START of interval k. The cyclic
    constraint (VIII) E_TES(0) = E_TES(t_f) is enforced implicitly by wrapping
    the dynamics recursion at k = N-1 back to k = 0 (modular indexing), so no
    separate cyclic constraint is needed.
  * cG, cel are the *uncertain* parameters in the PDF (sampled within ranges).
    Here they are single placeholder values -- SET THEM to your sample/midpoint
    or loop the build+solve over your samples.
  * Pgrid is import-only (NonNegativeReals): the power balance is an exact
    equality and the grid covers the deficit, so the CHPs can never produce
    more electricity than P_D at any step (no export). Set PGRID_DOMAIN = Reals
    to allow export (Pgrid < 0).
  * The heat/power balances (Demand I/II) are equalities, exactly as written.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import expm
from pyomo.environ import (
    ConcreteModel, Set, RangeSet, Param, Var, Expression, Objective, Constraint,
    Reals, NonNegativeReals, Binary, SolverFactory, value, minimize,
)

# ---------------------------------------------------------------------------
# 0. User-settable inputs
# ---------------------------------------------------------------------------
DEMAND_CSV = "Project/energy_demands.csv"     # columns: time [h], elec demand [kW], heat demand [kW]
PGRID_DOMAIN = NonNegativeReals       # import only (no export); use Reals to allow export

# Uncertain cost parameters (PLACEHOLDERS -- replace with your sampled values)
cG = 0.16     # gas cost  [currency / kWh]
cel = 0.21   # grid electricity cost [currency / kWh]

# ---------------------------------------------------------------------------
# 1. Parameters (numeric values from the PDF parameter table)
# ---------------------------------------------------------------------------
dt = 1.0                              # [h]   General

# TES
tau_loss = 200.0                      # [h]
tau_in = tau_out = 1.0                # [h]
eta_in_TES = eta_out_TES = 0.95       # [-]
E_nom_TES = 1000.0                    # [kWh]
E_min_TES = 0.0                       # [kWh]
Qin_min_TES = 0.0                     # [kW]
Qout_min_TES = 0.0                    # [kW]

# Boiler (identical units i in {1,2})
Qout_nom_B = 530.0                    # [kW]
eta_nom_B = 0.8                       # [-]
lam_in_min_B = 0.173                  # [-]
lam_out_min_B = 0.2                   # [-]

# CHP (identical units i in {1,2})
Qout_nom_CHP = 470.0                  # [kW] nominal thermal output
Pout_nom_CHP = 380.0                  # [kW] nominal electrical output
eta_nom_CHP_th = 0.481                # [-]
eta_nom_CHP_el = 0.389                # [-]
lam_in_min_CHP = 0.582                # [-]  shared minimal fuel load (th = el)
lam_out_min_CHP_th = 0.622            # [-]
lam_out_min_CHP_el = 0.5             # [-]

# Part-load slopes  beta_j = (1 - lam_in_min) / (1 - lam_out_min)
beta_B = (1.0 - lam_in_min_B) / (1.0 - lam_out_min_B)
beta_CHP_th = (1.0 - lam_in_min_CHP) / (1.0 - lam_out_min_CHP_th)
beta_CHP_el = (1.0 - lam_in_min_CHP) / (1.0 - lam_out_min_CHP_el)

# ---------------------------------------------------------------------------
# 2. Exact ZOH discretization of the TES ODE  (section 1.1)
#       x' = A x + B u,  A = -1/tau_loss,  B = [eta_in_TES, -1/eta_out_TES]
#    Exact: Ad = exp(A dt),  Bd = (int_0^dt exp(A tau) dtau) * B
#    Computed via the van Loan augmented matrix exponential:
#       expm([[A, B],[0, 0]] * dt) = [[Ad, Bd],[0, I]]
# ---------------------------------------------------------------------------
A = np.array([[-1.0 / tau_loss]])                       # 1 x 1
Bc = np.array([[eta_in_TES, -1.0 / eta_out_TES]])       # 1 x 2
n_x, n_u = A.shape[0], Bc.shape[1]
M = np.zeros((n_x + n_u, n_x + n_u))
M[:n_x, :n_x] = A
M[:n_x, n_x:] = Bc
Md = expm(M * dt)
a = float(Md[0, 0])                  # ~ 0.99501248
b1, b2 = (float(v) for v in Md[:n_x, n_x:].ravel())     # ~ 0.94762895, -1.05000438

# ---------------------------------------------------------------------------
# 3. Demand time series from CSV
# ---------------------------------------------------------------------------
df = pd.read_csv(DEMAND_CSV)
P_D_series = df["hourly electricity demand [kW]"].to_numpy()
Q_D_series = df["hourly heat demand [kW]"].to_numpy()
N = len(df)                          # number of intervals (= 168 for tf = 168 h, dt = 1 h)

# ---------------------------------------------------------------------------
# 4. Model
# ---------------------------------------------------------------------------
m = ConcreteModel("ModE_P5_TES_dispatch")

m.K = RangeSet(0, N - 1)             # time intervals
m.B = Set(initialize=[1, 2])         # boilers
m.CHP = Set(initialize=[1, 2])       # CHP units

m.Q_D = Param(m.K, initialize={k: float(Q_D_series[k]) for k in range(N)})
m.P_D = Param(m.K, initialize={k: float(P_D_series[k]) for k in range(N)})

# --- Independent (x_tilde) decision variables ---
m.Qin_B = Var(m.B, m.K, domain=NonNegativeReals)        # boiler fuel input [kW]
m.Qin_CHP = Var(m.CHP, m.K, domain=NonNegativeReals)    # CHP fuel input [kW]
m.Qin_TES = Var(m.K, domain=NonNegativeReals)           # TES charge flux [kW]
m.Qout_TES = Var(m.K, domain=NonNegativeReals)          # TES discharge flux [kW]
m.Pgrid = Var(m.K, domain=PGRID_DOMAIN)                 # grid power [kW]
m.dB = Var(m.B, m.K, domain=Binary)                     # boiler on/off
m.dCHP = Var(m.CHP, m.K, domain=Binary)                 # CHP on/off
m.din_TES = Var(m.K, domain=Binary)                     # TES charging flag
m.dout_TES = Var(m.K, domain=Binary)                    # TES discharging flag

# --- Auxiliary (y) decision variables, kept for the SPARSE formulation ---
m.E_TES = Var(m.K, domain=NonNegativeReals, bounds=(E_min_TES, E_nom_TES))  # (II)+(III)
m.Qout_B = Var(m.B, m.K, domain=Reals)                  # boiler thermal output [kW]
m.Qout_CHP = Var(m.CHP, m.K, domain=Reals)              # CHP thermal output [kW]
m.Pout_CHP = Var(m.CHP, m.K, domain=Reals)              # CHP electrical output [kW]

# Per-step gas/electricity energy (y-entries E_G, E_el). Kept as named Expressions
# rather than Vars+equalities: identical math, no redundant rows. Swap to Var if
# you genuinely need them as explicit columns.
@m.Expression(m.K)
def E_G(m, k):
    return dt * (sum(m.Qin_B[i, k] for i in m.B) + sum(m.Qin_CHP[i, k] for i in m.CHP))

@m.Expression(m.K)
def E_el(m, k):
    return dt * m.Pgrid[k]

# ---------------------------------------------------------------------------
# 5. Constraints  (decorator style, indexed over the relevant sets)
# ---------------------------------------------------------------------------

# --- TES ---
@m.Constraint(m.K)                                      # TES (I) discretized + (VIII) cyclic
def tes_dynamics(m, k):
    k_next = (k + 1) % N                                # wrap-around enforces E_TES(0)=E_TES(tf)
    return m.E_TES[k_next] == a * m.E_TES[k] + b1 * m.Qin_TES[k] + b2 * m.Qout_TES[k]

@m.Constraint(m.K)                                      # TES (IV)
def tes_charge_ub(m, k):
    return m.Qin_TES[k] <= m.din_TES[k] * E_nom_TES / tau_in

@m.Constraint(m.K)                                      # TES (V)
def tes_charge_lb(m, k):
    return m.Qin_TES[k] >= m.din_TES[k] * Qin_min_TES

@m.Constraint(m.K)                                      # TES (VI)
def tes_discharge_ub(m, k):
    return m.Qout_TES[k] <= m.dout_TES[k] * E_nom_TES / tau_out

@m.Constraint(m.K)                                      # TES (VII)
def tes_discharge_lb(m, k):
    return m.Qout_TES[k] >= m.dout_TES[k] * Qout_min_TES

@m.Constraint(m.K)                                      # TES (IX)
def tes_no_simultaneous(m, k):
    return m.din_TES[k] + m.dout_TES[k] <= 1

# --- Boilers ---
@m.Constraint(m.B, m.K)                                 # Boiler (I): part-load thermal output
def boiler_output(m, i, k):
    return m.Qout_B[i, k] == Qout_nom_B * (
        m.dB[i, k] * lam_out_min_B
        + (1.0 / beta_B) * (m.Qin_B[i, k] * eta_nom_B / Qout_nom_B - m.dB[i, k] * lam_in_min_B)
    )

@m.Constraint(m.B, m.K)                                 # Boiler (II): fuel upper bound
def boiler_fuel_ub(m, i, k):
    return m.Qin_B[i, k] <= m.dB[i, k] * Qout_nom_B / eta_nom_B

@m.Constraint(m.B, m.K)                                 # Boiler (III): fuel lower bound
def boiler_fuel_lb(m, i, k):
    return m.Qin_B[i, k] >= m.dB[i, k] * lam_in_min_B * Qout_nom_B / eta_nom_B

# --- CHPs ---
@m.Constraint(m.CHP, m.K)                               # CHP (I): part-load thermal output
def chp_heat(m, i, k):
    return m.Qout_CHP[i, k] == Qout_nom_CHP * (
        m.dCHP[i, k] * lam_out_min_CHP_th
        + (1.0 / beta_CHP_th) * (m.Qin_CHP[i, k] * eta_nom_CHP_th / Qout_nom_CHP - m.dCHP[i, k] * lam_in_min_CHP)
    )

@m.Constraint(m.CHP, m.K)                               # CHP (II): part-load electrical output
def chp_power(m, i, k):
    return m.Pout_CHP[i, k] == Pout_nom_CHP * (
        m.dCHP[i, k] * lam_out_min_CHP_el
        + (1.0 / beta_CHP_el) * (m.Qin_CHP[i, k] * eta_nom_CHP_el / Pout_nom_CHP - m.dCHP[i, k] * lam_in_min_CHP)
    )

@m.Constraint(m.CHP, m.K)                               # CHP (III): fuel upper bound (thermal basis)
def chp_fuel_ub(m, i, k):
    return m.Qin_CHP[i, k] <= m.dCHP[i, k] * Qout_nom_CHP / eta_nom_CHP_th

@m.Constraint(m.CHP, m.K)                               # CHP (IV): fuel lower bound (electrical basis)
def chp_fuel_lb(m, i, k):
    return m.Qin_CHP[i, k] >= m.dCHP[i, k] * lam_in_min_CHP * Pout_nom_CHP / eta_nom_CHP_el

# --- Balances ---
@m.Constraint(m.K)                                      # Demand (I): heat balance
def heat_balance(m, k):
    return (
        sum(m.Qout_CHP[i, k] for i in m.CHP)
        + sum(m.Qout_B[i, k] for i in m.B)
        + m.Qout_TES[k] - m.Qin_TES[k]
        == m.Q_D[k]
    )

@m.Constraint(m.K)                                      # Demand (II): power balance
def power_balance(m, k):
    return sum(m.Pout_CHP[i, k] for i in m.CHP) + m.Pgrid[k] == m.P_D[k]

# ---------------------------------------------------------------------------
# 6. Objective
# ---------------------------------------------------------------------------
@m.Objective(sense=minimize)
def total_cost(m):
    return cG * sum(m.E_G[k] for k in m.K) + cel * sum(m.E_el[k] for k in m.K)


def plot_dispatch_results(dispatch: pd.DataFrame, output_path: str = "dispatch_overview_MILP.png"):
    """Create a compact dashboard of unit commitment and energy flows."""
    from matplotlib.patches import Patch

    k = dispatch["k"].to_numpy()

    on_matrix = np.vstack([
        (dispatch["dB1"].to_numpy() > 0.5).astype(float),
        (dispatch["dB2"].to_numpy() > 0.5).astype(float),
        (dispatch["dCHP1"].to_numpy() > 0.5).astype(float),
        (dispatch["dCHP2"].to_numpy() > 0.5).astype(float),
    ])

    fig, axes = plt.subplots(4, 1, figsize=(14, 13), sharex=True)

    # 1) Unit commitment timeline
    axes[0].imshow(
        on_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="YlGn",
        vmin=0,
        vmax=1,
        extent=[k[0] - 0.5, k[-1] + 0.5, -0.5, 3.5],
        origin="lower",
    )
    axes[0].set_yticks([0, 1, 2, 3])
    axes[0].set_yticklabels(["Boiler 1", "Boiler 2", "CHP 1", "CHP 2"])
    axes[0].set_title("Unit Commitment (On/Off)")
    axes[0].set_ylabel("Units")
    axes[0].legend(
        handles=[
            Patch(facecolor="#ffffe5", edgecolor="black", label="Off"),
            Patch(facecolor="#238443", edgecolor="black", label="On"),
        ],
        loc="upper right",
        title="Status",
    )

    # 2) TES charging/discharging and TES state of charge
    axes[1].bar(k, dispatch["Qout_TES"], width=0.9, label="TES discharge", color="#2C7FB8", alpha=0.9)
    axes[1].bar(k, -dispatch["Qin_TES"], width=0.9, label="TES charge", color="#EF3B2C", alpha=0.7)
    axes[1].axhline(0.0, color="black", linewidth=0.9)
    axes[1].set_ylabel("TES power [kW]")
    axes[1].set_title("TES Operation")
    axes[1].grid(True, axis="y", linestyle=":", linewidth=0.8, alpha=0.7)

    ax1_twin = axes[1].twinx()
    ax1_twin.plot(k, dispatch["E_TES"], color="#6A3D9A", linewidth=2.0, label="TES energy")
    ax1_twin.set_ylabel("TES energy [kWh]")

    lines1, labels1 = axes[1].get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    axes[1].legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    # 3) Grid imports and electric balance context
    p_chp_total = dispatch["Pout_CHP1"] + dispatch["Pout_CHP2"]
    axes[2].fill_between(k, 0, dispatch["Pgrid"], step="mid", alpha=0.45, color="#FDAE61", label="Grid import")
    axes[2].plot(k, p_chp_total, color="#1B9E77", linewidth=2, label="CHP electric output")
    axes[2].plot(k, dispatch["P_D"], color="#111111", linewidth=1.7, linestyle="--", label="Electric demand")
    axes[2].set_title("Electrical Supply Mix")
    axes[2].set_ylabel("Power [kW]")
    axes[2].set_xlabel("Time step k [-]")
    axes[2].grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    axes[2].legend(loc="upper right")

    # 4) Heat supply mix and gas purchase per step
    q_boiler_total = dispatch["Qout_B1"] + dispatch["Qout_B2"]
    q_chp_total = dispatch["Qout_CHP1"] + dispatch["Qout_CHP2"]
    q_tes_net = dispatch["Qout_TES"] - dispatch["Qin_TES"]
    q_gas_total = dispatch["Qin_B1"] + dispatch["Qin_B2"] + dispatch["Qin_CHP1"] + dispatch["Qin_CHP2"]

    axes[3].plot(k, q_boiler_total, color="#E31A1C", linewidth=1.8, label="Boiler heat output")
    axes[3].plot(k, q_chp_total, color="#FF7F00", linewidth=1.8, label="CHP heat output")
    axes[3].plot(k, q_tes_net, color="#2C7FB8", linewidth=1.8, label="TES net heat (discharge-charge)")
    axes[3].plot(k, dispatch["Q_D"], color="#111111", linewidth=1.7, linestyle="--", label="Heat demand")
    axes[3].set_title("Heat Supply and Gas Purchase")
    axes[3].set_ylabel("Heat flow [kW]")
    axes[3].set_xlabel("Time step k [-]")
    axes[3].grid(True, linestyle=":", linewidth=0.8, alpha=0.7)

    ax3_twin = axes[3].twinx()
    ax3_twin.bar(k, q_gas_total, width=0.85, alpha=0.22, color="#33A02C", label="Gas purchased (fuel input)")
    ax3_twin.set_ylabel("Gas purchased [kW_fuel]")

    lines3, labels3 = axes[3].get_legend_handles_labels()
    lines4, labels4 = ax3_twin.get_legend_handles_labels()
    axes[3].legend(lines3 + lines4, labels3 + labels4, loc="upper right")

    fig.suptitle(f"Operational Dispatch Overview — gas={cG:.3f}, el={cel:.3f}", fontsize=14, y=0.99)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

# ---------------------------------------------------------------------------
# 7. Solve with Gurobi
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"N = {N} intervals, dt = {dt} h  ->  discretization a={a:.6f}, b1={b1:.6f}, b2={b2:.6f}")
    print(f"beta_B={beta_B:.5f}, beta_CHP_th={beta_CHP_th:.5f}, beta_CHP_el={beta_CHP_el:.5f}")

    solver = SolverFactory("gurobi")
    solver.options["MIPGap"] = 1e-3
    solver.options["TimeLimit"] = 300
    results = solver.solve(m, tee=True)

    print("\nSolver status :", results.solver.status)
    print("Termination   :", results.solver.termination_condition)
    print(f"Total cost    : {value(m.total_cost):.2f}")

    # --- Extract dispatch ---
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
            **{f"dB{i}": value(m.dB[i, k]) for i in m.B},
            **{f"dCHP{i}": value(m.dCHP[i, k]) for i in m.CHP},
            **{f"Qin_B{i}": value(m.Qin_B[i, k]) for i in m.B},
            **{f"Qout_B{i}": value(m.Qout_B[i, k]) for i in m.B},
            **{f"Qin_CHP{i}": value(m.Qin_CHP[i, k]) for i in m.CHP},
            **{f"Qout_CHP{i}": value(m.Qout_CHP[i, k]) for i in m.CHP},
            **{f"Pout_CHP{i}": value(m.Pout_CHP[i, k]) for i in m.CHP},
        })
    dispatch = pd.DataFrame(rows)
    dispatch.to_csv("dispatch_result_MILP.csv", index=False)
    print("Dispatch written to dispatch_result_MILP.csv")

    plot_dispatch_results(dispatch, output_path="dispatch_overview_MILP.png")
    print("Dispatch visualization written to dispatch_overview_MILP.png")

