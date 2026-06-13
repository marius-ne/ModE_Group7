# src/optimization.py
"""
Optimization module for minimzation of the OPEX of a multi-energy system (MES).
"""
import string
from pathlib import Path
import pandas as pd
import numpy as np
import pyomo
import pyomo.environ as pyo
import gurobipy

# == System parameters =====================

# -- Time horizon ----------------------------
DELTA_T = 1.0 # time step [h]
N = 168 # number of time steps (7 days × 24 h)

# -- TES parameters --------------------------
tau_loss = 200.0 # heat-loss time constant [h]
tau_in = 1.0 # minimum charging time constant [h]
tau_out = 1.0 # minimum discharging time constant [h]
eta_in_TES = 0.95 # charging efficiency [-]
eta_out_TES = 0.95 # discharging efficiency [-]
E_nom_TES = 1000.0 # nominal storage capacity [kWh]
E_min_TES = 0.0 # minimum energy level [kWh]
Q_in_min_TES = 0.0 # minimum charging flux [kW]
Q_out_min_TES = 0.0 # minimum discharging flux [kW]

# Exact matrix-exponential discretization
a_TES = np.exp(-DELTA_T / tau_loss)
b1_TES = eta_in_TES  * tau_loss * (1 - a_TES)
b2_TES = -(1/eta_out_TES) * tau_loss * (1 - a_TES)

# -- Boiler parameters (identical for B1 and B2) -----
Q_out_nom_B = 530.0 # nominal thermal output [kW]
eta_nom_B = 0.80 # nominal efficiency [-]
lambda_in_min_B = 0.173
lambda_out_min_B = 0.200
beta_B = (1 - lambda_in_min_B) / (1 - lambda_out_min_B)

# -- CHP parameters (identical for CHP1 and CHP2) ------
Q_out_nom_CHP = 470.0  # nominal thermal output [kW]
P_out_nom_CHP = 380.0  # nominal electrical output [kWh]
eta_nom_CHP_th = 0.481 # thermal efficiency [-]
eta_nom_CHP_el = 0.389 # electrical efficiency [-]
lambda_in_min_CHP = 0.582
lambda_out_min_CHP_th = 0.622
lambda_out_min_CHP_el = 0.500
beta_CHP_th = (1 - lambda_in_min_CHP) / (1 - lambda_out_min_CHP_th)
beta_CHP_el = (1 - lambda_in_min_CHP) / (1 - lambda_out_min_CHP_el)


def load_demands(csv_path: Path) -> tuple[pd.Series, pd.Series]:
    """
    Loads demand data from csv file.
    :param csv_path: Path to csv file containing demand data
    :return: Tuple of two pandas Series: (Q_D, P_D)
     - Q_D: Series containing heat demand data [kW]
     - P_D: Series containing electricity demand data [kW]
     The csv file is expected to have the following structure:
     | Time | Electricity Demand (kW) | Heat Demand (kW) |
    """
    try:
        df = pd.read_csv(csv_path, index_col=0)
        Q_D = df.iloc[:, 1] # heat demand [kW]
        P_D = df.iloc[:, 0] # electricity demand [kW]

        return Q_D, P_D

    except FileNotFoundError:
        raise FileNotFoundError()


def build_milp(
        Q_D: pd.Series,
        P_D: pd.Series,
        c_g: float,
        c_el: float
) -> pyo.ConcreteModel:
    """
    Builds a mixed-integer linear programming (MILP) model for the multi-energy system optimization problem.
    The model minimizes the operational expenditure (OPEX) of the system while satisfying the demand
    and operational constraints of the components.
    The model is built using the Pyomo optimization modeling language and solved by Gurobi.
    :param Q_D: Series containing heat demand data [kW]
    :param P_D: Series containing electricity demand data [kW]
    :param c_g: Gas price [€/kWh]
    :param c_el: Electricity price [€/kWh]
    :return: pyo.ConcreteModel MILP optimization model
    """
    m = pyo.ConcreteModel(name="MILP")


    # ─────────────────────────────────────────────────────────────────────────
    # 3.1  SETS
    # ─────────────────────────────────────────────────────────────────────────
    m.K = pyo.Set(initialize=range(1, N + 1), doc="Time steps k = 1..N")
    m.K0 = pyo.Set(initialize=range(0, N + 1), doc="Extended set k = 0..N for TES state")
    m.B = pyo.Set(initialize=[1, 2], doc="Boiler units")
    m.C = pyo.Set(initialize=[1, 2], doc="CHP units")


    # ─────────────────────────────────────────────────────────────────────────
    # 3.2  PARAMETERS
    # ─────────────────────────────────────────────────────────────────────────
    m.c_g = pyo.Param(initialize=c_g, doc="Gas price [€/kWh]")
    m.c_el = pyo.Param(initialize=c_el, doc="Electricity price [€/kWh]")
    m.dt = pyo.Param(initialize=DELTA_T, doc="Time step [h]")

    # Demand (indexed by time step k)
    m.Q_D = pyo.Param(m.K, initialize={k: float(Q_D[k]) for k in m.K}, doc="Heat demand [kW]")
    m.P_D = pyo.Param(m.K, initialize={k: float(P_D[k]) for k in m.K}, doc="Electrical demand [kW]")

    # TES
    m.a_TES = pyo.Param(initialize=a_TES, doc="TES discrete state matrix")
    m.b1_TES = pyo.Param(initialize=b1_TES, doc="TES input matrix (charging)")
    m.b2_TES = pyo.Param(initialize=b2_TES, doc="TES input matrix (discharging)")
    m.E_nom_TES = pyo.Param(initialize=E_nom_TES, doc="TES nominal capacity [kWh]")
    m.E_min_TES = pyo.Param(initialize=E_min_TES, doc="TES minimum energy [kWh]")
    m.tau_in = pyo.Param(initialize=tau_in, doc="TES charge rate parameter [h]")
    m.tau_out = pyo.Param(initialize=tau_out, doc="TES discharge rate parameter [h]")
    m.Q_in_min_TES = pyo.Param(initialize=Q_in_min_TES)
    m.Q_out_min_TES = pyo.Param(initialize=Q_out_min_TES)

    # Boiler (same for both units)
    m.Q_out_nom_B = pyo.Param(initialize=Q_out_nom_B)
    m.eta_nom_B = pyo.Param(initialize=eta_nom_B)
    m.lam_in_min_B = pyo.Param(initialize=lambda_in_min_B)
    m.lam_out_min_B = pyo.Param(initialize=lambda_out_min_B)
    m.beta_B = pyo.Param(initialize=beta_B)

    # CHP (same for both units)
    m.Q_out_nom_C = pyo.Param(initialize=Q_out_nom_CHP)
    m.P_out_nom_C = pyo.Param(initialize=P_out_nom_CHP)
    m.eta_nom_C_th = pyo.Param(initialize=eta_nom_CHP_th)
    m.eta_nom_C_el = pyo.Param(initialize=eta_nom_CHP_el)
    m.lam_in_min_C = pyo.Param(initialize=lambda_in_min_CHP)
    m.lam_out_min_C_th = pyo.Param(initialize=lambda_out_min_CHP_th)
    m.lam_out_min_C_el = pyo.Param(initialize=lambda_out_min_CHP_el)
    m.beta_C_th = pyo.Param(initialize=beta_CHP_th)
    m.beta_C_el = pyo.Param(initialize=beta_CHP_el)


    # ─────────────────────────────────────────────────────────────────────────
    # 3.3  DECISION VARIABLES
    # ─────────────────────────────────────────────────────────────────────────
    # ── TES ──
    m.E_TES = pyo.Var(m.K0, domain=pyo.NonNegativeReals, doc="TES energy content [kWh]")  # ETES,k (k=0..N)
    m.Q_in_TES = pyo.Var(m.K, domain=pyo.NonNegativeReals, doc="TES charging flux [kW]")
    m.Q_out_TES = pyo.Var(m.K, domain=pyo.NonNegativeReals, doc="TES discharging flux [kW]")
    m.delta_in_TES = pyo.Var(m.K, domain=pyo.Binary, doc="TES charging binary")
    m.delta_out_TES = pyo.Var(m.K, domain=pyo.Binary, doc="TES discharging binary")

    # ── Boilers ──
    m.Q_in_B = pyo.Var(m.B, m.K, domain=pyo.NonNegativeReals, doc="Boiler gas input [kW]")
    m.Q_out_B = pyo.Var(m.B, m.K, domain=pyo.NonNegativeReals, doc="Boiler heat output [kW]")
    m.delta_B = pyo.Var(m.B, m.K, domain=pyo.Binary, doc="Boiler on/off binary")

    # ── CHPs ──
    m.Q_in_CHP = pyo.Var(m.C, m.K, domain=pyo.NonNegativeReals, doc="CHP gas input [kW]")
    m.Q_out_CHP = pyo.Var(m.C, m.K, domain=pyo.NonNegativeReals, doc="CHP thermal output [kW]")
    m.P_out_CHP = pyo.Var(m.C, m.K, domain=pyo.NonNegativeReals, doc="CHP electrical output [kW]")
    m.delta_CHP = pyo.Var(m.C, m.K, domain=pyo.Binary, doc="CHP on/off binary")

    # ── Grid ──
    m.P_grid = pyo.Var(m.K, domain=pyo.NonNegativeReals, doc="Grid electricity import [kW]")

    # ── Auxiliary energy variables (for sparse objective) ──
    m.E_gas = pyo.Var(m.K, domain=pyo.Reals, doc="Gas energy per step [kWh]")
    m.E_el = pyo.Var(m.K, domain=pyo.Reals, doc="Grid electricity per step [kWh]")


    # ─────────────────────────────────────────────────────────────────────────
    # 3.4  OBJECTIVE
    # ─────────────────────────────────────────────────────────────────────────
    def obj_rule(m):
        return m.c_g * sum(m.E_gas[k] for k in m.K) + m.c_el * sum(m.E_el[k] for k in m.K)
    m.OBJ = pyo.Objective(rule=obj_rule, sense=pyo.minimize)


    # ─────────────────────────────────────────────────────────────────────────
    # 3.5  CONSTRAINTS
    # ─────────────────────────────────────────────────────────────────────────
    # ── Auxiliary definitions ────────────────────────────────────────────────
    # Gas energy consumed at time step k
    def e_gas_def(m, k):
        return m.E_gas[k] == m.dt * (
                sum(m.Q_in_B[i, k] for i in m.B) +
                sum(m.Q_in_CHP[i, k] for i in m.C)
        )
    m.c_E_gas = pyo.Constraint(m.K, rule=e_gas_def)

    # Grid electricity energy consumed at time step k
    def e_el_def(m, k):
        return m.E_el[k] == m.dt * m.P_grid[k]

    m.c_E_el = pyo.Constraint(m.K, rule=e_el_def)

    # ── TES constraints ──────────────────────────────────────────────────────
    # (I) Exact discrete-time dynamics (matrix-exponential discretization)
    def tes_dynamics(m, k):
        return m.E_TES[k] == (m.a_TES * m.E_TES[k - 1]
                              + m.b1_TES * m.Q_in_TES[k]
                              + m.b2_TES * m.Q_out_TES[k])
    m.c_TES_dyn = pyo.Constraint(m.K, rule=tes_dynamics)

    # (II) Upper bound on TES energy
    def tes_ub(m, k):
        return m.E_TES[k] <= m.E_nom_TES
    m.c_TES_ub = pyo.Constraint(m.K0, rule=tes_ub)

    # (III) Lower bound on TES energy
    def tes_lb(m, k):
        return m.E_TES[k] >= m.E_min_TES
    m.c_TES_lb = pyo.Constraint(m.K0, rule=tes_lb)

    # (IV) Maximum charging flux (big-M with binary)
    def tes_in_ub(m, k):
        return m.Q_in_TES[k] <= m.delta_in_TES[k] * (m.E_nom_TES / m.tau_in)
    m.c_TES_in_ub = pyo.Constraint(m.K, rule=tes_in_ub)

    # (V) Minimum charging flux (if charging is active)
    def tes_in_lb(m, k):
        return m.Q_in_TES[k] >= m.delta_in_TES[k] * m.Q_in_min_TES
    m.c_TES_in_lb = pyo.Constraint(m.K, rule=tes_in_lb)

    # (VI) Maximum discharging flux
    def tes_out_ub(m, k):
        return m.Q_out_TES[k] <= m.delta_out_TES[k] * (m.E_nom_TES / m.tau_out)
    m.c_TES_out_ub = pyo.Constraint(m.K, rule=tes_out_ub)

    # (VII) Minimum discharging flux (if discharging is active)
    def tes_out_lb(m, k):
        return m.Q_out_TES[k] >= m.delta_out_TES[k] * m.Q_out_min_TES
    m.c_TES_out_lb = pyo.Constraint(m.K, rule=tes_out_lb)

    # (VIII) Cyclic constraint: initial == final state
    def tes_cycle(m):
        return m.E_TES[0] == m.E_TES[N]
    m.c_TES_cycle = pyo.Constraint(rule=tes_cycle)

    # (IX) Cannot charge AND discharge simultaneously
    def tes_mutex(m, k):
        return m.delta_in_TES[k] + m.delta_out_TES[k] <= 1
    m.c_TES_mutex = pyo.Constraint(m.K, rule=tes_mutex)

    # ── Boiler constraints ───────────────────────────────────────────────────
    # (I) Part-load efficiency curve:
    def boiler_pl(m, i, k):
        return (m.Q_out_B[i, k] == m.Q_out_nom_B *
                (m.delta_B[i, k] * m.lam_out_min_B
                 + (1.0 / m.beta_B) * (m.Q_in_B[i, k] * m.eta_nom_B / m.Q_out_nom_B - m.delta_B[i, k] * m.lam_in_min_B)
                )
        )
    m.c_B_pl = pyo.Constraint(m.B, m.K, rule=boiler_pl)

    # (II) Upper bound on gas input (forces Q_in_B = 0 when delta_B = 0)
    def boiler_in_ub(m, i, k):
        return m.Q_in_B[i, k] <= m.delta_B[i, k] * (m.Q_out_nom_B / m.eta_nom_B)
    m.c_B_in_ub = pyo.Constraint(m.B, m.K, rule=boiler_in_ub)

    # (III) Lower bound on gas input (minimum part load)
    def boiler_in_lb(m, i, k):
        return m.Q_in_B[i, k] >= m.delta_B[i, k] * m.lam_in_min_B * (m.Q_out_nom_B / m.eta_nom_B)
    m.c_B_in_lb = pyo.Constraint(m.B, m.K, rule=boiler_in_lb)

    # ── CHP constraints ──────────────────────────────────────────────────────
    # (I) Thermal part-load curve
    def chp_th_pl(m, i, k):
        return (m.Q_out_CHP[i, k] == m.Q_out_nom_C *
                (m.delta_CHP[i, k] * m.lam_out_min_C_th
                 + (1.0 / m.beta_C_th) * (m.Q_in_CHP[i, k] * m.eta_nom_C_th / m.Q_out_nom_C - m.delta_CHP[i, k] * m.lam_in_min_C)
                )
        )
    m.c_CHP_th = pyo.Constraint(m.C, m.K, rule=chp_th_pl)

    # (II) Electrical part-load curve
    def chp_el_pl(m, i, k):
        return (m.P_out_CHP[i, k] == m.P_out_nom_C *
                (m.delta_CHP[i, k] * m.lam_out_min_C_el
                 + (1.0 / m.beta_C_el) * (m.Q_in_CHP[i, k] * m.eta_nom_C_el / m.P_out_nom_C - m.delta_CHP[i, k] * m.lam_in_min_C)
                )
        )
    m.c_CHP_el = pyo.Constraint(m.C, m.K, rule=chp_el_pl)

    # (III) Upper bound on CHP gas input
    def chp_in_ub(m, i, k):
        return m.Q_in_CHP[i, k] <= m.delta_CHP[i, k] * (m.Q_out_nom_C / m.eta_nom_C_th)
    m.c_CHP_in_ub = pyo.Constraint(m.C, m.K, rule=chp_in_ub)

    # (IV) Lower bound on CHP gas input
    def chp_in_lb(m, i, k):
        return m.Q_in_CHP[i, k] >= m.delta_CHP[i, k] * m.lam_in_min_C * (m.P_out_nom_C / m.eta_nom_C_el)
    m.c_CHP_in_lb = pyo.Constraint(m.C, m.K, rule=chp_in_lb)

    # ── Demand satisfaction ──────────────────────────────────────────────────
    # (I) Heat demand balance:
    def heat_balance(m, k):
        tes_net = m.Q_out_TES[k] - m.Q_in_TES[k]
        return (sum(m.Q_out_CHP[i, k] for i in m.C) + sum(m.Q_out_B[i, k] for i in m.B) + tes_net == m.Q_D[k])
    m.c_heat = pyo.Constraint(m.K, rule=heat_balance)

    # (II) Electricity demand balance
    def elec_balance(m, k):
        return (sum(m.P_out_CHP[i, k] for i in m.C) + m.P_grid[k] == m.P_D[k])
    m.c_elec = pyo.Constraint(m.K, rule=elec_balance)

    return m


def solve_model(model: pyo.ConcreteModel, solver_name: str = "gurobi", **solver_options):
    """
    Solves the given optimization model using the specified solver.
    :param model: Pyomo ConcreteModel to be solved
    :param solver_name: Name of the solver to use (default: "gurobi")
    :param solver_options: Options for the solver
    :return: Optimization results of the model
    """
    try:
        solver = pyo.SolverFactory(solver_name)
        solver.options.update(solver_options)

        results = solver.solve(model, tee=True)

        return results
    except Exception as e:
        print(f"Error solving model: {e}")
        raise e


def extract_solution(m: pyo.ConcreteModel) -> pd.DataFrame:
    """
    Extracts the solution from the optimization model and returns it as a pandas DataFrame.
    :param m: Pyomo ConcreteModel
    :return: Pandas DataFrame
    """
    solution = []
    for k in m.K:
        sol = {
            "k": k,
            "Q_D": pyo.value(m.Q_D[k]),
            "P_D": pyo.value(m.P_D[k]),

            # TES
            "E_TES": pyo.value(m.E_TES[k]),
            "Q_in_TES": pyo.value(m.Q_in_TES[k]),
            "Q_out_TES": pyo.value(m.Q_out_TES[k]),
            "delta_in_TES": pyo.value(m.delta_in_TES[k]),
            "delta_out_TES": pyo.value(m.delta_out_TES[k]),

            # Boilers
            "Q_in_B1":   pyo.value(m.Q_in_B[1, k]),
            "Q_in_B2":   pyo.value(m.Q_in_B[2, k]),
            "Q_out_B1":  pyo.value(m.Q_out_B[1, k]),
            "Q_out_B2":  pyo.value(m.Q_out_B[2, k]),

            # CHPs
            "Q_in_CHP1": pyo.value(m.Q_in_CHP[1, k]),
            "Q_in_CHP2": pyo.value(m.Q_in_CHP[2, k]),
            "Q_out_CHP1": pyo.value(m.Q_out_CHP[1, k]),
            "Q_out_CHP2": pyo.value(m.Q_out_CHP[2, k]),
            "P_out_CHP1": pyo.value(m.P_out_CHP[1, k]),
            "P_out_CHP2": pyo.value(m.P_out_CHP[2, k]),

            # Grid
            "P_grid": pyo.value(m.P_grid[k]),
        }

        solution.append(sol)

    return pd.DataFrame(solution)