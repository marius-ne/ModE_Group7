# src/optimization/core.py
"""
Optimization module for minimzation of the OPEX of a multi-energy system (MES).
"""
from pathlib import Path
import pandas as pd
import numpy as np
import pyomo.environ as pyo
import gurobipy
import datetime
import json
from src.misc.constants import RESULTS_DIR

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
        c_el: float,
        *,
        normalize: bool = False,
        strict_demand_satisfaction: bool = True,
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


    # -------------------------------------------------------------------------
    # 3.1  SETS
    # -------------------------------------------------------------------------
    m.K = pyo.Set(initialize=range(1, N + 1), doc="Time steps k = 1..N")
    m.K0 = pyo.Set(initialize=range(0, N + 1), doc="Extended set k = 0..N for TES state")
    m.B = pyo.Set(initialize=[1, 2], doc="Boiler units")
    m.C = pyo.Set(initialize=[1, 2], doc="CHP units")


    # -------------------------------------------------------------------------
    # 3.2  PARAMETERS
    # -------------------------------------------------------------------------
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


    # -------------------------------------------------------------------------
    # 3.3  DECISION VARIABLES
    # -------------------------------------------------------------------------
    # -- TES --
    m.E_TES = pyo.Var(m.K0, domain=pyo.NonNegativeReals, doc="TES energy content [kWh]")  # ETES,k (k=0..N)
    m.Q_in_TES = pyo.Var(m.K, domain=pyo.NonNegativeReals, doc="TES charging flux [kW]")
    m.Q_out_TES = pyo.Var(m.K, domain=pyo.NonNegativeReals, doc="TES discharging flux [kW]")
    m.delta_in_TES = pyo.Var(m.K, domain=pyo.Binary, doc="TES charging binary")
    m.delta_out_TES = pyo.Var(m.K, domain=pyo.Binary, doc="TES discharging binary")

    # -- Boilers --
    m.Q_in_B = pyo.Var(m.B, m.K, domain=pyo.NonNegativeReals, doc="Boiler gas input [kW]")
    m.Q_out_B = pyo.Var(m.B, m.K, domain=pyo.NonNegativeReals, doc="Boiler heat output [kW]")
    m.delta_B = pyo.Var(m.B, m.K, domain=pyo.Binary, doc="Boiler on/off binary")

    # -- CHPs --
    m.Q_in_CHP = pyo.Var(m.C, m.K, domain=pyo.NonNegativeReals, doc="CHP gas input [kW]")
    m.Q_out_CHP = pyo.Var(m.C, m.K, domain=pyo.NonNegativeReals, doc="CHP thermal output [kW]")
    m.P_out_CHP = pyo.Var(m.C, m.K, domain=pyo.NonNegativeReals, doc="CHP electrical output [kW]")
    m.delta_CHP = pyo.Var(m.C, m.K, domain=pyo.Binary, doc="CHP on/off binary")

    # -- Grid --
    m.P_grid = pyo.Var(m.K, domain=pyo.NonNegativeReals, doc="Grid electricity import [kW]")

    # -- Auxiliary energy variables (for sparse objective) --
    m.E_gas = pyo.Var(m.K, domain=pyo.Reals, doc="Gas energy per step [kWh]")
    m.E_el = pyo.Var(m.K, domain=pyo.Reals, doc="Grid electricity per step [kWh]")


    # -------------------------------------------------------------------------
    # 3.4  OBJECTIVE
    # -------------------------------------------------------------------------
    def obj_rule(m):
        if normalize:
            return (c_g / c_el) * sum(m.E_gas[k] for k in m.K) + sum(m.E_el[k] for k in m.K)
        return m.c_g * sum(m.E_gas[k] for k in m.K) + m.c_el * sum(m.E_el[k] for k in m.K)
    m.OBJ = pyo.Objective(rule=obj_rule, sense=pyo.minimize)


    # -------------------------------------------------------------------------
    # 3.5  CONSTRAINTS
    # -------------------------------------------------------------------------
    # -- Auxiliary definitions ------------------------------------------------
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

    # -- TES constraints ------------------------------------------------------
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

    # -- Boiler constraints ---------------------------------------------------
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

    # -- CHP constraints ------------------------------------------------------
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

    # -- Demand satisfaction --------------------------------------------------
    _op = (lambda lhs, rhs: lhs == rhs) if strict_demand_satisfaction else (lambda lhs, rhs: lhs >= rhs)

    # (I) Heat demand balance:
    def heat_balance(m, k):
        tes_net = m.Q_out_TES[k] - m.Q_in_TES[k]
        return _op(
            sum(m.Q_out_CHP[i, k] for i in m.C) + sum(m.Q_out_B[i, k] for i in m.B) + tes_net,
            m.Q_D[k],
        )
    m.c_heat = pyo.Constraint(m.K, rule=heat_balance)

    # (II) Electricity demand balance
    def elec_balance(m, k):
        return _op(sum(m.P_out_CHP[i, k] for i in m.C) + m.P_grid[k], m.P_D[k])
    m.c_elec = pyo.Constraint(m.K, rule=elec_balance)

    return m


def solve_model(model: pyo.ConcreteModel, solver_name: str = "gurobi", tee: bool = True, **solver_options):
        """
        Solves the given optimization model using the specified solver.
        :param model: Pyomo ConcreteModel to be solved
        :param solver_name: Name of the solver to use (default: "gurobi")
        :param tee: Stream solver output to console (default: True)
        :param solver_options: Options for the solver
        :return: Optimization results of the model
        """
        try:
            solver = pyo.SolverFactory(solver_name)
            solver.options.update(solver_options)

            results = solver.solve(model, tee=tee)

            return results
        except Exception as e:
            print(f"Error solving model: {e}")
            raise e


def extract_solver_metadata(results: pyo.SolverFactory, model: pyo.ConcreteModel) -> dict:
    """
    Extracts the solver metadata from the results of the model
    :param results: Pyomo SolverResults object
    :param model: Pyomo ConcreteModel to be solved
    :return: Dictionary with the solver metadata
    """
    metadata = {
        "timestamp": datetime.datetime.now().isoformat(),
        "objective_value": pyo.value(model.OBJ),
        "gas_price": pyo.value(model.c_g),
        "electricity_price": pyo.value(model.c_el),
        "solver_status": str(results.solver.status),
        "solver_termination_condition": str(results.solver.termination_condition),
        "mip_gap": abs(pyo.value(model.OBJ) - results.problem.lower_bound) / abs(results.problem.lower_bound) * 100,
        "solve_time": float(results.solver.wall_time),
        "num_variables": results.problem.number_of_variables,
        "num_constraints": results.problem.number_of_constraints,
        "num_binary_variables": results.problem.number_of_binary_variables,
        "num_nonzeros": results.problem.number_of_nonzeros,
    }

    return metadata


def extract_solver_solution(m: pyo.ConcreteModel) -> pd.DataFrame:
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
            "delta_B1": pyo.value(m.delta_B[1, k]),
            "delta_B2": pyo.value(m.delta_B[2, k]),

            # CHPs
            "Q_in_CHP1": pyo.value(m.Q_in_CHP[1, k]),
            "Q_in_CHP2": pyo.value(m.Q_in_CHP[2, k]),
            "Q_out_CHP1": pyo.value(m.Q_out_CHP[1, k]),
            "Q_out_CHP2": pyo.value(m.Q_out_CHP[2, k]),
            "P_out_CHP1": pyo.value(m.P_out_CHP[1, k]),
            "P_out_CHP2": pyo.value(m.P_out_CHP[2, k]),
            "delta_CHP1": pyo.value(m.delta_CHP[1, k]),
            "delta_CHP2": pyo.value(m.delta_CHP[2, k]),

            # Grid
            "P_grid": pyo.value(m.P_grid[k]),
        }

        solution.append(sol)

    solution_df = pd.DataFrame(solution)

    return solution_df


def save_optimization_results(
    m: pyo.ConcreteModel,
    results: pyo.SolverFactory,
    optimization_type: str,
    sampling_method: str,
    sample_id: str,
) -> tuple[pd.DataFrame, dict]:
    """
    Extracts the solution and metadata from the optimization results and saves them to files.
    :param m: Pyomo ConcreteModel
    :param results: Pyomo SolverResults object
    :param optimization_type: Type of optimization ("MILP", "LP_Relaxed", ...)
    :param sampling_method: String with the name of the sampling method (either "LHS" or "Sobol")
    :param sample_id: String with the sample id as file name [(gas_price)_(elec_price)_(sample_index)]
    :return: Tuple of (solution DataFrame, metadata dictionary)
    """
    metadata = extract_solver_metadata(results, m)
    solution_df = extract_solver_solution(m)

    base_dir = RESULTS_DIR / optimization_type / sampling_method
    base_dir.mkdir(parents=True, exist_ok=True)

    file_stem = f"{optimization_type}_result_{sample_id}"

    try:
        # Save time series of optimized decision variables as parquet
        parquet_path = base_dir / f"{file_stem}.parquet"
        solution_df.to_parquet(parquet_path)

        # Save metadata as JSON
        json_path = base_dir / f"{file_stem}.json"
        with open(json_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Optimization results saved to {parquet_path}")
        print(f"Optimization metadata saved to {json_path}")

    except Exception as e:
        print(f"Failed to save optimization results!")
        raise e

    return solution_df, metadata


def load_optimization_results(
    optimization_type: str,
    sampling_method: str,
    sample_id: str | int | None = None,
    file_stem: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Loads the optimization results saved as a pandas DataFrame and the metadata as a dictionary.
    :param optimization_type: Type of optimization ("MILP", "LP_Relaxed", ...)
    :param sampling_method: String with the name of the sampling method (either "LHS" or "Sobol")
    :param sample_id: String or integer with the sample id [1, n_samples]
    :param file_stem: String with the name of the optimization results file (without extension)
    :return: Tuple of (solution DataFrame, metadata dictionary)
    """
    # Results directory
    base_dir = RESULTS_DIR / optimization_type / sampling_method

    if not base_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {base_dir}")

    # Exactly one selector should be provided
    if (sample_id is None and file_stem is None) or (sample_id is not None and file_stem is not None):
        raise ValueError("Provide exactly one of `sample_id` or `file_stem`.")

    # Resolve file_stem
    if file_stem is None:
        sid = f"{int(sample_id):03d}"  # normalizes 5 -> "005"
        candidates = sorted(base_dir.glob(f"{optimization_type}_result_*_{sid}.parquet"))

        if len(candidates) == 0:
            raise FileNotFoundError(
                f"No result found for sample_id={sid} in {base_dir} "
                f"(pattern: {optimization_type}_result_*_{sid}.parquet)"
            )
        if len(candidates) > 1:
            names = ", ".join(p.stem for p in candidates)
            raise ValueError(
                f"sample_id={sid} is ambiguous. Multiple matches found: {names}. "
                "Use `file_stem` for exact loading."
            )

        file_stem = candidates[0].stem

    # Define file paths
    parquet_path = base_dir / f"{file_stem}.parquet"
    json_path = base_dir / f"{file_stem}.json"

    # Validate that files exist
    if not parquet_path.exists():
        raise FileNotFoundError(f"Solution file not found: {parquet_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {json_path}")

    try:
        # Load solution time series
        solution_df = pd.read_parquet(parquet_path)
        print(f"Optimization results loaded from:\n{parquet_path}")

        # Load metadata
        with open(json_path, "r") as f:
            metadata_dict = json.load(f)
        print(f"Metadata loaded from:\n{json_path}")


    except pd.errors.ParquetError as e:
        raise ValueError(f"Failed to parse parquet file {parquet_path}: {e}")

    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON file {json_path}: {e}")

    except Exception as e:
        print(f"Unexpected error loading optimization results: {e}")
        raise

    return solution_df, metadata_dict


# == LP approximated linearization slopes =====================================
# Fitted by MSE minimization over the operating region
m_B_heat_fitted   = 0.8078
m_CHP_heat_fitted = 0.3948
m_CHP_el_fitted   = 0.2980

# Mean of the efficiency at minimum and maximum load within the lambda region
_lp_Qin_CHP_min = lambda_in_min_CHP * P_out_nom_CHP / eta_nom_CHP_el
m_B_heat_mean   = ((lambda_out_min_B * eta_nom_B / lambda_in_min_B) + eta_nom_B) / 2
m_CHP_heat_mean = ((lambda_out_min_CHP_th * Q_out_nom_CHP / _lp_Qin_CHP_min) + eta_nom_CHP_th) / 2
m_CHP_el_mean   = ((lambda_out_min_CHP_el * P_out_nom_CHP / _lp_Qin_CHP_min) + eta_nom_CHP_el) / 2


def build_lp_lower(
        Q_D,
        P_D,
        c_g: float,
        c_el: float,
        *,
        strict_demand_satisfaction: bool = True,
) -> pyo.ConcreteModel:
    """
    Builds the LP binary-relaxation of the MILP (lower bound on MILP optimum).
    All four binary commitment variables are relaxed to continuous [0, 1].
    :param Q_D: Heat demand array [kW], length n
    :param P_D: Electrical demand array [kW], length n
    :param c_g: Gas price [€/kWh]
    :param c_el: Electricity price [€/kWh]
    :param strict_demand_satisfaction: True → equality balances; False → >= inequalities
    :return: Unsolved pyo.ConcreteModel
    """
    Q_D_arr = np.asarray(Q_D, dtype=float)
    P_D_arr = np.asarray(P_D, dtype=float)
    n = len(Q_D_arr)

    m = pyo.ConcreteModel(name="LP_lower")
    m.K   = pyo.RangeSet(0, n - 1)
    m.B   = pyo.Set(initialize=[1, 2])
    m.CHP = pyo.Set(initialize=[1, 2])

    m.c_g  = pyo.Param(initialize=c_g)
    m.c_el = pyo.Param(initialize=c_el)
    m.Q_D  = pyo.Param(m.K, initialize={k: Q_D_arr[k] for k in range(n)})
    m.P_D  = pyo.Param(m.K, initialize={k: P_D_arr[k] for k in range(n)})

    # Relaxed commitment variables [0, 1]
    m.dB       = pyo.Var(m.B,   m.K, domain=pyo.NonNegativeReals, bounds=(0.0, 1.0))
    m.dCHP     = pyo.Var(m.CHP, m.K, domain=pyo.NonNegativeReals, bounds=(0.0, 1.0))
    m.din_TES  = pyo.Var(m.K,   domain=pyo.NonNegativeReals, bounds=(0.0, 1.0))
    m.dout_TES = pyo.Var(m.K,   domain=pyo.NonNegativeReals, bounds=(0.0, 1.0))

    # Continuous decision variables
    m.Qin_B    = pyo.Var(m.B,   m.K, domain=pyo.NonNegativeReals)
    m.Qin_CHP  = pyo.Var(m.CHP, m.K, domain=pyo.NonNegativeReals)
    m.Qin_TES  = pyo.Var(m.K,   domain=pyo.NonNegativeReals)
    m.Qout_TES = pyo.Var(m.K,   domain=pyo.NonNegativeReals)
    m.Pgrid    = pyo.Var(m.K,   domain=pyo.NonNegativeReals)

    # Auxiliary variables
    m.E_TES    = pyo.Var(m.K,   domain=pyo.NonNegativeReals, bounds=(E_min_TES, E_nom_TES))
    m.Qout_B   = pyo.Var(m.B,   m.K, domain=pyo.Reals)
    m.Qout_CHP = pyo.Var(m.CHP, m.K, domain=pyo.Reals)
    m.Pout_CHP = pyo.Var(m.CHP, m.K, domain=pyo.Reals)

    def obj_rule(m):
        return (
            m.c_g  * DELTA_T * sum(sum(m.Qin_B[i, k] for i in m.B) + sum(m.Qin_CHP[i, k] for i in m.CHP) for k in m.K)
            + m.c_el * DELTA_T * sum(m.Pgrid[k] for k in m.K)
        )
    m.OBJ = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # TES
    def tes_dyn(m, k):
        return m.E_TES[(k + 1) % n] == a_TES * m.E_TES[k] + b1_TES * m.Qin_TES[k] + b2_TES * m.Qout_TES[k]
    m.c_TES_dyn = pyo.Constraint(m.K, rule=tes_dyn)

    def tes_in_ub(m, k):
        return m.Qin_TES[k] <= m.din_TES[k] * E_nom_TES / tau_in
    m.c_TES_in_ub = pyo.Constraint(m.K, rule=tes_in_ub)

    def tes_in_lb(m, k):
        return m.Qin_TES[k] >= m.din_TES[k] * Q_in_min_TES
    m.c_TES_in_lb = pyo.Constraint(m.K, rule=tes_in_lb)

    def tes_out_ub(m, k):
        return m.Qout_TES[k] <= m.dout_TES[k] * E_nom_TES / tau_out
    m.c_TES_out_ub = pyo.Constraint(m.K, rule=tes_out_ub)

    def tes_out_lb(m, k):
        return m.Qout_TES[k] >= m.dout_TES[k] * Q_out_min_TES
    m.c_TES_out_lb = pyo.Constraint(m.K, rule=tes_out_lb)

    def tes_mutex(m, k):
        return m.din_TES[k] + m.dout_TES[k] <= 1
    m.c_TES_mutex = pyo.Constraint(m.K, rule=tes_mutex)

    # Boilers
    def boiler_pl(m, i, k):
        return m.Qout_B[i, k] == Q_out_nom_B * (
            m.dB[i, k] * lambda_out_min_B
            + (1.0 / beta_B) * (m.Qin_B[i, k] * eta_nom_B / Q_out_nom_B - m.dB[i, k] * lambda_in_min_B)
        )
    m.c_B_pl = pyo.Constraint(m.B, m.K, rule=boiler_pl)

    def boiler_in_ub(m, i, k):
        return m.Qin_B[i, k] <= m.dB[i, k] * Q_out_nom_B / eta_nom_B
    m.c_B_in_ub = pyo.Constraint(m.B, m.K, rule=boiler_in_ub)

    def boiler_in_lb(m, i, k):
        return m.Qin_B[i, k] >= m.dB[i, k] * lambda_in_min_B * Q_out_nom_B / eta_nom_B
    m.c_B_in_lb = pyo.Constraint(m.B, m.K, rule=boiler_in_lb)

    # CHPs
    def chp_th_pl(m, i, k):
        return m.Qout_CHP[i, k] == Q_out_nom_CHP * (
            m.dCHP[i, k] * lambda_out_min_CHP_th
            + (1.0 / beta_CHP_th) * (m.Qin_CHP[i, k] * eta_nom_CHP_th / Q_out_nom_CHP - m.dCHP[i, k] * lambda_in_min_CHP)
        )
    m.c_CHP_th = pyo.Constraint(m.CHP, m.K, rule=chp_th_pl)

    def chp_el_pl(m, i, k):
        return m.Pout_CHP[i, k] == P_out_nom_CHP * (
            m.dCHP[i, k] * lambda_out_min_CHP_el
            + (1.0 / beta_CHP_el) * (m.Qin_CHP[i, k] * eta_nom_CHP_el / P_out_nom_CHP - m.dCHP[i, k] * lambda_in_min_CHP)
        )
    m.c_CHP_el = pyo.Constraint(m.CHP, m.K, rule=chp_el_pl)

    def chp_in_ub(m, i, k):
        return m.Qin_CHP[i, k] <= m.dCHP[i, k] * Q_out_nom_CHP / eta_nom_CHP_th
    m.c_CHP_in_ub = pyo.Constraint(m.CHP, m.K, rule=chp_in_ub)

    def chp_in_lb(m, i, k):
        return m.Qin_CHP[i, k] >= m.dCHP[i, k] * lambda_in_min_CHP * P_out_nom_CHP / eta_nom_CHP_el
    m.c_CHP_in_lb = pyo.Constraint(m.CHP, m.K, rule=chp_in_lb)

    _op = (lambda lhs, rhs: lhs == rhs) if strict_demand_satisfaction else (lambda lhs, rhs: lhs >= rhs)

    def heat_balance(m, k):
        return _op(
            sum(m.Qout_CHP[i, k] for i in m.CHP) + sum(m.Qout_B[i, k] for i in m.B)
            + m.Qout_TES[k] - m.Qin_TES[k],
            m.Q_D[k],
        )
    m.c_heat = pyo.Constraint(m.K, rule=heat_balance)

    def elec_balance(m, k):
        return _op(sum(m.Pout_CHP[i, k] for i in m.CHP) + m.Pgrid[k], m.P_D[k])
    m.c_elec = pyo.Constraint(m.K, rule=elec_balance)

    return m


def build_lp_approximated(
        Q_D,
        P_D,
        c_g: float,
        c_el: float,
        *,
        mode: str = "mean_efficiency",
        strict_demand_satisfaction: bool = True,
) -> pyo.ConcreteModel:
    """
    Builds the LP approximation of the MILP: binary commitment variables are dropped
    and part-load curves are replaced by proportional (through-the-origin) slopes.
    :param Q_D: Heat demand array [kW], length n
    :param P_D: Electrical demand array [kW], length n
    :param c_g: Gas price [€/kWh]
    :param c_el: Electricity price [€/kWh]
    :param mode: 'fitted' (MSE-fitted slopes) or 'mean_efficiency' (analytic mean)
    :param strict_demand_satisfaction: True → equality balances; False → >= inequalities
    :return: Unsolved pyo.ConcreteModel
    """
    if mode not in ("fitted", "mean_efficiency"):
        raise ValueError("mode must be 'fitted' or 'mean_efficiency'")

    if mode == "mean_efficiency":
        slope_B = m_B_heat_mean
        slope_Q = m_CHP_heat_mean
        slope_P = m_CHP_el_mean
    else:
        slope_B = m_B_heat_fitted
        slope_Q = m_CHP_heat_fitted
        slope_P = m_CHP_el_fitted

    Q_D_arr = np.asarray(Q_D, dtype=float)
    P_D_arr = np.asarray(P_D, dtype=float)
    n = len(Q_D_arr)

    m = pyo.ConcreteModel(name="LP_approximated")
    m.K   = pyo.RangeSet(0, n - 1)
    m.B   = pyo.Set(initialize=[1, 2])
    m.CHP = pyo.Set(initialize=[1, 2])

    m.c_g  = pyo.Param(initialize=c_g)
    m.c_el = pyo.Param(initialize=c_el)
    m.Q_D  = pyo.Param(m.K, initialize={k: Q_D_arr[k] for k in range(n)})
    m.P_D  = pyo.Param(m.K, initialize={k: P_D_arr[k] for k in range(n)})

    # Continuous decision variables (no commitment variables)
    m.Qin_B    = pyo.Var(m.B,   m.K, domain=pyo.NonNegativeReals)
    m.Qin_CHP  = pyo.Var(m.CHP, m.K, domain=pyo.NonNegativeReals)
    m.Qin_TES  = pyo.Var(m.K,   domain=pyo.NonNegativeReals)
    m.Qout_TES = pyo.Var(m.K,   domain=pyo.NonNegativeReals)
    m.Pgrid    = pyo.Var(m.K,   domain=pyo.NonNegativeReals)

    # Auxiliary variables
    m.E_TES    = pyo.Var(m.K,   domain=pyo.NonNegativeReals, bounds=(E_min_TES, E_nom_TES))
    m.Qout_B   = pyo.Var(m.B,   m.K, domain=pyo.NonNegativeReals)
    m.Qout_CHP = pyo.Var(m.CHP, m.K, domain=pyo.NonNegativeReals)
    m.Pout_CHP = pyo.Var(m.CHP, m.K, domain=pyo.NonNegativeReals)

    def obj_rule(m):
        return (
            m.c_g  * DELTA_T * sum(sum(m.Qin_B[i, k] for i in m.B) + sum(m.Qin_CHP[i, k] for i in m.CHP) for k in m.K)
            + m.c_el * DELTA_T * sum(m.Pgrid[k] for k in m.K)
        )
    m.OBJ = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # TES
    def tes_dyn(m, k):
        return m.E_TES[(k + 1) % n] == a_TES * m.E_TES[k] + b1_TES * m.Qin_TES[k] + b2_TES * m.Qout_TES[k]
    m.c_TES_dyn = pyo.Constraint(m.K, rule=tes_dyn)

    def tes_in_ub(m, k):
        return m.Qin_TES[k] <= E_nom_TES / tau_in
    m.c_TES_in_ub = pyo.Constraint(m.K, rule=tes_in_ub)

    def tes_out_ub(m, k):
        return m.Qout_TES[k] <= E_nom_TES / tau_out
    m.c_TES_out_ub = pyo.Constraint(m.K, rule=tes_out_ub)

    # Boilers: linearized output
    def boiler_pl(m, i, k):
        return m.Qout_B[i, k] == slope_B * m.Qin_B[i, k]
    m.c_B_pl = pyo.Constraint(m.B, m.K, rule=boiler_pl)

    def boiler_in_ub(m, i, k):
        return m.Qin_B[i, k] <= Q_out_nom_B / eta_nom_B
    m.c_B_in_ub = pyo.Constraint(m.B, m.K, rule=boiler_in_ub)

    # CHPs: linearized thermal and electrical outputs
    def chp_heat(m, i, k):
        return m.Qout_CHP[i, k] == slope_Q * m.Qin_CHP[i, k]
    m.c_CHP_th = pyo.Constraint(m.CHP, m.K, rule=chp_heat)

    def chp_power(m, i, k):
        return m.Pout_CHP[i, k] == slope_P * m.Qin_CHP[i, k]
    m.c_CHP_el = pyo.Constraint(m.CHP, m.K, rule=chp_power)

    def chp_in_ub(m, i, k):
        return m.Qin_CHP[i, k] <= Q_out_nom_CHP / eta_nom_CHP_th
    m.c_CHP_in_ub = pyo.Constraint(m.CHP, m.K, rule=chp_in_ub)

    _op = (lambda lhs, rhs: lhs == rhs) if strict_demand_satisfaction else (lambda lhs, rhs: lhs >= rhs)

    def heat_balance(m, k):
        return _op(
            sum(m.Qout_CHP[i, k] for i in m.CHP) + sum(m.Qout_B[i, k] for i in m.B)
            + m.Qout_TES[k] - m.Qin_TES[k],
            m.Q_D[k],
        )
    m.c_heat = pyo.Constraint(m.K, rule=heat_balance)

    def elec_balance(m, k):
        return _op(sum(m.Pout_CHP[i, k] for i in m.CHP) + m.Pgrid[k], m.P_D[k])
    m.c_elec = pyo.Constraint(m.K, rule=elec_balance)

    return m


def _build_lp_upper_fixed(
        Q_D,
        P_D,
        c_g: float,
        c_el: float,
        dB_fixed: dict,
        dCHP_fixed: dict,
        din_fixed: dict,
        dout_fixed: dict,
        *,
        strict_demand_satisfaction: bool = True,
) -> pyo.ConcreteModel:
    """
    Builds the LP upper-bound model with pre-fixed unit-commitment binary values.
    :param dB_fixed:   {(i, k): float}  boiler commitment schedule
    :param dCHP_fixed: {(i, k): float}  CHP commitment schedule
    :param din_fixed:  {k: float}       TES charge permission schedule
    :param dout_fixed: {k: float}       TES discharge permission schedule
    """
    Q_D_arr = np.asarray(Q_D, dtype=float)
    P_D_arr = np.asarray(P_D, dtype=float)
    n = len(Q_D_arr)

    m = pyo.ConcreteModel(name="LP_upper")
    m.K   = pyo.RangeSet(0, n - 1)
    m.B   = pyo.Set(initialize=[1, 2])
    m.CHP = pyo.Set(initialize=[1, 2])

    m.c_g  = pyo.Param(initialize=c_g)
    m.c_el = pyo.Param(initialize=c_el)
    m.Q_D  = pyo.Param(m.K, initialize={k: Q_D_arr[k] for k in range(n)})
    m.P_D  = pyo.Param(m.K, initialize={k: P_D_arr[k] for k in range(n)})

    # Continuous decision variables
    m.Qin_B    = pyo.Var(m.B,   m.K, domain=pyo.NonNegativeReals)
    m.Qin_CHP  = pyo.Var(m.CHP, m.K, domain=pyo.NonNegativeReals)
    m.Qin_TES  = pyo.Var(m.K,   domain=pyo.NonNegativeReals)
    m.Qout_TES = pyo.Var(m.K,   domain=pyo.NonNegativeReals)
    m.Pgrid    = pyo.Var(m.K,   domain=pyo.NonNegativeReals)

    # Auxiliary variables
    m.E_TES    = pyo.Var(m.K,   domain=pyo.NonNegativeReals, bounds=(E_min_TES, E_nom_TES))
    m.Qout_B   = pyo.Var(m.B,   m.K, domain=pyo.Reals)
    m.Qout_CHP = pyo.Var(m.CHP, m.K, domain=pyo.Reals)
    m.Pout_CHP = pyo.Var(m.CHP, m.K, domain=pyo.Reals)

    def obj_rule(m):
        return (
            m.c_g  * DELTA_T * sum(sum(m.Qin_B[i, k] for i in m.B) + sum(m.Qin_CHP[i, k] for i in m.CHP) for k in m.K)
            + m.c_el * DELTA_T * sum(m.Pgrid[k] for k in m.K)
        )
    m.OBJ = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # TES (fixed commitment upper bounds)
    def tes_dyn(m, k):
        return m.E_TES[(k + 1) % n] == a_TES * m.E_TES[k] + b1_TES * m.Qin_TES[k] + b2_TES * m.Qout_TES[k]
    m.c_TES_dyn = pyo.Constraint(m.K, rule=tes_dyn)

    def tes_in_ub(m, k):
        return m.Qin_TES[k] <= din_fixed[k] * E_nom_TES / tau_in
    m.c_TES_in_ub = pyo.Constraint(m.K, rule=tes_in_ub)

    def tes_out_ub(m, k):
        return m.Qout_TES[k] <= dout_fixed[k] * E_nom_TES / tau_out
    m.c_TES_out_ub = pyo.Constraint(m.K, rule=tes_out_ub)

    # Boilers (fixed commitment)
    def boiler_pl(m, i, k):
        d = dB_fixed[i, k]
        return m.Qout_B[i, k] == Q_out_nom_B * (
            d * lambda_out_min_B
            + (1.0 / beta_B) * (m.Qin_B[i, k] * eta_nom_B / Q_out_nom_B - d * lambda_in_min_B)
        )
    m.c_B_pl = pyo.Constraint(m.B, m.K, rule=boiler_pl)

    def boiler_in_ub(m, i, k):
        return m.Qin_B[i, k] <= dB_fixed[i, k] * Q_out_nom_B / eta_nom_B
    m.c_B_in_ub = pyo.Constraint(m.B, m.K, rule=boiler_in_ub)

    def boiler_in_lb(m, i, k):
        return m.Qin_B[i, k] >= dB_fixed[i, k] * lambda_in_min_B * Q_out_nom_B / eta_nom_B
    m.c_B_in_lb = pyo.Constraint(m.B, m.K, rule=boiler_in_lb)

    # CHPs (fixed commitment)
    def chp_th_pl(m, i, k):
        d = dCHP_fixed[i, k]
        return m.Qout_CHP[i, k] == Q_out_nom_CHP * (
            d * lambda_out_min_CHP_th
            + (1.0 / beta_CHP_th) * (m.Qin_CHP[i, k] * eta_nom_CHP_th / Q_out_nom_CHP - d * lambda_in_min_CHP)
        )
    m.c_CHP_th = pyo.Constraint(m.CHP, m.K, rule=chp_th_pl)

    def chp_el_pl(m, i, k):
        d = dCHP_fixed[i, k]
        return m.Pout_CHP[i, k] == P_out_nom_CHP * (
            d * lambda_out_min_CHP_el
            + (1.0 / beta_CHP_el) * (m.Qin_CHP[i, k] * eta_nom_CHP_el / P_out_nom_CHP - d * lambda_in_min_CHP)
        )
    m.c_CHP_el = pyo.Constraint(m.CHP, m.K, rule=chp_el_pl)

    def chp_in_ub(m, i, k):
        return m.Qin_CHP[i, k] <= dCHP_fixed[i, k] * Q_out_nom_CHP / eta_nom_CHP_th
    m.c_CHP_in_ub = pyo.Constraint(m.CHP, m.K, rule=chp_in_ub)

    def chp_in_lb(m, i, k):
        return m.Qin_CHP[i, k] >= dCHP_fixed[i, k] * lambda_in_min_CHP * P_out_nom_CHP / eta_nom_CHP_el
    m.c_CHP_in_lb = pyo.Constraint(m.CHP, m.K, rule=chp_in_lb)

    _op = (lambda lhs, rhs: lhs == rhs) if strict_demand_satisfaction else (lambda lhs, rhs: lhs >= rhs)

    def heat_balance(m, k):
        return _op(
            sum(m.Qout_CHP[i, k] for i in m.CHP) + sum(m.Qout_B[i, k] for i in m.B)
            + m.Qout_TES[k] - m.Qin_TES[k],
            m.Q_D[k],
        )
    m.c_heat = pyo.Constraint(m.K, rule=heat_balance)

    def elec_balance(m, k):
        return _op(sum(m.Pout_CHP[i, k] for i in m.CHP) + m.Pgrid[k], m.P_D[k])
    m.c_elec = pyo.Constraint(m.K, rule=elec_balance)

    return m


# == Solve wrapper helpers =====================================================

def _extract_lp_solution(
        m: pyo.ConcreteModel,
        dB_sched=None,
        dCHP_sched=None,
        din_sched=None,
        dout_sched=None,
) -> pd.DataFrame:
    """
    Extract LP model solution into a Marius-format DataFrame.
    Pass *_sched dicts/callables for LP upper (fixed binaries) or LP lower
    (pass None to read from model variables m.dB, m.dCHP, m.din_TES, m.dout_TES).
    """
    rows = []
    has_delta_vars = hasattr(m, "dB")
    for k in m.K:
        row = {
            "k":        k,
            "Q_D":      pyo.value(m.Q_D[k]),
            "P_D":      pyo.value(m.P_D[k]),
            "E_TES":    pyo.value(m.E_TES[k]),
            "Qin_TES":  pyo.value(m.Qin_TES[k]),
            "Qout_TES": pyo.value(m.Qout_TES[k]),
            "Pgrid":    pyo.value(m.Pgrid[k]),
        }
        if dB_sched is not None:
            row["dB1"]    = dB_sched[1, k]
            row["dB2"]    = dB_sched[2, k]
            row["dCHP1"]  = dCHP_sched[1, k]
            row["dCHP2"]  = dCHP_sched[2, k]
            row["din_TES"]  = din_sched[k]
            row["dout_TES"] = dout_sched[k]
        elif has_delta_vars:
            row["dB1"]    = pyo.value(m.dB[1, k])
            row["dB2"]    = pyo.value(m.dB[2, k])
            row["dCHP1"]  = pyo.value(m.dCHP[1, k])
            row["dCHP2"]  = pyo.value(m.dCHP[2, k])
            row["din_TES"]  = pyo.value(m.din_TES[k])
            row["dout_TES"] = pyo.value(m.dout_TES[k])
        for i in [1, 2]:
            row[f"Qin_B{i}"]   = pyo.value(m.Qin_B[i, k])
            row[f"Qout_B{i}"]  = pyo.value(m.Qout_B[i, k])
            row[f"Qin_CHP{i}"] = pyo.value(m.Qin_CHP[i, k])
            row[f"Qout_CHP{i}"]= pyo.value(m.Qout_CHP[i, k])
            row[f"Pout_CHP{i}"]= pyo.value(m.Pout_CHP[i, k])
        rows.append(row)
    return pd.DataFrame(rows)


def solve_milp(
        Q_D,
        P_D,
        c_g: float,
        c_el: float,
        *,
        mip_gap: float = 1e-3,
        normalize: bool = False,
        raw_normalized: bool = False,
        strict_demand_satisfaction: bool = True,
        tee: bool = False,
) -> tuple[float, pd.DataFrame]:
    """
    Build, solve, and extract the MILP dispatch model.
    :param Q_D: Heat demand array [kW], length N
    :param P_D: Electrical demand array [kW], length N
    :param c_g: Gas price [€/kWh]
    :param c_el: Electricity price [€/kWh]
    :param mip_gap: Gurobi MIPGap tolerance
    :param normalize: Divide objective by c_el for better numerical scaling
    :param raw_normalized: (only with normalize=True) return OPEX/c_el instead of OPEX
    :param strict_demand_satisfaction: True → equality balances; False → >= inequalities
    :param tee: Stream solver output to console
    :return: (opex, dispatch_df) with Marius-format column names
    """
    Q_D_s = pd.Series(np.asarray(Q_D, dtype=float), index=range(1, N + 1))
    P_D_s = pd.Series(np.asarray(P_D, dtype=float), index=range(1, N + 1))

    m = build_milp(Q_D_s, P_D_s, c_g, c_el, normalize=normalize,
                   strict_demand_satisfaction=strict_demand_satisfaction)

    solve_model(m, MIPGap=mip_gap, TimeLimit=300, tee=tee)

    raw_obj = pyo.value(m.OBJ)
    if normalize:
        opex = raw_obj if raw_normalized else c_el * raw_obj
    else:
        opex = raw_obj

    rows = []
    for k in m.K:  # k = 1..N
        rows.append({
            "k":         k - 1,
            "Q_D":       pyo.value(m.Q_D[k]),
            "P_D":       pyo.value(m.P_D[k]),
            "E_TES":     pyo.value(m.E_TES[k]),
            "Qin_TES":   pyo.value(m.Q_in_TES[k]),
            "Qout_TES":  pyo.value(m.Q_out_TES[k]),
            "Pgrid":     pyo.value(m.P_grid[k]),
            "dB1":       pyo.value(m.delta_B[1, k]),
            "dB2":       pyo.value(m.delta_B[2, k]),
            "dCHP1":     pyo.value(m.delta_CHP[1, k]),
            "dCHP2":     pyo.value(m.delta_CHP[2, k]),
            "din_TES":   pyo.value(m.delta_in_TES[k]),
            "dout_TES":  pyo.value(m.delta_out_TES[k]),
            "Qin_B1":    pyo.value(m.Q_in_B[1, k]),
            "Qin_B2":    pyo.value(m.Q_in_B[2, k]),
            "Qout_B1":   pyo.value(m.Q_out_B[1, k]),
            "Qout_B2":   pyo.value(m.Q_out_B[2, k]),
            "Qin_CHP1":  pyo.value(m.Q_in_CHP[1, k]),
            "Qin_CHP2":  pyo.value(m.Q_in_CHP[2, k]),
            "Qout_CHP1": pyo.value(m.Q_out_CHP[1, k]),
            "Qout_CHP2": pyo.value(m.Q_out_CHP[2, k]),
            "Pout_CHP1": pyo.value(m.P_out_CHP[1, k]),
            "Pout_CHP2": pyo.value(m.P_out_CHP[2, k]),
        })
    return opex, pd.DataFrame(rows)


def solve_lp_lower(
        Q_D,
        P_D,
        c_g: float,
        c_el: float,
        *,
        strict_demand_satisfaction: bool = True,
        tee: bool = False,
) -> tuple[float, pd.DataFrame]:
    """
    Build, solve, and extract the LP lower-bound (binary-relaxation) model.
    Returns (opex, dispatch_df) with Marius-format column names.
    """
    m = build_lp_lower(Q_D, P_D, c_g, c_el,
                       strict_demand_satisfaction=strict_demand_satisfaction)
    solve_model(m, TimeLimit=300, tee=tee)
    return pyo.value(m.OBJ), _extract_lp_solution(m)


def solve_lp_upper(
        Q_D,
        P_D,
        c_g: float,
        c_el: float,
        *,
        mode: str = "min",
        return_both: bool = False,
        strict_demand_satisfaction: bool = True,
        tee: bool = False,
) -> tuple[float, pd.DataFrame]:
    """
    Build, solve, and extract the LP upper-bound model.
    Returns (opex, dispatch_df) with Marius-format column names.

    Modes
    -----
    mode='min' (default): run both 'boilers_on' and 'chp_on', return the cheaper.
    mode='boilers_on' | 'chp_on': run that single heuristic.
    mode='rounded': solve LP lower, round deltas, fix and re-solve as LP.
    return_both=True: return ((opex_bo, df_bo), (opex_chp, df_chp)) regardless of mode.
    Returns (nan, empty DataFrame) when the solver cannot find a feasible solution.

    chp_on heuristic — decision rules (evaluated per timestep, first match wins)
    ----------------------------------------------------------------------------
    Notation:
        CHP_max_Q = Q_out_nom_CHP                       max CHP thermal output per unit [kW]
        CHP_min_Q = Q_out_nom_CHP · λ_out_min_CHP_th   min CHP thermal output per unit [kW]
        CHP_min_P = P_out_nom_CHP · λ_out_min_CHP_el   min CHP electrical output per unit [kW]
        B_max_Q   = Q_out_nom_B                         max boiler thermal output [kW]
        B_min_Q   = Q_out_nom_B · λ_out_min_B           min boiler thermal output [kW]

    I.  CHP_min_Q > Q_D  OR  CHP_min_P > P_D
        Demand is below the minimum feasible CHP output (thermal or electrical).
        → CHP1=0, CHP2=0, B1=1, B2=(1 if 2·B_min_Q ≤ Q_D else 0).

    II. 2·CHP_min_Q ≤ Q_D  AND  2·CHP_min_P ≤ P_D
        Demand is large enough that both CHPs can run at minimum load.
        → CHP1=1, CHP2=1, B1=0, B2=0.

    III. CHP_max_Q ≥ Q_D  AND  CHP_min_P ≤ P_D
        One CHP alone can cover all heat demand.
        → CHP1=1, CHP2=0, B1=0, B2=0.

    IV. CHP_max_Q + B_max_Q ≥ Q_D  AND  CHP_min_Q + B_min_Q ≤ Q_D  AND  CHP_min_P ≤ P_D
        Demand falls within the combined feasible range of one CHP and one boiler.
        → CHP1=1, CHP2=0, B1=1, B2=0.

    V.  (all remaining cases)
        One CHP and two boilers needed.
        → CHP1=1, CHP2=0, B1=1, B2=1.

    boilers_on heuristic
    --------------------
    CHPs always off. B1 always on. B2=1 if B_max_Q < Q_D, else B2=0.
    """
    Q_D_arr = np.asarray(Q_D, dtype=float)
    P_D_arr = np.asarray(P_D, dtype=float)
    n = len(Q_D_arr)

    def _run_single(single_mode: str):
        if single_mode == "rounded":
            _, lp_sol = solve_lp_lower(Q_D_arr, P_D_arr, c_g, c_el,
                                       strict_demand_satisfaction=strict_demand_satisfaction,
                                       tee=tee)
            dB_f   = {(i, k): round(lp_sol.iloc[k]["dB1" if i == 1 else "dB2"])   for i in [1, 2] for k in range(n)}
            dCHP_f = {(i, k): round(lp_sol.iloc[k]["dCHP1" if i == 1 else "dCHP2"]) for i in [1, 2] for k in range(n)}
            din_f  = {k: round(lp_sol.iloc[k]["din_TES"])  for k in range(n)}
            dout_f = {k: round(lp_sol.iloc[k]["dout_TES"]) for k in range(n)}
        elif single_mode == "chp_on":
            CHP_max_Q = Q_out_nom_CHP
            CHP_min_Q = Q_out_nom_CHP * lambda_out_min_CHP_th
            CHP_min_P = P_out_nom_CHP * lambda_out_min_CHP_el
            B_max_Q   = Q_out_nom_B
            B_min_Q   = Q_out_nom_B * lambda_out_min_B

            if any(2 * CHP_max_Q < Q_D_arr[k] for k in range(n)):
                bad = [k for k in range(n) if 2 * CHP_max_Q < Q_D_arr[k]]
                raise ValueError(
                    f"chp_on: combined CHP max heat ({2*CHP_max_Q:.1f} kW) "
                    f"insufficient at timesteps {bad[:5]}{'...' if len(bad)>5 else ''}."
                )

            _chp1, _chp2, _b1, _b2 = {}, {}, {}, {}
            for k in range(n):
                heat_d = Q_D_arr[k]
                pow_d  = P_D_arr[k]
                if CHP_min_Q > heat_d or CHP_min_P > pow_d:
                    _chp1[k] = 0.0; _chp2[k] = 0.0
                    _b1[k] = 1.0; _b2[k] = 0.0 if 2 * B_min_Q > heat_d else 1.0
                elif 2 * CHP_min_Q <= heat_d and 2 * CHP_min_P <= pow_d:
                    _chp1[k] = 1.0; _chp2[k] = 1.0; _b1[k] = 0.0; _b2[k] = 0.0
                elif CHP_max_Q >= heat_d and CHP_min_P <= pow_d:
                    _chp1[k] = 1.0; _chp2[k] = 0.0; _b1[k] = 0.0; _b2[k] = 0.0
                elif (CHP_max_Q + B_max_Q >= heat_d
                      and CHP_min_Q + B_min_Q <= heat_d
                      and CHP_min_P <= pow_d):
                    _chp1[k] = 1.0; _chp2[k] = 0.0; _b1[k] = 1.0; _b2[k] = 0.0
                else:
                    _chp1[k] = 1.0; _chp2[k] = 0.0; _b1[k] = 1.0; _b2[k] = 1.0

            dB_f   = {(i, k): (_b1[k] if i == 1 else _b2[k])   for i in [1, 2] for k in range(n)}
            dCHP_f = {(i, k): (_chp1[k] if i == 1 else _chp2[k]) for i in [1, 2] for k in range(n)}
            din_f  = {k: 1.0 if k % 2 == 0 else 0.0 for k in range(n)}
            dout_f = {k: 0.0 if k % 2 == 0 else 1.0 for k in range(n)}
        else:  # boilers_on
            B_max_Q = Q_out_nom_B
            _b1, _b2 = {}, {}
            for k in range(n):
                _b1[k] = 1.0
                _b2[k] = 0.0 if B_max_Q >= Q_D_arr[k] else 1.0

            dB_f   = {(i, k): (_b1[k] if i == 1 else _b2[k]) for i in [1, 2] for k in range(n)}
            dCHP_f = {(i, k): 0.0                              for i in [1, 2] for k in range(n)}
            din_f  = {k: 1.0 if k % 2 == 0 else 0.0 for k in range(n)}
            dout_f = {k: 0.0 if k % 2 == 0 else 1.0 for k in range(n)}

        mdl = _build_lp_upper_fixed(
            Q_D_arr, P_D_arr, c_g, c_el,
            dB_f, dCHP_f, din_f, dout_f,
            strict_demand_satisfaction=strict_demand_satisfaction,
        )
        results = solve_model(mdl, TimeLimit=300, tee=tee)
        tc = str(results.solver.termination_condition)
        if tc not in ("optimal", "feasible"):
            return float("nan"), pd.DataFrame()
        dispatch = _extract_lp_solution(mdl, dB_f, dCHP_f, din_f, dout_f)
        return pyo.value(mdl.OBJ), dispatch

    if mode == "min" or return_both:
        bo  = _run_single("boilers_on")
        chp = _run_single("chp_on")
        if return_both:
            return bo, chp
        bo_opex, chp_opex = bo[0], chp[0]
        if np.isnan(bo_opex) and np.isnan(chp_opex):
            return float("nan"), pd.DataFrame()
        if np.isnan(bo_opex):
            return chp
        if np.isnan(chp_opex):
            return bo
        return bo if bo_opex <= chp_opex else chp

    return _run_single(mode)


def solve_lp_approximated(
        Q_D,
        P_D,
        c_g: float,
        c_el: float,
        *,
        mode: str = "mean_efficiency",
        strict_demand_satisfaction: bool = True,
        tee: bool = False,
) -> tuple[float, pd.DataFrame]:
    """
    Build, solve, and extract the LP approximation model.
    Returns (opex, dispatch_df) with Marius-format column names.
    mode: 'fitted' or 'mean_efficiency'
    """
    m = build_lp_approximated(Q_D, P_D, c_g, c_el, mode=mode,
                               strict_demand_satisfaction=strict_demand_satisfaction)
    solve_model(m, TimeLimit=300, tee=tee)
    return pyo.value(m.OBJ), _extract_lp_solution(m)

