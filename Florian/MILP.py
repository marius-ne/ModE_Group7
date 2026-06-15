import pyomo.environ as pyo
import pandas as pd
import numpy as np

def run_mode_project_milp_exact(c_gas=0.16, c_el=0.21, demand_csv_path=None):
    """
    Erstellt und löst das MILP-Modell basierend auf der exakten Diskretisierung 
    (Zustandsraummodell) des Speichers. Demand ist wieder streng durch == gekoppelt.
    """
    m = pyo.ConcreteModel(name="ModE_Project_5_Exact")
    
    # --- 1. Sets ---
    N_steps = 168
    m.N = pyo.Param(initialize=N_steps)
    m.K = pyo.RangeSet(0, m.N - 1)
    m.K0 = pyo.RangeSet(0, m.N)
    m.B_set = pyo.Set(initialize=[1, 2])
    m.CHP_set = pyo.Set(initialize=[1, 2])
    
    # --- 2. Preis-Zuweisung ---
    c_gas_dict = {k: float(c_gas) for k in range(0, N_steps)}
    c_el_dict = {k: float(c_el) for k in range(0, N_steps)}

    m.c_G = pyo.Param(m.K, initialize=c_gas_dict)
    m.c_el = pyo.Param(m.K, initialize=c_el_dict)

    # --- 3. Demand Daten Import ---
    if demand_csv_path:
        try:
            df_demand = pd.read_csv(demand_csv_path)
            dict_P_D = {k: float(df_demand.iloc[k, 1]) for k in range(0, N_steps)}
            dict_Q_D = {k: float(df_demand.iloc[k, 2]) for k in range(0, N_steps)}
            print(f"Daten aus {demand_csv_path} geladen.")
        except Exception as e:
            print(f"Fehler beim Laden der CSV: {e}. Verwende Dummy-Daten.")
            demand_csv_path = None

    if not demand_csv_path:
        np.random.seed(42)
        dict_P_D = {k: float(np.random.uniform(100, 500)) for k in range(0, N_steps)}
        dict_Q_D = {k: float(np.random.uniform(300, 900)) for k in range(0, N_steps)}

    m.Q_D = pyo.Param(m.K, initialize=dict_Q_D)
    m.P_D = pyo.Param(m.K, initialize=dict_P_D)

    # --- 4. Technische Parameter ---
    m.dt = pyo.Param(initialize=1.0)
    
    # TES - Exakte Diskretisierung Parameter
    m.a_tes = pyo.Param(initialize=0.995012)
    m.b1_tes = pyo.Param(initialize=0.94763)
    m.b2_tes = pyo.Param(initialize=-1.05)
    
    m.tau_in = pyo.Param(initialize=1.0)
    m.tau_out = pyo.Param(initialize=1.0)
    m.E_TES_nom = pyo.Param(initialize=1000.0)
    m.E_TES_min = pyo.Param(initialize=0.0)
    m.Q_TES_in_min = pyo.Param(initialize=0.0)
    m.Q_TES_out_min = pyo.Param(initialize=0.0)

    # Boiler
    m.Q_B_out_nom = pyo.Param(initialize=530.0)
    m.eta_B_nom = pyo.Param(initialize=0.8)
    m.lam_B_in_min = pyo.Param(initialize=0.173)
    m.lam_B_out_min = pyo.Param(initialize=0.2)
    m.beta_B = pyo.Param(initialize=(1 - m.lam_B_in_min) / (1 - m.lam_B_out_min))

    # CHP
    m.Q_CHP_out_nom = pyo.Param(initialize=470.0)
    m.P_CHP_out_nom = pyo.Param(initialize=380.0)
    m.eta_CHP_th_nom = pyo.Param(initialize=0.481)
    m.eta_CHP_el_nom = pyo.Param(initialize=0.389)
    m.lam_CHP_in_min = pyo.Param(initialize=0.582)
    m.lam_CHP_th_out_min = pyo.Param(initialize=0.622)
    m.lam_CHP_el_out_min = pyo.Param(initialize=0.5)
    m.beta_CHP_th = pyo.Param(initialize=(1 - m.lam_CHP_in_min) / (1 - m.lam_CHP_th_out_min))
    m.beta_CHP_el = pyo.Param(initialize=(1 - m.lam_CHP_in_min) / (1 - m.lam_CHP_el_out_min))

    # --- 5. Variablen ---
    m.delta_TES_in = pyo.Var(m.K, domain=pyo.Binary)
    m.delta_TES_out = pyo.Var(m.K, domain=pyo.Binary)
    m.delta_B = pyo.Var(m.B_set, m.K, domain=pyo.Binary)
    m.delta_CHP = pyo.Var(m.CHP_set, m.K, domain=pyo.Binary)

    m.Q_B_in = pyo.Var(m.B_set, m.K, domain=pyo.NonNegativeReals)
    m.Q_B_out = pyo.Var(m.B_set, m.K, domain=pyo.NonNegativeReals)
    
    m.Q_CHP_in = pyo.Var(m.CHP_set, m.K, domain=pyo.NonNegativeReals)
    m.Q_CHP_out = pyo.Var(m.CHP_set, m.K, domain=pyo.NonNegativeReals)
    m.P_CHP_out = pyo.Var(m.CHP_set, m.K, domain=pyo.NonNegativeReals)
    
    m.Q_TES_in = pyo.Var(m.K, domain=pyo.NonNegativeReals)
    m.Q_TES_out = pyo.Var(m.K, domain=pyo.NonNegativeReals)
    m.E_TES = pyo.Var(m.K0, bounds=(0, m.E_TES_nom)) 
    
    m.P_grid = pyo.Var(m.K, domain=pyo.NonNegativeReals) 
    m.E_Gas = pyo.Var(m.K, domain=pyo.NonNegativeReals)
    m.E_el = pyo.Var(m.K, domain=pyo.Reals)

    # --- 6. Constraints ---
    
    def gas_rule(model, k):
        return model.E_Gas[k] == model.dt * (sum(model.Q_B_in[i, k] for i in model.B_set) + sum(model.Q_CHP_in[i, k] for i in model.CHP_set))
    m.con_gas = pyo.Constraint(m.K, rule=gas_rule)

    def el_rule(model, k):
        return model.E_el[k] == model.dt * model.P_grid[k]
    m.con_el = pyo.Constraint(m.K, rule=el_rule)

    # TES Dynamik (NEU: State Space Formulierung)
    def tes_dyn_rule(model, k):
        # E_TES[k+1] = a * E_TES[k] + b1 * Q_TES_in[k] + b2 * Q_TES_out[k]
        return model.E_TES[k+1] == model.a_tes * model.E_TES[k] + \
                                 model.b1_tes * model.Q_TES_in[k] + \
                                 model.b2_tes * model.Q_TES_out[k]
    m.con_tes_dyn = pyo.Constraint(m.K, rule=tes_dyn_rule)

    def tes_e_max_rule(model, k):
        return model.E_TES[k] <= model.E_TES_nom
    m.con_tes_e_max = pyo.Constraint(m.K0, rule=tes_e_max_rule)

    def tes_in_max_rule(model, k):
        return model.Q_TES_in[k] <= model.delta_TES_in[k] * (model.E_TES_nom / model.tau_in)
    m.con_tes_in_max = pyo.Constraint(m.K, rule=tes_in_max_rule)

    def tes_in_min_rule(model, k):
        return model.Q_TES_in[k] >= model.delta_TES_in[k] * model.Q_TES_in_min
    m.con_tes_in_min = pyo.Constraint(m.K, rule=tes_in_min_rule)

    def tes_out_max_rule(model, k):
        return model.Q_TES_out[k] <= model.delta_TES_out[k] * (model.E_TES_nom / model.tau_out)
    m.con_tes_out_max = pyo.Constraint(m.K, rule=tes_out_max_rule)

    def tes_out_min_rule(model, k):
        return model.Q_TES_out[k] >= model.delta_TES_out[k] * model.Q_TES_out_min
    m.con_tes_out_min = pyo.Constraint(m.K, rule=tes_out_min_rule)

    def tes_cycle_rule(model):
        return model.E_TES[0] == model.E_TES[N_steps]
    m.con_tes_cycle = pyo.Constraint(rule=tes_cycle_rule)

    def tes_excl_rule(model, k):
        return model.delta_TES_in[k] + model.delta_TES_out[k] <= 1
    m.con_tes_excl = pyo.Constraint(m.K, rule=tes_excl_rule)

    # Boiler Constraints
    def boiler_out_rule(model, i, k):
        term_1 = model.delta_B[i, k] * model.lam_B_out_min
        term_2 = (1 / model.beta_B) * ((model.Q_B_in[i, k] * model.eta_B_nom / model.Q_B_out_nom) - model.delta_B[i, k] * model.lam_B_in_min)
        return model.Q_B_out[i, k] == model.Q_B_out_nom * (term_1 + term_2)
    m.con_boiler_out = pyo.Constraint(m.B_set, m.K, rule=boiler_out_rule)

    def boiler_in_max_rule(model, i, k):
        return model.Q_B_in[i, k] <= model.delta_B[i, k] * (model.Q_B_out_nom / model.eta_B_nom)
    m.con_boiler_in_max = pyo.Constraint(m.B_set, m.K, rule=boiler_in_max_rule)

    def boiler_in_min_rule(model, i, k):
        return model.Q_B_in[i, k] >= model.delta_B[i, k] * model.lam_B_in_min * (model.Q_B_out_nom / model.eta_B_nom)
    m.con_boiler_in_min = pyo.Constraint(m.B_set, m.K, rule=boiler_in_min_rule)

    # CHP Constraints
    def chp_th_out_rule(model, i, k):
        term_1 = model.delta_CHP[i, k] * model.lam_CHP_th_out_min
        term_2 = (1 / model.beta_CHP_th) * ((model.Q_CHP_in[i, k] * model.eta_CHP_th_nom / model.Q_CHP_out_nom) - model.delta_CHP[i, k] * model.lam_CHP_in_min)
        return model.Q_CHP_out[i, k] == model.Q_CHP_out_nom * (term_1 + term_2)
    m.con_chp_th_out = pyo.Constraint(m.CHP_set, m.K, rule=chp_th_out_rule)

    def chp_el_out_rule(model, i, k):
        term_1 = model.delta_CHP[i, k] * model.lam_CHP_el_out_min
        term_2 = (1 / model.beta_CHP_el) * ((model.Q_CHP_in[i, k] * model.eta_CHP_el_nom / model.P_CHP_out_nom) - model.delta_CHP[i, k] * model.lam_CHP_in_min)
        return model.P_CHP_out[i, k] == model.P_CHP_out_nom * (term_1 + term_2)
    m.con_chp_el_out = pyo.Constraint(m.CHP_set, m.K, rule=chp_el_out_rule)

    def chp_in_max_rule(model, i, k):
        return model.Q_CHP_in[i, k] <= model.delta_CHP[i, k] * (model.Q_CHP_out_nom / model.eta_CHP_th_nom)
    m.con_chp_in_max = pyo.Constraint(m.CHP_set, m.K, rule=chp_in_max_rule)

    def chp_in_min_rule(model, i, k):
        # Gemäß Vorgabe aus dem PDF wird hier P_out_nom und eta_el_nom zur Berechnung der Input-Kapazität genutzt
        return model.Q_CHP_in[i, k] >= model.delta_CHP[i, k] * model.lam_CHP_in_min * (model.P_CHP_out_nom / model.eta_CHP_el_nom)
    m.con_chp_in_min = pyo.Constraint(m.CHP_set, m.K, rule=chp_in_min_rule)

    # Demand Constraints (ACHTUNG: PDF verlangt hier strikte Gleichheit == )
    def heat_demand_rule(model, k):
        return model.Q_D[k] == sum(model.Q_CHP_out[i, k] for i in model.CHP_set) + \
                               sum(model.Q_B_out[i, k] for i in model.B_set) + \
                               model.Q_TES_out[k] - model.Q_TES_in[k]
    m.con_heat_demand = pyo.Constraint(m.K, rule=heat_demand_rule)

    def el_demand_rule(model, k):
        return model.P_D[k] == sum(model.P_CHP_out[i, k] for i in model.CHP_set) + model.P_grid[k]
    m.con_el_demand = pyo.Constraint(m.K, rule=el_demand_rule)

    # --- 7. Zielfunktion ---
    def obj_rule(model):
        return sum(model.c_G[k] * model.E_Gas[k] + model.c_el[k] * model.E_el[k] for k in model.K)
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # --- 8. Lösen ---
    solver = pyo.SolverFactory('gurobi')
    solver.options['MIPGap'] = 0.001 
    solver.options['TimeLimit'] = 60  # Optional: Zeitlimit von 5 Minuten setzen
    
    print("Starte MILP Optimierung (Exakte Diskretisierung)...")
    results = solver.solve(m, tee=True) 
    
    print("\n--- Ergebnisse ---")
    print(f"Status: {results.solver.status}")
    print(f"Termination Condition: {results.solver.termination_condition}")
    print(f"Optimale Kosten: {pyo.value(m.obj):.2f} €")
    
    return m, results

# ==========================================
# AUFRUF DES SKRIPTS
# ==========================================

# after solve:




# # --- Terminal table output: show OPEX for this run in the same format as LP_V2.py ---
# try:
#     opex = float(pyo.value(model_milp_exact.obj))
#     gas_price_MWh = float(model_milp_exact.c_G[0] * 1000)
#     electricity_price_MWh = float(model_milp_exact.c_el[0] * 1000)
#     disp = pd.DataFrame([{
#         'scenario': 1,
#         'gas_price_MWh': gas_price_MWh,
#         'electricity_price_MWh': electricity_price_MWh,
#         'opex': opex,
#     }])
#     disp['opex'] = disp['opex'].map(lambda x: f"{x:,.2f} €")
#     print('\nOPEX per scenario:')
#     print(disp.to_string(index=False))
# except Exception as e:
#     print(f"Fehler beim Erstellen der Ergebnis-Tabelle: {e}")

price_csv_path = "Erdem/training_samples_sobol.csv"
demand_csv_path = "Erdem/energy_demands.csv"

prices_df = pd.read_csv(price_csv_path)
scenario_results = []
for idx, row in prices_df.iterrows():
    gas_mwh = float(row['gas_price'])
    el_mwh = float(row['electricity_price'])
    # Preise in CSV sind in €/MWh, demands in kWh -> Umrechnung in €/kWh
    gas_kwh = gas_mwh / 1000.0
    el_kwh = el_mwh / 1000.0
    model_s, results_s = run_mode_project_milp_exact(c_gas=gas_kwh, c_el=el_kwh, demand_csv_path=demand_csv_path)
    opex = float(pyo.value(model_s.obj))
    scenario_results.append({'gas_price_MWh': gas_mwh, 'electricity_price_MWh': el_mwh, 'opex': f"{opex:.2f}"})
    
df_res = pd.DataFrame(scenario_results)

# --- Save all scenario results to CSV ---
df_res.to_csv("Florian/OPEX_results_scenarios_MILP.csv", index=False)
print(f"\nOPEX results saved to Florian/OPEX_results_scenarios_MILP.csv")
