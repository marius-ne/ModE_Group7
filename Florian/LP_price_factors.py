import pyomo.environ as pyo
import pandas as pd
import numpy as np

def run_mode_project_LP(price_factor=1.0, demand_csv_path=None):
    """
    Erstellt und löst das LP-Modell basierend auf der exakten Diskretisierung 
    (Zustandsraummodell) des Speichers. Demand ist wieder streng durch == gekoppelt.
    Der Parameter price_factor skaliert die Gas-Kosten in der Zielfunktion.
    """
    m = pyo.ConcreteModel(name="ModE_Project_5_LP_V2")
    
    # --- 1. Sets ---
    N_steps = 168
    m.N = pyo.Param(initialize=N_steps)
    m.K = pyo.RangeSet(1, m.N)
    m.K0 = pyo.RangeSet(0, m.N)
    m.B_set = pyo.Set(initialize=[1, 2])
    m.CHP_set = pyo.Set(initialize=[1, 2])
    
    # --- 2. Preis-Zuweisung ---
    c_pf_dict = {k: float(price_factor) for k in range(1, N_steps + 1)}

    m.pf = pyo.Param(m.K, initialize=c_pf_dict)
 

    # --- 3. Demand Daten Import ---
    if demand_csv_path:
        try:
            df_demand = pd.read_csv(demand_csv_path)
            dict_P_D = {k: float(df_demand.iloc[k-1, 1]) for k in range(1, N_steps + 1)}
            dict_Q_D = {k: float(df_demand.iloc[k-1, 2]) for k in range(1, N_steps + 1)}
            print(f"Daten aus {demand_csv_path} geladen.")
        except Exception as e:
            print(f"Fehler beim Laden der CSV: {e}. Verwende Dummy-Daten.")
            demand_csv_path = None

    if not demand_csv_path:
        np.random.seed(42)
        dict_P_D = {k: float(np.random.uniform(100, 500)) for k in range(1, N_steps + 1)}
        dict_Q_D = {k: float(np.random.uniform(300, 900)) for k in range(1, N_steps + 1)}

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
    m.eta_B_mid = pyo.Param(initialize=0.86245)

    # CHP
    m.Q_CHP_out_nom = pyo.Param(initialize=470.0)
    m.P_CHP_out_nom = pyo.Param(initialize=380.0)
    m.eta_CHP_th_mid = pyo.Param(initialize=0.49753)
    m.eta_CHP_el_mid = pyo.Param(initialize=0.36162)

    # --- 5. Variablen ---

    m.Q_B_in = pyo.Var(m.B_set, m.K, domain=pyo.NonNegativeReals)
    m.Q_B_out = pyo.Var(m.B_set, m.K, domain=pyo.NonNegativeReals)
    
    m.Q_CHP_in = pyo.Var(m.CHP_set, m.K, domain=pyo.NonNegativeReals)
    m.Q_CHP_out = pyo.Var(m.CHP_set, m.K, domain=pyo.NonNegativeReals)
    m.P_CHP_out = pyo.Var(m.CHP_set, m.K, domain=pyo.NonNegativeReals)
    
    m.Q_TES_in = pyo.Var(m.K, domain=pyo.NonNegativeReals)
    m.Q_TES_out = pyo.Var(m.K, domain=pyo.NonNegativeReals)
    m.E_TES = pyo.Var(m.K0, bounds=(0, m.E_TES_nom)) 
    
    m.P_grid = pyo.Var(m.K, domain=pyo.Reals) 
    m.E_Gas = pyo.Var(m.K, domain=pyo.NonNegativeReals)
    m.E_el = pyo.Var(m.K, domain=pyo.Reals)

    # --- 6. Constraints ---
    
    def gas_rule(model, k):
        return model.E_Gas[k] == model.dt * (sum(model.Q_B_in[i, k] for i in model.B_set) + sum(model.Q_CHP_in[i, k] for i in model.CHP_set))
    m.con_gas = pyo.Constraint(m.K, rule=gas_rule)

    def el_rule(model, k):
        return model.E_el[k] == model.dt * model.P_grid[k] # E_el ist menege in KWh, P_grid ist leistung in KW
    m.con_el = pyo.Constraint(m.K, rule=el_rule)

    def p_min_rule(model, k):
        return model.P_grid[k] >= 0
    m.con_p_min = pyo.Constraint(m.K, rule=p_min_rule)

    # TES Dynamik (NEU: State Space Formulierung)
    def tes_dyn_rule(model, k):
        # E_TES,k = a*E_TES_k-1 + b1*Q_in + b2*Q_out
        # (Achtung: b2 ist als -1.05 definiert, daher addieren wir es einfach)
        return model.E_TES[k] == model.a_tes * model.E_TES[k-1] + \
                                 model.b1_tes * model.Q_TES_in[k] + \
                                 model.b2_tes * model.Q_TES_out[k]
    m.con_tes_dyn = pyo.Constraint(m.K, rule=tes_dyn_rule)

    def tes_e_max_rule(model, k):
        return model.E_TES[k] <= model.E_TES_nom
    m.con_tes_e_max = pyo.Constraint(m.K, rule=tes_e_max_rule)

    def tes_in_max_rule(model, k):
        return model.Q_TES_in[k] <= (model.E_TES_nom / model.tau_in)
    m.con_tes_in_max = pyo.Constraint(m.K, rule=tes_in_max_rule)


    def tes_out_max_rule(model, k):
        return model.Q_TES_out[k] <= (model.E_TES_nom / model.tau_out)
    m.con_tes_out_max = pyo.Constraint(m.K, rule=tes_out_max_rule)

    def tes_cycle_rule(model):
        return model.E_TES[0] == model.a_tes * model.E_TES[model.N] + \
               model.b1_tes * model.Q_TES_in[model.N] + \
               model.b2_tes * model.Q_TES_out[model.N]
    m.con_tes_cycle = pyo.Constraint(rule=tes_cycle_rule)

    # Boiler Constraints


    # def boiler_out_rule(model, i, k):
    def boiler_out_rule(model, i, k):
        
        return model.Q_B_out[i, k] == model.Q_B_in[i, k]* model.eta_B_mid  
    m.con_boiler_out = pyo.Constraint(m.B_set, m.K, rule=boiler_out_rule)

    def boiler_in_max_rule(model, i, k):
        return model.Q_B_in[i, k] <=  (model.Q_B_out_nom / model.eta_B_mid)
    m.con_boiler_in_max = pyo.Constraint(m.B_set, m.K, rule=boiler_in_max_rule)

    def boiler_in_min_rule(model, i, k):
        return model.Q_B_in[i, k] >= 0
    m.con_boiler_in_min = pyo.Constraint(m.B_set, m.K, rule=boiler_in_min_rule)

    # CHP Constraints
    def chp_th_out_rule(model, i, k):
        
        return model.Q_CHP_out[i, k] ==  model.Q_CHP_in[i, k] * model.eta_CHP_th_mid 
    m.con_chp_th_out = pyo.Constraint(m.CHP_set, m.K, rule=chp_th_out_rule)

    def chp_el_out_rule(model, i, k):
       
       
        return model.P_CHP_out[i, k] == model.Q_CHP_in[i, k] * model.eta_CHP_el_mid 
    m.con_chp_el_out = pyo.Constraint(m.CHP_set, m.K, rule=chp_el_out_rule)

    def chp_in_max_rule(model, i, k):
        return model.Q_CHP_in[i, k] <= (model.Q_CHP_out_nom / model.eta_CHP_th_mid)
    m.con_chp_in_max = pyo.Constraint(m.CHP_set, m.K, rule=chp_in_max_rule)


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
        return sum(model.pf[k] * model.E_Gas[k] +  model.E_el[k] for k in model.K)
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # --- 8. Lösen ---
    solver = pyo.SolverFactory('gurobi')
    solver.options['MIPGap'] = 0.001  # 0.1% optimality gap
    
    print("Starte MILP Optimierung (Exakte Diskretisierung)...")
    results = solver.solve(m, tee=True) 
    
    print("\n--- Ergebnisse ---")
    print(f"Status: {results.solver.status}")
    print(f"Termination Condition: {results.solver.termination_condition}")
    print(f"Optimale Kosten: {pyo.value(m.obj):.2f} €")
    
    return m, results

# ==========================================
# AUFRUF DES SKRIPTS: Szenarien durchlaufen und OPEX auswerten
# ==========================================

if __name__ == "__main__":
    price_csv_path = "Erdem/results/Sampling/log_training_samples.csv"
    demand_csv_path = "Erdem/energy_demands.csv"

    prices_df = pd.read_csv(price_csv_path)
    scenario_results = []
    dispatch_results = []
    print(f"Lade Preis-Szenarien aus {price_csv_path}, insgesamt {len(prices_df)} Szenarien.")
    for idx, row in prices_df.iterrows():
        price_factor = float(row['ratios'])
        model_s, results_s = run_mode_project_LP(price_factor=price_factor, demand_csv_path=demand_csv_path)
        opex = float(pyo.value(model_s.obj))
        scenario_results.append({'price_factor': price_factor, 'opex/C_el[€/kWh]': f"{opex:.2f}"})

        for k in range(1, 169):
            dispatch_results.append({
                'price_factor': price_factor,
                'k': k,
                'Q_D': float(pyo.value(model_s.Q_D[k])),
                'P_D': float(pyo.value(model_s.P_D[k])),
                'E_TES': float(pyo.value(model_s.E_TES[k])),
                'Qin_TES': float(pyo.value(model_s.Q_TES_in[k])),
                'Qout_TES': float(pyo.value(model_s.Q_TES_out[k])),
                'Pgrid': float(pyo.value(model_s.P_grid[k])),
                'Qin_B1': float(pyo.value(model_s.Q_B_in[1, k])),
                'Qin_B2': float(pyo.value(model_s.Q_B_in[2, k])),
                'Qout_B1': float(pyo.value(model_s.Q_B_out[1, k])),
                'Qout_B2': float(pyo.value(model_s.Q_B_out[2, k])),
                'Qin_CHP1': float(pyo.value(model_s.Q_CHP_in[1, k])),
                'Qin_CHP2': float(pyo.value(model_s.Q_CHP_in[2, k])),
                'Qout_CHP1': float(pyo.value(model_s.Q_CHP_out[1, k])),
                'Qout_CHP2': float(pyo.value(model_s.Q_CHP_out[2, k])),
                'Pout_CHP1': float(pyo.value(model_s.P_CHP_out[1, k])),
                'Pout_CHP2': float(pyo.value(model_s.P_CHP_out[2, k])),
            })

    df_res = pd.DataFrame(scenario_results)
    df_dispatch = pd.DataFrame(dispatch_results)

    # --- Save all scenario results to CSV ---
    df_res.to_csv("Florian/results/OPEX_results_scenarios_LP_price_factors.csv", index=False)
    df_dispatch.to_csv("Florian/results/OPEX_dispatch_scenarios_LP_price_factors.csv", index=False)
    print(f"\nOPEX results saved to Florian/results/OPEX_results_scenarios_LP_price_factors.csv")
    print(f"Dispatch results saved to Florian/results/OPEX_dispatch_scenarios_LP_price_factors.csv")
