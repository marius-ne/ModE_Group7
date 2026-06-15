import pyomo.environ as pyo
import pandas as pd
import numpy as np

def run_mode_project_LP(c_gas=0.16, c_el=0.21, demand_csv_path=None):
    """
    Erstellt und löst das MILP-Modell basierend auf der exakten Diskretisierung 
    (Zustandsraummodell) des Speichers. Demand ist wieder streng durch == gekoppelt.
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
    c_gas_dict = {k: float(c_gas) for k in range(1, N_steps + 1)}
    c_el_dict = {k: float(c_el) for k in range(1, N_steps + 1)}

    m.c_G = pyo.Param(m.K, initialize=c_gas_dict)
    m.c_el = pyo.Param(m.K, initialize=c_el_dict)

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
    m.eta_CHP_el_mid = pyo.Param(initialize=0.41705)

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
        return model.E_TES[0] == 0
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
        return sum(model.c_G[k] * model.E_Gas[k] + model.c_el[k] * model.E_el[k] for k in model.K)
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # --- 8. Lösen ---
    solver = pyo.SolverFactory('gurobi')
    solver.options['MIPGap'] = 0.01 
    
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

price_csv_path = "Erdem/training_samples_sobol.csv"
demand_csv_path = "Erdem/energy_demands.csv"

prices_df = pd.read_csv(price_csv_path)
results_list = []
print(f"Lade Preis-Szenarien aus {price_csv_path}, insgesamt {len(prices_df)} Szenarien.")
for idx, row in prices_df.iterrows():
    gas_mwh = float(row['gas_price'])
    el_mwh = float(row['electricity_price'])
    # Preise in CSV sind in €/MWh, demands in kWh -> Umrechnung in €/kWh
    gas_kwh = gas_mwh / 1000.0
    el_kwh = el_mwh / 1000.0
    print(f"Szenario {idx+1}/{len(prices_df)}: gas={gas_mwh} €/MWh, el={el_mwh} €/MWh -> gas={gas_kwh} €/kWh, el={el_kwh} €/kWh")
    model_s, results_s = run_mode_project_LP(c_gas=gas_kwh, c_el=el_kwh, demand_csv_path=demand_csv_path)
    opex = float(pyo.value(model_s.obj))
    results_list.append({'gas_price_MWh': gas_mwh, 'electricity_price_MWh': el_mwh, 'opex': opex})
    print(f"Szenario {idx+1}/{len(prices_df)} Ergebnis: OPEX = {opex:.2f} €, gas={gas_mwh} €/MWh, el={el_mwh} €/MWh")

df_res = pd.DataFrame(results_list)
# --- Terminal table output: show OPEX for all scenarios in a nice table ---
try:
    disp = df_res.copy()
    # add scenario index (1-based) and format OPEX
    disp.insert(0, 'scenario', range(1, len(disp) + 1))
    disp['opex'] = disp['opex'].map(lambda x: f"{x:,.2f} €")
    disp = disp[['scenario', 'gas_price_MWh', 'electricity_price_MWh', 'opex']]
    print('\nOPEX per scenario:')
    print(disp.to_string(index=False))
except Exception as e:
    print(f"Fehler beim Erstellen der Ergebnis-Tabelle: {e}")
import matplotlib.pyplot as plt
plt.figure(figsize=(8, 6))
# Scatter plot: one point per scenario, colored by OPEX
sc = plt.scatter(df_res['electricity_price_MWh'], df_res['gas_price_MWh'], c=df_res['opex'], cmap='viridis', s=80, edgecolors='k')
plt.xlabel('Electricity price (€/MWh)')
plt.ylabel('Gas price (€/MWh)')
plt.title('OPEX scatter (each point = scenario)')
cbar = plt.colorbar(sc)
cbar.set_label('OPEX (€)')
plt.tight_layout()
plt.show()
