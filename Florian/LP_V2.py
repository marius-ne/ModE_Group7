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
# AUFRUF DES SKRIPTS
# ==========================================
csv_pfad = "energy_demands.csv"
model_milp_exact, results_milp_exact = run_mode_project_LP(demand_csv_path=csv_pfad)

import matplotlib.pyplot as plt

# after solve:
model = model_milp_exact

data = []
for k in model.K:
    e_tes = pyo.value(model.E_TES[k])
    q_chp_1= pyo.value(model.Q_CHP_out[1, k])
    q_chp_2 = pyo.value(model.Q_CHP_out[2, k])
    q_b_1 = pyo.value(model.Q_B_out[1, k])
    q_b_2 = pyo.value(model.Q_B_out[2, k])
    q_t_in = pyo.value(model.Q_TES_in[k])
    q_t_out = pyo.value(model.Q_TES_out[k])
    q_chp = q_chp_1 + q_chp_2
    q_b = q_b_1 + q_b_2
    e_grid = pyo.value(model.P_grid[k])
    data.append({'k': k, 'energy': e_tes, 'q_chp': q_chp, 'q_b': q_b, 'e_grid': e_grid, 'q_t_in': q_t_in, 'q_t_out': q_t_out})

x = [row['k'] for row in data]

y_1 = [row['energy'] for row in data]


y_2 = [row['q_chp'] for row in data]

y_3 = [row['q_b'] for row in data]


y_4 = [row['e_grid'] for row in data]

y_5 = [row['q_t_in'] for row in data]

y_6 = [row['q_t_out'] for row in data]

fig, axs = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

axs[0].plot(x, y_1, label='E_TES', color='tab:blue')
axs[0].set_title('TES Energy Content')
axs[0].set_ylabel('E_TES')
axs[0].legend()
axs[0].grid(True)

axs[1].plot(x, y_2, label='Q_CHP_out', color='tab:orange')
axs[1].plot(x, y_3, label='Q_B_out', color='tab:green')
axs[1].plot(x, y_4, label='P_grid', color='tab:red')
axs[1].plot(x, y_5, label='Q_TES_in', color='tab:purple')
axs[1].plot(x, y_6, label='Q_TES_out', color='tab:brown')
axs[1].set_title('Plant Outputs')
axs[1].set_xlabel('Time step k')
axs[1].set_ylabel('Power / Heat')
axs[1].legend()
axs[1].grid(True)

plt.tight_layout()
plt.show()
