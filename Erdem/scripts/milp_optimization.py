from Erdem.src.optimization.optimization import *
from pathlib import Path

csv_file = Path(__file__).parent.parent / "energy_demands.csv"
Q_D, P_D = load_demands(csv_file)

c_g = 0.12
c_el = 0.25

model = build_milp(Q_D, P_D, c_g, c_el)

results = solve_model(model, MIPGap=1e-3, TimeLimit=300)

solution = extract_solution(model)
