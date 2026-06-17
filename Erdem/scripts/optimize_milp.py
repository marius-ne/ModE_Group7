from src.optimization.core import (
    load_demands,
    build_milp,
    solve_model,
    save_optimization_results
)
from src.sampling.core import load_samples
from src.misc.constants import PROJ_ROOT


if __name__ == "__main__":
    # Load the heat and power demand from the given csv
    csv_file = PROJ_ROOT / "energy_demands.csv"
    Q_D, P_D = load_demands(csv_file)

    # Load the training samples
    lhs_df, _ = load_samples("lhs", "training")
    sobol_df, _ = load_samples("sobol", "training")

    sample_sets = {
        "LHS": lhs_df,
        "Sobol": sobol_df
    }

    # Loop over all samples and solve the MILP
    for method, samples in sample_sets.items():
        for i, (c_g, c_el) in enumerate(zip(samples["gas_price"], samples["electricity_price"]), start=1):

            # Conversion of the price units from €/MWh to €/kWh
            c_g_kwh, c_el_kwh = c_g / 1000, c_el / 1000

            # Build the Pyomo MILP optimization model
            model = build_milp(Q_D, P_D, c_g_kwh, c_el_kwh)

            # Solve the optimization model
            results = solve_model(model, MIPGap=1e-3, TimeLimit=120)

            # Save the optimization results
            sample_id = f"{int(c_g)}_{int(c_el)}_{i:03d}"
            solution_df, metadata_dict = save_optimization_results(model, results, sample_id, "MILP", method)
