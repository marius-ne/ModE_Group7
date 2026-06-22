import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

df_milp = pd.read_csv("Florian/results/regression/validation_opex_milp.csv")
df_lp_upper = pd.read_csv("Florian/results/regression/validation_opex_lp_upper.csv")
df_lp_lower = pd.read_csv("Florian/results/regression/validation_opex_lp_lower.csv")
df_lp_approx = pd.read_csv("Florian/results/regression/validation_opex_lp_approx.csv")


df_discrepancy_comparison = pd.concat([df_milp["y_test"],df_milp['y_pred'], df_lp_upper["y_pred"], df_lp_lower["y_pred"], df_lp_approx["y_pred"]], axis=1, keys=['_milp_actual', '_milp_pred', '_lp_upper_pred', '_lp_lower_pred', '_lp_approx_pred'])
print(df_discrepancy_comparison.head())

r2_list = []

r2_milp = r2_score(df_discrepancy_comparison['_milp_actual'], df_discrepancy_comparison['_milp_pred'])
r2_upper = r2_score(df_discrepancy_comparison['_milp_actual'], df_discrepancy_comparison['_lp_upper_pred'])
r2_lower = r2_score(df_discrepancy_comparison['_milp_actual'], df_discrepancy_comparison['_lp_lower_pred'])
r2_approx = r2_score(df_discrepancy_comparison['_milp_actual'], df_discrepancy_comparison['_lp_approx_pred'])

r2_list.append({
    "model": "MILP_pred",
    "r2_relative_to_MILP": r2_milp,
    "delta_to_milp" : "0.0"
})
r2_list.append({
    "model": "LP_Upper",
    "r2_relative_to_MILP": r2_upper,
    "delta_to_milp": r2_upper - r2_milp
})
r2_list.append({
    "model": "LP_Lower",
    "r2_relative_to_MILP": r2_lower,
    "delta_to_milp": r2_lower - r2_milp
})
r2_list.append({
    "model": "LP_Approx",
    "r2_relative_to_MILP": r2_approx,
    "delta_to_milp": r2_approx - r2_milp
})
r2_df = pd.DataFrame(r2_list)
print(r2_df)

r2_df.to_csv("Florian/results/regression/r2_score_compared.csv", index=False)
