import pandas as pd
import numpy as np

df_milp = pd.read_csv("Florian/OPEX_results_scenarios_MILP.csv", sep=',')
df_lp = pd.read_csv("Florian/OPEX_results_scenarios_LP.csv", sep=',')

df_opex_list = pd.merge(df_milp[['opex']], df_lp[['opex']], left_index=True, right_index=True, suffixes=('_milp', '_lp'))
print(df_opex_list.head())

df_opex_list["deviation_percentage_relative_to_MILP"] = (df_opex_list["opex_lp"] - df_opex_list["opex_milp"]) / df_opex_list["opex_milp"] * 100

df_opex_list.to_csv("Florian/OPEX_discrepancy_MILP_vs_LP.csv", index=False)