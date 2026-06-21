
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # NEU: Bibliothek für den Plot importieren

df_milp = pd.read_csv("Florian/OPEX_results_scenarios_MILP.csv", sep=',')

# FEHLERKORREKTUR: df_coefficient_list muss erst erstellt werden, 
# bevor du ihm eine Spalte zuweisen kannst.
df_coefficient_list = pd.DataFrame(index=df_milp.index)

# Deine ursprüngliche Berechnung
df_coefficient_list["gas_el_relation"] = (df_milp["gas_price_MWh"] / df_milp["electricity_price_MWh"])

# Dein ursprünglicher Merge
df_opex_list = pd.merge(df_coefficient_list, df_milp[['opex']], left_index=True, right_index=True, suffixes=('_coef', '_opex'))

# ==========================================
# NEU: PLOT ERSTELLEN
# ==========================================
plt.figure(figsize=(18, 6))

# Streudiagramm (Scatterplot): x = Verhältnis, y = OPEX
plt.scatter(df_opex_list["gas_el_relation"], df_opex_list["opex"], color="blue", alpha=0.7)

# Achsenbeschriftungen und Titel
plt.title("OPEX vs Price Ration for MILP Florian", fontsize=14)
plt.xlabel("Price ratio (c_G/c_el)")
plt.ylabel("OPEX")

# Ein leichtes Raster für die bessere Lesbarkeit
plt.grid(True, linestyle='--', alpha=0.6)

# Plot anzeigen
plt.xscale('log')  # Optional: Logarithmische Skalierung für die x-Achse, falls die Werte stark variieren
plt.xlim(0, 100)
plt.show()
plt.savefig("Florian/Visualizations/OPEX_vs_price_ratio_MILP.png", dpi=300)