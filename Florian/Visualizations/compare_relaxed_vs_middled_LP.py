import pandas as pd
import matplotlib.pyplot as plt

relaxed_df = pd.read_csv("Florian/results/OPEX_results_scenarios_LP_marius_price_factors.csv", sep=',')
middled_df = pd.read_csv("Florian/results/OPEX_results_scenarios_LP_price_factors.csv", sep=',')

relaxed_df = relaxed_df.rename(columns={"opex/C_el[€/kWh]": "opex_relaxed"})
middled_df = middled_df.rename(columns={"opex/C_el[€/kWh]": "opex_middled"})

plot_df = pd.merge(
    relaxed_df[["price_factor", "opex_relaxed"]],
    middled_df[["price_factor", "opex_middled"]],
    on="price_factor",
    how="inner",
)

plt.figure(figsize=(18, 6))
plt.plot(plot_df["price_factor"], plot_df["opex_relaxed"], label="Relaxed LP (Marius)", marker='o')
plt.plot(plot_df["price_factor"], plot_df["opex_middled"], label="Middled LP", marker='x')

plt.xlabel("price_factor")
plt.ylabel("opex / C_el [€/kWh]")
plt.title("Relaxed vs Middled LP: OPEX over price factor")
plt.grid(True)
plt.legend()
plt.tight_layout()
#plt.yscale('log')

plt.savefig("Florian/visualizations/compare_relaxed_vs_middled_LP.png", dpi=300)
plt.show()
