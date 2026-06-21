import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("Florian/results/OPEX_results_scenarios_LP_price_factors.csv", sep=',')
X = df[["price_factor"]]
y = df["opex/C_el[€/kWh]"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_reg = LinearRegression()
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")


joblib.dump(log_reg, "Florian/surrogate_models/surrogate_model_LP_price_factors_training.joblib")


#Plot der tatsächlichen vs. vorhergesagten Werte
import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred, color='blue', edgecolors='black')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Actual OPEX/C_el[€/kWh]')
plt.ylabel('Predicted OPEX/C_el[€/kWh]')
plt.title(f'Actual vs. Predicted OPEX/C_el[€/kWh]; r2={r2:.4f}; mse={mse:.4f}')
plt.grid()
plt.savefig("Florian/visualizations/surrogate_model_LP_ratios_training_actual_vs_predicted.png", dpi=300)
plt.show()

base_value = log_reg.intercept_
print(f"Basiswert (Intercept): {base_value}")
coefficients = log_reg.coef_
print(f"Price Factor Coefficient: {coefficients[0]}")