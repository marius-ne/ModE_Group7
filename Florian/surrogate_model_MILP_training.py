import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("Florian/OPEX_results_scenarios_MILP.csv", sep=',')
X = df[["gas_price_MWh","electricity_price_MWh"]]
y = df["opex"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

log_reg = LinearRegression()
log_reg.fit(X_train, y_train)

y_pred = log_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")


joblib.dump(log_reg, "Florian/surrogate_model_MILP_training.joblib")


#Plot der tatsächlichen vs. vorhergesagten Werte
import matplotlib.pyplot as plt
plt.scatter(y_test, y_pred, color='blue', edgecolors='black')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('Tatsächliche OPEX')
plt.ylabel('Vorhergesagte OPEX')
plt.title('Tatsächliche vs. Vorhergesagte OPEX')
plt.grid()
plt.xlim(0, 50000)
plt.ylim(0, 50000)
plt.savefig("Florian/surrogate_model_MILP_training_actual_vs_predicted.png", dpi=300)
plt.show()

base_value = log_reg.intercept_
print(f"Basiswert (Intercept): {base_value}")
coefficients = log_reg.coef_
print(f"Gaspreis-Koeffizient: {coefficients[0]}")
print(f"Strompreis-Koeffizient: {coefficients[1]}")