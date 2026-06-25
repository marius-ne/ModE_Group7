import pandas as pd
import joblib
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sicherstellen, dass die Ordner existieren
os.makedirs("Florian", exist_ok=True)
os.makedirs("Florian/results/regression", exist_ok=True)

# Lade den Datensatz (Passe den Dateipfad an, falls nötig)
df = pd.read_csv("Marius/results/evaluation_log_samples.csv", sep=',')
df_test = pd.read_csv("Marius/results/opex_random_sample_10.csv")
# Die unabhängige Variable (Feature) ist für alle Modelle die Ratio
# sklearn erwartet ein 2D-Array, daher doppelte eckige Klammern
X_train = df[["ratio"]]
X_test = df_test[["ratio"]]

# Liste der abhängigen Variablen (Targets)
opex_columns = ["opex_milp", "opex_lp_lower", "opex_lp_upper", "opex_lp_approx"]
test_opex_columns = {
    "opex_milp": "opex_milp",
    "opex_lp_lower": "opex_lp_lower",
    "opex_lp_upper": "opex_lp_upper",
    "opex_lp_approx": "opex_lp_approximated",
}
combined_predictions = []

# Schleife über alle OPEX-Spalten
for target in opex_columns:
    print(f"\n{'='*50}")
    print(f"Starte Modelltraining für: {target}")
    print(f"{'='*50}")

    y_train = df[target]
    y_test = df_test[test_opex_columns[target]]

    # Modell initialisieren und anlernen
   
    log_reg = LinearRegression()
    log_reg.fit(X_train, y_train)

    # Vorhersagen treffen
    y_pred = log_reg.predict(X_test)
    combined_predictions.append(pd.DataFrame({
        "target": target,
        "actual": y_test.to_numpy(),
        "predicted": y_pred,
    }))
    
    # Metriken berechnen
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:,.2f}")
    print(f"R^2 Score: {r2:.4f}")
    validation_df = pd.DataFrame({
        "y_test": y_test.to_numpy(),
        "y_pred": y_pred,
    })
    validation_df.to_csv(f"Florian/results/regression/validation_{target}.csv", index=False)
    # Modell abspeichern
    joblib_path = f"Florian/surrogate_model_{target}.joblib"
    joblib.dump(log_reg, joblib_path)
    print(f"Modell gespeichert unter: {joblib_path}")

    # Basiswert und Koeffizienten ausgeben
    base_value = log_reg.intercept_
    coefficients = log_reg.coef_
    print(f"Basiswert (Intercept): {base_value:,.2f}")
    print(f"Ratio-Koeffizient: {coefficients[0]:,.2f}")

    # Plot der tatsächlichen vs. vorhergesagten Werte
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, color='blue', edgecolors='black', alpha=0.7)
    
    # Perfekte Vorhersagelinie (Diagonale) berechnen basierend auf Min/Max-Werten
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Ideale Vorhersage')
    
    plt.xlabel(f'Actual  ')
    plt.ylabel(f'Predicted ')
    plt.title(f'Actual vs. Predicted OPEX/C_el[€/kwh] ({target});R^2:{r2:.4f}; MSE={mse:,.2f}')
    plt.legend()
    plt.grid(True)
    
    # Dynamische Achsenskalierung (statt harter 50.000er Grenze)
    plt.xlim(min_val * 0.9, max_val * 1.05)
    plt.ylim(min_val * 0.9, max_val * 1.05)
    
    # Plot speichern und anzeigen
    plot_path = f"Florian/results/regression/actual_vs_predicted_{target}.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

# Gemeinsamer Scatterplot für alle Modelle.
# Nur die Testdatenpunkte werden geplottet; die Modelle werden nur über die Farben unterschieden.
combined_df = pd.concat(combined_predictions, ignore_index=True)

plt.figure(figsize=(9, 7))
colors = {
    "opex_milp": "tab:blue",
    "opex_lp_lower": "tab:orange",
    "opex_lp_upper": "tab:green",
    "opex_lp_approx": "tab:red",
}

for target in opex_columns:
    target_df = combined_df[combined_df["target"] == target]
    plt.scatter(
        target_df["actual"],
        target_df["predicted"],
        color=colors[target],
        alpha=0.65,
        edgecolors="black",
        linewidths=0.4,
        label=target,
    )

min_val = min(combined_df["actual"].min(), combined_df["predicted"].min())
max_val = max(combined_df["actual"].max(), combined_df["predicted"].max())
plt.plot([min_val, max_val], [min_val, max_val], "k--", lw=2, label="Ideal prediciton")

plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.title("Actual vs. Predicted OPEX/C_el - alle Modelle")
plt.legend(title="Model")
plt.grid(True)
plt.xlim(min_val * 0.9, max_val * 1.05)
plt.ylim(min_val * 0.9, max_val * 1.05)

combined_plot_path = "Florian/results/regression/actual_vs_predicted_all_models.png"
plt.savefig(combined_plot_path, dpi=300)
plt.close()
