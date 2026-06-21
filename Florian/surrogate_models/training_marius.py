import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Sicherstellen, dass der Ordner existiert
os.makedirs("Florian", exist_ok=True)

# Lade den Datensatz (Passe den Dateipfad an, falls nötig)
df = pd.read_csv("Marius/results/evaluation_log_samples.csv", sep=',')

# Die unabhängige Variable (Feature) ist für alle Modelle die Ratio
# sklearn erwartet ein 2D-Array, daher doppelte eckige Klammern
X = df[["ratio"]]

# Liste der abhängigen Variablen (Targets)
opex_columns = ["opex_milp", "opex_lp_lower", "opex_lp_upper", "opex_lp_approx"]

# Schleife über alle OPEX-Spalten
for target in opex_columns:
    print(f"\n{'='*50}")
    print(f"Starte Modelltraining für: {target}")
    print(f"{'='*50}")

    y = df[target]

    # Train-Test-Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modell initialisieren und anlernen
   
    log_reg = LinearRegression()
    log_reg.fit(X_train, y_train)

    # Vorhersagen treffen
    y_pred = log_reg.predict(X_test)
    
    # Metriken berechnen
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"Mean Squared Error: {mse:,.2f}")
    print(f"R^2 Score: {r2:.4f}")

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
    plt.show()