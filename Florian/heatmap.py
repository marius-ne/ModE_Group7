# 1. Daten aus dem Dictionary extrahieren
x_el_prices = [data['price_el'] for data in results_all_scenarios.values()]
y_gas_prices = [data['price_gas'] for data in results_all_scenarios.values()]
c_opex = [data['cost'] for data in results_all_scenarios.values()]

# 2. Plot erstellen
plt.figure(figsize=(10, 8))

# Scatterplot mit Farbskala (cmap='viridis' ist eine gute Standard-Farbskala)
# s=100 definiert die Punktgröße, edgecolors='black' macht einen kleinen Rand um die Punkte
scatter = plt.scatter(x_el_prices, y_gas_prices, c=c_opex, cmap='viridis', s=100, edgecolors='black')

# 3. Farblegende (Colorbar) hinzufügen
cbar = plt.colorbar(scatter)
cbar.set_label('Gesamtkosten (OPEX) in €', fontsize=12)

# 4. Achsenbeschriftungen und Titel
plt.xlabel('Strompreis [€/MWh]', fontsize=12)
plt.ylabel('Gaspreis [€/MWh]', fontsize=12)
plt.title('Szenario-Analyse: OPEX in Abhängigkeit von Energiepreisen', fontsize=14)

# Ein leichtes Raster im Hintergrund hilft beim Ablesen
plt.grid(True, linestyle='--', alpha=0.6)

# Plot anzeigen
plt.show()