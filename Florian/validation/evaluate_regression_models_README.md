# Regression Model Evaluation Script

## Overview

`evaluate_regression_models.py` ist ein umfassendes Python-Skript zur Evaluierung von zwei Arten von Regressionmodellen:

1. **1D-Modelle** (Formel: $m \cdot x + b$)
   - Feature: Preisverhältnis (ratio = Gaspreis / Strompreis)
   - Trainiert auf 2, 5, 20, und 40 Samples
   - Testdaten: `Marius/results/evaluation_lhs_10_test_1D.csv`

2. **2D-Modelle** (Formel: $a \cdot c_g + b \cdot c_{el} + c$)
   - Features: Gaspreis und Strompreis (diskrete Preise)
   - Trainiert auf 40 Samples
   - Testdaten: `Marius/results/evaluation_lhs_10_test_2D.csv`

## Features

### 1. Automatische Modellvorhersagen
- Lädt alle `.joblib` Modelle aus `Florian/validation/joblibs/`
- Trifft Vorhersagen für vier OPEX-Ziele: MILP, LP lower, LP upper, LP approx
- Behandelt unterschiedliche Datenformate automatisch

### 2. Intelligente Skalierung für 1D-Modelle
- 1D Test-Daten enthalten nur spezifische (normalisierte) OPEX-Werte
- Das Skript inferiert automatisch einen Strompreis-Multiplikator
- Nutzt Archive-Dateien um den korrekten Multiplikator zu finden
- Konvertiert spezifische zu absoluten OPEX-Werten

### 3. R²-Score Berechnung
- Berechnet R² Scores zwischen vorhergesagten und echten Werten
- Speichert detaillierte Ergebnisse pro Modell
- Erstellt Summary-Tabellen für schnelle Übersicht

### 4. Vergleichende Analyse
- Vergleicht 1D vs. 2D Modell-Performance
- Berechnet Durchschnitts-R² für jedes Modelltyp
- Zeigt Differenzen zwischen 1D und 2D

## Output-Struktur

### 1D Model Results: `Florian/validation/results_1d_models/`

**Vorhersage-CSVs** (pro Training-Größe und Target):
```
{training_size}_train_10_test_ratio_opex_{target}.csv
```
Spalten: `y_test`, `y_pred`, `r2`

**Summary-Tabelle**:
```
r2_scores_summary.csv
```
Format:
```
training_size,MILP,LP lower,LP upper,LP approx
2,0.7076,0.6722,0.7773,0.6316
5,0.9172,0.9127,0.9364,0.9087
20,0.9323,0.9308,0.9485,0.9257
40,0.9354,0.9337,0.9507,0.9285
```

### 2D Model Results: `Florian/validation/results_2d_models/`

**Vorhersage-CSVs** (pro Target):
```
40_train_10_test_2d_discrete_opex_{target}.csv
```

**Summary-Tabelle**:
```
r2_scores_summary.csv
```
Format:
```
MILP,LP lower,LP upper,LP approx
0.9639,0.9632,0.9748,0.9574
```

### Comparison Summary: `Florian/validation/model_comparison_summary.csv`

```
,1D (Ratio-based),2D (Discrete prices),Difference (2D - 1D)
MILP,0.8731,0.9639,0.0908
LP lower,0.8623,0.9632,0.1009
LP upper,0.9032,0.9748,0.0715
LP approx,0.8486,0.9574,0.1087
```

## Usage

```bash
cd <repo_root>
python Florian/validation/evaluate_regression_models.py
```

## Anforderungen

- Python 3.7+
- pandas
- numpy
- scikit-learn (für R² Score)
- joblib (für Modell-Laden)

## Technische Details

### 1D-Modelle Evaluation

1. **Test-Daten laden**: `evaluation_lhs_10_test_1D.csv`
2. **Strompreis-Multiplikator inferieren**:
   - Sucht nach Archive-Dateien (`archive/*_ratio_opex_*.csv`)
   - Berechnet: `multiplier = y_test_absolut / y_pred_spezifisch`
   - Verwendet Median-Wert für Robustheit
3. **Vorhersagen treffen**:
   - Extrahiert `ratio` Feature
   - Modell-Output: spezifische OPEX
   - Multipliziert mit inferred multiplier: absolute OPEX
4. **R² berechnen**: `r2_score(y_test_absolut, y_pred_absolut)`

### 2D-Modelle Evaluation

1. **Test-Daten laden**: `evaluation_lhs_10_test_2D.csv`
2. **Features extrahieren**: `gas_price_MWh`, `electricity_price_MWh`
3. **Vorhersagen treffen**: Modell liefert direkt absolute OPEX
4. **R² berechnen**: `r2_score(y_test, y_pred)`

## Key Findings

### 1D vs 2D Performance

| Modell | 1D Average R² | 2D R² | Verbesserung |
|--------|--------------|-------|-------------|
| MILP | 0.8731 | 0.9639 | +9.1% |
| LP lower | 0.8623 | 0.9632 | +10.1% |
| LP upper | 0.9032 | 0.9748 | +7.2% |
| LP approx | 0.8486 | 0.9574 | +10.9% |

**Schlussfolgerung**: 2D Modelle mit diskreten Preisen sind im Durchschnitt ~9% besser als 1D Ratio-basierte Modelle.

### Training Sample Size Effect (1D)

- **2 Samples**: R² ~0.71 (schlecht)
- **5 Samples**: R² ~0.91 (gut)
- **20 Samples**: R² ~0.93 (sehr gut)
- **40 Samples**: R² ~0.94 (sehr gut)

**Schlussfolgerung**: 20+ Samples erreichen bereits Plateau bei R² ~0.93-0.94

## Error Handling

- **Modell nicht gefunden**: Warnung und Überspringung
- **Archive-Datei nicht gefunden**: Fallback auf default multiplier (100)
- **Unterschiedliche Test-Datenlängen**: Fehler mit detaillierter Meldung

## Erweiterungsmöglichkeiten

1. **Cross-Validierung**: Leave-one-out oder K-Fold CV
2. **Visualisierungen**: Scatter plots, Residual plots
3. **Andere Metriken**: MAE, RMSE, MAPE
4. **Feature Importance**: Für 2D Modelle
5. **Hyperparameter Tuning**: Optimale Trainings-Größe finden

## Notes

- Das Skript ignoriert Warnungen von scikit-learn bezüglich fehlender feature names
- Windows-Kompatibilität: UTF-8 Symbole wurden durch ASCII ersetzt
- Archive-Dateien werden für Multiplikator-Inferenz verwendet (nicht modifiziert)
