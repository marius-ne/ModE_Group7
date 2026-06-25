import joblib
model = joblib.load("Florian/surrogate_models/surrogate_model_MILP_training.joblib")
sample = [[50, 100]]  # Beispiel: Gaspreis = 50 €/MWh, Strompreis = 100 €/MWh
preds = model.predict(sample)
print(preds[0])
