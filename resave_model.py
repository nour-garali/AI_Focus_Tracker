from tensorflow.keras.models import load_model

# Charger ton ancien modèle
model_old = load_model("best_gaze_model.h5", compile=False)

# Re-sauvegarder pour compatibilité avec TF récent
model_old.save("best_gaze_model_v2.h5")
print("Modèle sauvegardé en best_gaze_model_v2.h5")
