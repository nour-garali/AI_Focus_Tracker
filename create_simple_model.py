# create_simple_model.py
import tensorflow as tf
import numpy as np
import os

print("🔄 Création modèle SANS Conv2D...")

# Supprime l'ancien
if os.path.exists("best_gaze_model.keras"):
    os.remove("best_gaze_model.keras")

# Modèle SANS Conv2D (pour éviter les erreurs de poids)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(64, 64, 3)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='tanh')
])

# Compile SIMPLEMENT
model.compile(optimizer='adam', loss='mse')

# Entraînement minimal
x = np.random.randn(10, 64, 64, 3).astype(np.float32)
y = np.random.randn(10, 1).astype(np.float32)
model.fit(x, y, epochs=2, verbose=0)

# Sauvegarde
model.save('best_gaze_model.keras')

print("✅ Modèle SIMPLE créé")
print(f"📏 Taille: {os.path.getsize('best_gaze_model.keras') / 1024:.1f} KB")

# Test
test = tf.keras.models.load_model('best_gaze_model.keras')
print(f"✅ Test chargement: {test.predict(x[:1])[0][0]:.4f}")