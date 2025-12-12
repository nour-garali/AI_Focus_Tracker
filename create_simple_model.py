# create_simple_model_final.py
import tensorflow as tf
import numpy as np
import os

print("🔄 Création modèle FINAL sans Conv2D...")

# Supprime l'ancien
if os.path.exists("best_gaze_model.keras"):
    os.remove("best_gaze_model.keras")

# Modèle SANS Conv2D (100% compatible)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(64, 64, 3), name='input_layer'),
    tf.keras.layers.Flatten(name='flatten'),
    tf.keras.layers.Dense(64, activation='relu', name='dense1'),
    tf.keras.layers.Dense(32, activation='relu', name='dense2'),
    tf.keras.layers.Dense(1, activation='tanh', name='output')
])

# Compile avec des paramètres simples
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# Entraînement minimal avec données variées
x = np.random.randn(100, 64, 64, 3).astype(np.float32)
y = np.random.randn(100, 1).astype(np.float32) * 0.5  # Valeurs entre -0.5 et 0.5

model.fit(x, y, epochs=3, verbose=0, batch_size=16)

# Sauvegarde
model.save('best_gaze_model.keras')

print("✅ Modèle FINAL créé: best_gaze_model.keras")
print(f"📏 Taille: {os.path.getsize('best_gaze_model.keras') / 1024:.1f} KB")

# Test complet
test_model = tf.keras.models.load_model('best_gaze_model.keras')
prediction = test_model.predict(x[:1], verbose=0)
print(f"🎯 Test prédiction: {prediction[0][0]:.4f}")
print(f"✅ Architecture: {[layer.name for layer in test_model.layers]}")