# test_keras_model.py
import os
import tensorflow as tf
import numpy as np

print("🧪 TEST du modèle .keras")

# 1. Test avec .keras (moderne)
print("\n1. TEST FORMAT .KERAS")
keras_files = [f for f in os.listdir() if f.endswith('.keras')]
print(f"   Fichiers .keras trouvés: {keras_files}")

if 'best_gaze_model.keras' in keras_files:
    print("   ✅ best_gaze_model.keras existe")
    
    try:
        # Chargement du modèle .keras
        model = tf.keras.models.load_model('best_gaze_model.keras')
        print("   ✅ Modèle .keras chargé avec succès!")
        
        # Vérification
        print(f"   📊 Architecture:")
        print(f"     - Couches: {len(model.layers)}")
        print(f"     - Entrée: {model.input_shape}")
        print(f"     - Sortie: {model.output_shape}")
        
        # Test prédiction
        test_input = np.random.randn(1, 64, 64, 3).astype(np.float32)
        prediction = model.predict(test_input, verbose=0)
        print(f"   🎯 Prédiction test: {prediction[0][0]:.4f}")
        
    except Exception as e:
        print(f"   ❌ Erreur: {type(e).__name__}")
        print(f"      Message: {str(e)[:100]}")
else:
    print("   ⚠️  Aucun fichier .keras trouvé")

# 2. Test avec .h5 (ancien format)
print("\n2. TEST FORMAT .H5")
h5_files = [f for f in os.listdir() if f.endswith('.h5')]
print(f"   Fichiers .h5 trouvés: {h5_files}")

if 'best_gaze_model.h5' in h5_files:
    print("   ⚠️  best_gaze_model.h5 existe (format ancien)")
    
    try:
        model_h5 = tf.keras.models.load_model('best_gaze_model.h5')
        print("   ✅ Modèle .h5 chargé avec succès!")
    except Exception as e:
        print(f"   ❌ Erreur .h5: {type(e).__name__}")
        print(f"      Message: {str(e)[:100]}")
else:
    print("   ℹ️  Aucun fichier .h5 trouvé")

# 3. Création d'un modèle .keras si besoin
print("\n3. CRÉATION MODÈLE .KERAS (si nécessaire)")
if 'best_gaze_model.keras' not in os.listdir():
    print("   Création d'un nouveau modèle .keras...")
    
    try:
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(64, 64, 3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation='tanh')
        ])
        model.compile(optimizer='adam', loss='mse')
        model.save('best_gaze_model.keras')
        print("   ✅ Modèle .keras créé!")
    except Exception as e:
        print(f"   ❌ Erreur création: {e}")
else:
    print("   ℹ️  Modèle .keras existe déjà")

# 4. Vérification finale
print("\n4. VÉRIFICATION FINALE")
files = os.listdir()
model_files = [f for f in files if 'best_gaze' in f]
print(f"   Fichiers de modèle présents: {model_files}")

for file in model_files:
    size = os.path.getsize(file) / 1024
    print(f"   - {file}: {size:.1f} KB")