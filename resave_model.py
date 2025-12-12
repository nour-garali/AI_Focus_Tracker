# test_simple.py - PAS DE PANDAS, juste tester TensorFlow
import os
print("🧪 Test minimal de TensorFlow...")

try:
    # Tester si TensorFlow fonctionne
    import tensorflow as tf
    print(f"✅ TensorFlow {tf.__version__}")
    
    # Tester NumPy
    import numpy as np
    print(f"✅ NumPy {np.__version__}")
    
    # Tester si un modèle existe
    model_files = [f for f in os.listdir() if f.endswith('.h5')]
    if model_files:
        print(f"📁 Modèles trouvés: {model_files}")
        
        # Essayer de charger le premier
        model_path = model_files[0]
        print(f"🔄 Chargement de {model_path}...")
        
        try:
            model = tf.keras.models.load_model(model_path, compile=False)
            print("✅ Modèle chargé avec succès!")
            
            # Test rapide
            test_input = np.random.randn(1, 64, 64, 3).astype('float32')
            prediction = model.predict(test_input, verbose=0)
            print(f"📊 Prédiction test: {prediction[0][0]:.4f}")
            
        except Exception as e:
            print(f"❌ Erreur de chargement: {type(e).__name__}")
            
    else:
        print("⚠️ Aucun fichier .h5 trouvé")
        
except ImportError as e:
    print(f"❌ Import impossible: {e}")
    print("Recréez l'environnement avec les commandes ci-dessus.")