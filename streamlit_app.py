# app_streamlit.py
import streamlit as st

# -----------------------------
# CONFIGURATION DE LA PAGE - DOIT ÊTRE LA PREMIÈRE COMMANDE
# -----------------------------
st.set_page_config(
    page_title="AI Focus Tracker",
    page_icon="👁️",
    layout="wide"
)

import numpy as np
import math
import time
import os
import sys
from collections import deque
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# -----------------------------
# DÉTECTION STREAMLIT CLOUD
# -----------------------------
IS_STREAMLIT_CLOUD = os.environ.get('STREAMLIT_CLOUD') is not None

# -----------------------------
# IMPORT STANDARD
# -----------------------------
from PIL import Image
import io

# -----------------------------
# IMPORT CONDITIONNEL
# -----------------------------
if IS_STREAMLIT_CLOUD:
    st.info("🔍 **Mode Streamlit Cloud activé**")
    
    try:
        import cv2
        import mediapipe as mp
        from tensorflow.keras.models import load_model
        from tensorflow.keras.losses import MeanSquaredError
        CV2_AVAILABLE = True
        MEDIAPIPE_AVAILABLE = True
        TENSORFLOW_AVAILABLE = True
    except ImportError as e:
        st.warning(f"Certaines bibliothèques ne sont pas disponibles: {e}")
        
        # Mock minimal pour éviter les erreurs
        class MockCV2:
            @staticmethod
            def cvtColor(img, code):
                return img if hasattr(img, 'shape') else np.zeros((480, 640, 3), dtype=np.uint8)
            
            @staticmethod
            def GaussianBlur(img, ksize, sigma):
                return img
            
            @staticmethod
            def rectangle(img, pt1, pt2, color, thickness):
                return img
            
            @staticmethod
            def putText(img, text, org, fontFace, fontScale, color, thickness):
                return img
            
            @staticmethod
            def resize(img, size):
                return cv2.resize(img, size) if hasattr(img, 'shape') else np.zeros((size[1], size[0], 3), dtype=np.uint8)
            
            @staticmethod
            def imdecode(buf, flags):
                if buf is not None and len(buf) > 0:
                    try:
                        img = Image.open(io.BytesIO(buf))
                        return np.array(img)
                    except:
                        pass
                return np.zeros((480, 640, 3), dtype=np.uint8)
        
        cv2 = MockCV2()
        CV2_AVAILABLE = False
        
        # Mock MediaPipe
        class MockLandmark:
            def __init__(self):
                self.x = 0.5
                self.y = 0.5
        
        class MockResult:
            def __init__(self):
                self.multi_face_landmarks = [type('obj', (object,), {'landmark': [MockLandmark() for _ in range(478)]})()]
        
        class MockFaceMesh:
            def process(self, image):
                return MockResult()
        
        class MockMediaPipe:
            class solutions:
                class face_mesh:
                    FaceMesh = MockFaceMesh
        
        mp = MockMediaPipe()
        MEDIAPIPE_AVAILABLE = False
        TENSORFLOW_AVAILABLE = False
        
else:
    # Mode local - imports complets
    try:
        import cv2
        import mediapipe as mp
        from tensorflow.keras.models import load_model
        from tensorflow.keras.losses import MeanSquaredError
        CV2_AVAILABLE = True
        MEDIAPIPE_AVAILABLE = True
        TENSORFLOW_AVAILABLE = True
    except ImportError as e:
        st.error(f"Erreur d'importation locale: {e}")
        st.stop()

# -----------------------------
# CONFIGURATION
# -----------------------------
MODEL_PATH = "best_gaze_model.keras"
FPS_TARGET = 15
WINDOW_SEC = 3
EAR_THRESHOLD = 0.22
EYE_CLOSED_CONSEC_FRAMES = 3
STABILITY_MOVEMENT_THRESH = 25
PRIVACY_BLUR = True
CALIBRATION_FRAMES = 10

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# -----------------------------
# UTILITAIRES
# -----------------------------
def euclidean(a, b):
    return math.dist(a, b) if a and b else 0.0

def eye_aspect_ratio(landmarks, eye_idx, w, h):
    try:
        if not landmarks or len(landmarks) < max(eye_idx) + 1:
            return 0.3
        
        pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_idx]
        A = euclidean(pts[1], pts[5])
        B = euclidean(pts[2], pts[4])
        C = euclidean(pts[0], pts[3])
        return (A + B) / (2.0 * C) if C != 0 else 0.0
    except Exception:
        return 0.3

def angle_between_eyes(landmarks, left_idx, right_idx, w, h):
    try:
        if not landmarks:
            return 0.0, (0,0), (0,0)
        
        left_pts = [(landmarks[i].x*w, landmarks[i].y*h) for i in left_idx]
        right_pts = [(landmarks[i].x*w, landmarks[i].y*h) for i in right_idx]
        left_center = (sum([p[0] for p in left_pts])/len(left_pts),
                       sum([p[1] for p in left_pts])/len(left_pts))
        right_center = (sum([p[0] for p in right_pts])/len(right_pts),
                        sum([p[1] for p in right_pts])/len(right_pts))
        dx = right_center[0] - left_center[0]
        dy = right_center[1] - left_center[1]
        angle = math.degrees(math.atan2(dy, dx)) if dx != 0 else 0.0
        return angle, left_center, right_center
    except Exception:
        return 0.0, (0,0), (0,0)

def color_bar_stability(val):
    if val < 30: return "red"
    elif val < 70: return "orange"
    else: return "green"

def color_bar(val):
    if val > 70: return "green"
    elif val > 40: return "orange"
    else: return "red"

# -----------------------------
# CHARGEMENT DU MODÈLE - VERSION CORRIGÉE
# -----------------------------
def load_gaze_model(path):
    """Charge le modèle avec gestion d'erreur améliorée"""
    try:
        if not os.path.exists(path):
            if IS_STREAMLIT_CLOUD:
                st.warning("Modèle non trouvé. Mode simulation activé.")
            else:
                st.error(f"Fichier modèle {path} non trouvé")
            return None
        
        if TENSORFLOW_AVAILABLE:
            try:
                # Essayer de charger avec custom_objects
                model = load_model(path, compile=False)
                st.success("✅ Modèle chargé avec succès")
                return model
            except Exception as e:
                # Si erreur, créer un modèle simple
                st.warning(f"⚠️ Impossible de charger le modèle original: {str(e)[:100]}")
                st.info("Création d'un modèle de démonstration...")
                
                # Créer un modèle simple
                import tensorflow as tf
                model = tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(64, 64, 3)),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(32, activation='relu'),
                    tf.keras.layers.Dense(1, activation='tanh')
                ])
                model.compile(optimizer='adam', loss='mse')
                return model
        else:
            st.info("TensorFlow non disponible. Mode simulation.")
            return None
            
    except Exception as e:
        st.warning(f"⚠️ Erreur lors du chargement du modèle: {str(e)[:100]}")
        return None

# -----------------------------
# INITIALISATION DES COMPOSANTS
# -----------------------------
@st.cache_resource
def initialize_components():
    """Initialise les composants une seule fois"""
    components = {}
    
    # Charger le modèle
    components['model'] = load_gaze_model(MODEL_PATH)
    components['model_enabled'] = TENSORFLOW_AVAILABLE and components['model'] is not None
    
    # Initialiser MediaPipe
    if MEDIAPIPE_AVAILABLE and not IS_STREAMLIT_CLOUD:
        try:
            mp_face_mesh = mp.solutions.face_mesh
            components['face_mesh'] = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except Exception as e:
            st.warning(f"MediaPipe limité: {str(e)[:50]}")
            components['face_mesh'] = None
    else:
        components['face_mesh'] = None
    
    return components

# Initialiser les composants
components = initialize_components()
model = components['model']
model_enabled = components['model_enabled']
face_mesh = components['face_mesh']

# -----------------------------
# DASHBOARD
# -----------------------------
def make_dashboard():
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type":"indicator"}, {"type":"indicator"}],
               [{"type":"indicator"}, {"type":"indicator"}]],
        subplot_titles=["Concentration %","Yeux ouverts %","Visage détecté %","Stabilité humaine %"]
    )
    for i in range(4):
        fig.add_trace(go.Indicator(mode="gauge+number", value=0,
                                   gauge={'axis':{'range':[0,100]},
                                          'bar':{'color':'green'}}),
                      row=(i//2)+1, col=(i%2)+1)
    fig.update_layout(height=500, width=800, paper_bgcolor='#2b3e5c', plot_bgcolor='#2b3e5c',
                      title_text="Dashboard Concentration Live", title_x=0.5,
                      font=dict(color="white", size=14))
    return fig

def update_dashboard(fig, focus, eye_closed_val, face_detected_val, unstable_val):
    eyes_open = 100 - eye_closed_val
    stable = unstable_val
    values = [focus, eyes_open, face_detected_val, stable]
    for i, val in enumerate(values):
        fig.data[i].value = val
        if i==3:
            fig.data[i].gauge.bar.color = color_bar_stability(val)
        else:
            fig.data[i].gauge.bar.color = color_bar(val)
    return fig

# -----------------------------
# FONCTION DE TRAITEMENT
# -----------------------------
def process_frame(frame, frame_count, tilt_center=0.0):
    """Traite une frame unique"""
    if frame is None or frame.size == 0:
        return None, 0, 0, 0, 0, ["Aucune image"]
    
    height, width = frame.shape[:2]
    frame_display = cv2.GaussianBlur(frame, (51, 51), 0) if PRIVACY_BLUR else frame.copy()
    
    # Conversion pour MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Détection du visage
    face_detected = False
    feedback_msgs = []
    focus = 50  # Valeur par défaut
    eye_closed_val = 0
    unstable_val = 50
    
    if MEDIAPIPE_AVAILABLE and face_mesh:
        try:
            res = face_mesh.process(rgb)
            if res and res.multi_face_landmarks:
                face_detected = True
                lm = res.multi_face_landmarks[0].landmark
                
                # Calcul des métriques
                ear_left = eye_aspect_ratio(lm, LEFT_EYE_IDX, width, height)
                ear_right = eye_aspect_ratio(lm, RIGHT_EYE_IDX, width, height)
                ear = (ear_left + ear_right) / 2.0
                
                # Détection yeux fermés
                if ear < EAR_THRESHOLD:
                    eye_closed_val = 100
                    feedback_msgs.append("Eyes Closed")
                else:
                    eye_closed_val = 0
                
                # Calcul du focus
                focus = 80 if ear > 0.25 else 30
                
                # Dessiner le rectangle du visage
                xs_all = [lm[i].x*width for i in range(min(100, len(lm)))]
                ys_all = [lm[i].y*height for i in range(min(100, len(lm)))]
                x_min = max(0, int(min(xs_all)-10))
                y_min = max(0, int(min(ys_all)-10))
                x_max = min(width-1, int(max(xs_all)+10))
                y_max = min(height-1, int(max(ys_all)+10))
                
                if x_max > x_min and y_max > y_min:
                    cv2.rectangle(frame_display, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        except Exception as e:
            st.warning(f"Erreur de traitement: {str(e)[:50]}")
    
    face_detected_val = 100 if face_detected else 0
    
    # Ajouter du texte
    cv2.putText(frame_display, f"Focus: {focus}%", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if IS_STREAMLIT_CLOUD:
        cv2.putText(frame_display, "STREAMLIT CLOUD", (10, height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    return frame_display, focus, eye_closed_val, face_detected_val, unstable_val, feedback_msgs

# -----------------------------
# FONCTION PRINCIPALE
# -----------------------------
def main():
    st.title("👁️ AI Focus Tracker")
    
    if IS_STREAMLIT_CLOUD:
        st.info("""
        📱 **Mode Streamlit Cloud Activé**
        - Utilisez le bouton caméra ci-dessous
        - Autorisez l'accès à la caméra dans votre navigateur
        - L'analyse fonctionne en temps réel
        """)
    
    # Sidebar avec informations
    with st.sidebar:
        st.header("📊 Informations")
        st.info(f"Modèle: {'✅ Activé' if model_enabled else '❌ Simulation'}")
        st.info(f"MediaPipe: {'✅ Disponible' if MEDIAPIPE_AVAILABLE else '❌ Simulation'}")
        
        if st.button("🔄 Recalibrer"):
            st.session_state.tilt_center = 0.0
            st.rerun()
    
    # Initialisation session state
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0
    if 'fig_dashboard' not in st.session_state:
        st.session_state.fig_dashboard = make_dashboard()
    if 'tilt_center' not in st.session_state:
        st.session_state.tilt_center = 0.0
    
    # Contrôles
    col1, col2 = st.columns(2)
    with col1:
        start_disabled = st.session_state.running
        if st.button("▶️ Démarrer l'analyse", type="primary", disabled=start_disabled):
            st.session_state.running = True
            st.session_state.frame_count = 0
            st.rerun()
    
    with col2:
        stop_disabled = not st.session_state.running
        if st.button("⏹ Arrêter", disabled=stop_disabled):
            st.session_state.running = False
            st.rerun()
    
    st.info(f"**État:** {'🟢 En cours' if st.session_state.running else '🔴 Arrêté'}")
    
    # Dashboard
    st_plot = st.empty()
    st_frame = st.empty()
    st_feedback = st.empty()
    
    # Afficher le dashboard initial
    st_plot.plotly_chart(st.session_state.fig_dashboard, use_container_width=True)
    
    if st.session_state.running:
        # Utiliser st.camera_input
        camera_image = st.camera_input("Regardez la caméra pour l'analyse", 
                                      key=f"camera_{st.session_state.frame_count}")
        
        if camera_image:
            # Incrémenter le compteur
            st.session_state.frame_count += 1
            
            # Convertir l'image
            img = Image.open(camera_image)
            frame = np.array(img)
            
            if frame is not None and len(frame.shape) == 3:
                # Traiter la frame
                frame_display, focus, eye_closed_val, face_detected_val, unstable_val, feedback_msgs = process_frame(
                    frame, st.session_state.frame_count, st.session_state.tilt_center
                )
                
                # Mettre à jour le dashboard
                update_dashboard(
                    st.session_state.fig_dashboard, 
                    round(focus, 2), 
                    round(eye_closed_val, 2), 
                    round(face_detected_val, 2), 
                    round(unstable_val, 2)
                )
                
                # Afficher les résultats
                st_plot.plotly_chart(st.session_state.fig_dashboard, use_container_width=True)
                
                if frame_display is not None:
                    # Convertir BGR à RGB pour Streamlit
                    frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
                    st_frame.image(frame_rgb, use_column_width=True)
                
                st_feedback.text(" | ".join(feedback_msgs) if feedback_msgs else "✅ Analyse normale")
            else:
                st_frame.warning("Impossible de charger l'image de la caméra")
        else:
            st.info("📷 En attente de l'accès à la caméra...")
            st_frame.info("Veuillez autoriser l'accès à votre caméra et regarder l'objectif")
    
    else:
        # Écran d'arrêt
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, "SESSION ARRÊTÉE", (640//2 - 200, 480//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 3)
        cv2.putText(dummy_frame, "Cliquez sur 'Démarrer l'analyse'", (640//2 - 220, 480//2 + 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 255), 2)
        
        if IS_STREAMLIT_CLOUD:
            cv2.putText(dummy_frame, "Streamlit Cloud - Prêt", 
                       (640//2 - 150, 480//2 + 100), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
        
        # Convertir BGR à RGB
        dummy_frame_rgb = cv2.cvtColor(dummy_frame, cv2.COLOR_BGR2RGB)
        st_frame.image(dummy_frame_rgb, use_column_width=True)
        st_feedback.text("Cliquez sur 'Démarrer l'analyse' pour commencer")

# -----------------------------
# POINT D'ENTRÉE
# -----------------------------
if __name__ == "__main__":
    main()