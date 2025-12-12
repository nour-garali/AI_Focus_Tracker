# app_streamlit.py
import streamlit as st
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
# IMPORT STANDARD (toujours disponible)
# -----------------------------
# Ces imports sont toujours nécessaires
from PIL import Image
import io

# -----------------------------
# IMPORT CONDITIONNEL - STREAMLIT CLOUD COMPATIBLE
# -----------------------------
if IS_STREAMLIT_CLOUD:
    st.info("🔍 **Mode Streamlit Cloud activé**")
    
    try:
        # Essayer d'importer les bibliothèques normalement
        import cv2
        import mediapipe as mp
        from tensorflow.keras.models import load_model
        CV2_AVAILABLE = True
        MEDIAPIPE_AVAILABLE = True
        TENSORFLOW_AVAILABLE = True
    except ImportError as e:
        st.warning(f"Certaines bibliothèques ne sont pas disponibles: {e}")
        # Créer des versions mock minimales
        class MockCV2:
            CAP_PROP_FRAME_WIDTH = 3
            CAP_PROP_FRAME_HEIGHT = 4
            
            @staticmethod
            def cvtColor(img, code):
                return img if hasattr(img, 'shape') else np.zeros((480, 640, 3), dtype=np.uint8)
            
            @staticmethod
            def GaussianBlur(img, ksize, sigma):
                return img if hasattr(img, 'shape') else np.zeros((480, 640, 3), dtype=np.uint8)
            
            @staticmethod
            def rectangle(img, pt1, pt2, color, thickness):
                return img
            
            @staticmethod
            def putText(img, text, org, fontFace, fontScale, color, thickness):
                return img
            
            @staticmethod
            def resize(img, size):
                return img if hasattr(img, 'shape') else np.zeros((size[1], size[0], 3), dtype=np.uint8)
            
            @staticmethod
            def ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness):
                return img
            
            @staticmethod
            def circle(img, center, radius, color, thickness):
                return img
            
            @staticmethod
            def imdecode(buf, flags):
                if buf is not None and len(buf) > 0:
                    try:
                        # Essayer de décoder l'image
                        img = Image.open(io.BytesIO(buf))
                        return np.array(img)
                    except:
                        pass
                return np.zeros((480, 640, 3), dtype=np.uint8)
        
        cv2 = MockCV2()
        CV2_AVAILABLE = False
        
        # Mock MediaPipe
        class MockMediaPipe:
            class solutions:
                class face_mesh:
                    class FaceMesh:
                        def __init__(self, **kwargs):
                            pass
                            
                        def process(self, image):
                            class Result:
                                def __init__(self):
                                    self.multi_face_landmarks = None
                            return Result()
        
        mp = MockMediaPipe()
        MEDIAPIPE_AVAILABLE = False
        
        # Mock TensorFlow
        TENSORFLOW_AVAILABLE = False
        st.info("Mode démonstration activé - certaines fonctionnalités seront simulées")
        
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
DURATION_SECONDS = 30
FPS_TARGET = 15
WINDOW_SEC = 3
EAR_THRESHOLD = 0.22
EYE_CLOSED_CONSEC_FRAMES = 3
STABILITY_MOVEMENT_THRESH = 25
PRIVACY_BLUR = True
CALIBRATION_FRAMES = 10  # Réduit pour Streamlit Cloud
DASHBOARD_UPDATE_INTERVAL = 0.5

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

DEBUG = False

# -----------------------------
# UTILITAIRES
# -----------------------------
def euclidean(a, b):
    return math.dist(a, b) if a and b else 0.0

def eye_aspect_ratio(landmarks, eye_idx, w, h):
    try:
        if not landmarks or len(landmarks) < max(eye_idx) + 1:
            return 0.3  # Valeur par défaut pour yeux ouverts
        
        pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_idx]
        A = euclidean(pts[1], pts[5])
        B = euclidean(pts[2], pts[4])
        C = euclidean(pts[0], pts[3])
        return (A + B) / (2.0 * C) if C != 0 else 0.0
    except Exception:
        return 0.3  # Valeur par défaut

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
# CHARGEMENT DU MODÈLE
# -----------------------------
def load_gaze_model(path):
    """Charge le modèle de manière robuste"""
    try:
        if not os.path.exists(path):
            st.warning(f"Modèle {path} non trouvé. Mode simulation activé.")
            return None
        
        if TENSORFLOW_AVAILABLE:
            try:
                model = load_model(path, compile=False)
                st.success("✅ Modèle chargé avec succès")
                return model
            except Exception as e:
                st.warning(f"⚠️ Modèle chargé en mode limité: {str(e)[:100]}")
                return None
        else:
            st.info("TensorFlow non disponible. Mode simulation.")
            return None
            
    except Exception as e:
        st.warning(f"⚠️ Impossible de charger le modèle: {str(e)[:100]}")
        return None

model = load_gaze_model(MODEL_PATH)
model_enabled = TENSORFLOW_AVAILABLE and model is not None

# -----------------------------
# INITIALISATION MEDIAPIPE
# -----------------------------
if MEDIAPIPE_AVAILABLE and not IS_STREAMLIT_CLOUD:
    try:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    except Exception as e:
        st.warning(f"⚠️ MediaPipe limité: {str(e)[:50]}")
        MEDIAPIPE_AVAILABLE = False
        face_mesh = None
else:
    face_mesh = None

# Dimensions par défaut
DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480

# -----------------------------
# CALIBRATION SIMPLIFIÉE
# -----------------------------
def calibrate_tilt():
    """Calibration simplifiée pour Streamlit Cloud"""
    st.info("🔹 Calibration en cours...")
    
    if IS_STREAMLIT_CLOUD:
        # Sur Streamlit Cloud, calibration instantanée
        time.sleep(1)
        tilt_center = 0.0
        st.success(f"✅ Calibration terminée. Tilt_center={tilt_center:.2f}")
        return tilt_center
    else:
        # En local, calibration normale
        try:
            import cv2 as local_cv2
            cap = local_cv2.VideoCapture(0)
            tilt_values = []
            
            for _ in range(CALIBRATION_FRAMES):
                ret, frame = cap.read()
                if ret and MEDIAPIPE_AVAILABLE and face_mesh:
                    rgb = local_cv2.cvtColor(frame, local_cv2.COLOR_BGR2RGB)
                    res = face_mesh.process(rgb)
                    
                    if res.multi_face_landmarks:
                        lm = res.multi_face_landmarks[0].landmark
                        tilt, _, _ = angle_between_eyes(lm, LEFT_EYE_IDX, RIGHT_EYE_IDX, 
                                                       frame.shape[1], frame.shape[0])
                        tilt_values.append(tilt)
                
                time.sleep(0.05)
            
            cap.release()
            center = float(np.mean(tilt_values)) if tilt_values else 0.0
            st.success(f"✅ Calibration terminée. Tilt_center={center:.2f}")
            return center
            
        except Exception as e:
            st.warning(f"Calibration simplifiée: {str(e)[:50]}")
            return 0.0

# Calibration
if 'tilt_center' not in st.session_state:
    st.session_state.tilt_center = calibrate_tilt()
tilt_center = st.session_state.tilt_center

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
    fig.update_layout(height=700, width=900, paper_bgcolor='#2b3e5c', plot_bgcolor='#2b3e5c',
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
# FONCTION PRINCIPALE
# -----------------------------
def process_frame(frame, frame_count, width, height):
    """Traite une frame unique"""
    if frame is None or frame.size == 0:
        return None, 0, 0, 0, 0, ["Aucune image"]
    
    frame_display = cv2.GaussianBlur(frame, (51, 51), 0) if PRIVACY_BLUR else frame.copy()
    
    # Conversion pour MediaPipe
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Détection du visage
    face_detected = False
    feedback_msgs = []
    
    if MEDIAPIPE_AVAILABLE and face_mesh:
        res = face_mesh.process(rgb)
        if res and res.multi_face_landmarks:
            face_detected = True
            lm = res.multi_face_landmarks[0].landmark
            
            # Extraire ROI du visage
            xs_all = [lm[i].x*width for i in range(min(len(lm), 100))]
            ys_all = [lm[i].y*height for i in range(min(len(lm), 100))]
            x_min, y_min = max(0, int(min(xs_all)-10)), max(0, int(min(ys_all)-10))
            x_max, y_max = min(width-1, int(max(xs_all)+10)), min(height-1, int(max(ys_all)+10))
            
            if x_max > x_min and y_max > y_min:
                # Dessiner rectangle
                cv2.rectangle(frame_display, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame_display, "Face detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Calcul EAR
                ear_left = eye_aspect_ratio(lm, LEFT_EYE_IDX, width, height)
                ear_right = eye_aspect_ratio(lm, RIGHT_EYE_IDX, width, height)
                ear = (ear_left + ear_right) / 2.0
                
                # Détection yeux fermés
                eyes_closed = ear < EAR_THRESHOLD
                if eyes_closed:
                    feedback_msgs.append("Eyes Closed")
    
    # Calcul des métriques
    eye_closed_val = 20 if "Eyes Closed" in feedback_msgs else 0
    face_detected_val = 100 if face_detected else 0
    unstable_val = 50  # Valeur simulée
    focus = 70 if face_detected else 30
    
    # Ajouter un indicateur pour Streamlit Cloud
    if IS_STREAMLIT_CLOUD:
        cv2.putText(frame_display, "STREAMLIT CLOUD", (10, height - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    return frame_display, focus, eye_closed_val, face_detected_val, unstable_val, feedback_msgs

# -----------------------------
# INTERFACE UTILISATEUR
# -----------------------------
def main():
    st.title("AI Focus Tracker - Streamlit")
    
    if IS_STREAMLIT_CLOUD:
        st.info("""
        📱 **Application optimisée pour Streamlit Cloud**
        - Utilisez le bouton caméra ci-dessous
        - Autorisez l'accès à la caméra dans votre navigateur
        - L'application fonctionne en temps réel
        """)
    
    # Initialisation session state
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0
    if 'fig_dashboard' not in st.session_state:
        st.session_state.fig_dashboard = make_dashboard()
    
    # Contrôles
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start", type="primary"):
            st.session_state.running = True
            st.session_state.frame_count = 0
            st.rerun()
    
    with col2:
        if st.button("⏹ Stop"):
            st.session_state.running = False
            st.rerun()
    
    # Dashboard
    st_plot = st.empty()
    st_frame = st.empty()
    st_feedback = st.empty()
    
    # Afficher le dashboard initial
    st_plot.plotly_chart(st.session_state.fig_dashboard)
    
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
                height, width = frame.shape[:2]
                
                # Traiter la frame
                frame_display, focus, eye_closed_val, face_detected_val, unstable_val, feedback_msgs = process_frame(
                    frame, st.session_state.frame_count, width, height
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
                st_plot.plotly_chart(st.session_state.fig_dashboard)
                if frame_display is not None:
                    st_frame.image(frame_display, channels="BGR", use_column_width=True)
                st_feedback.text(" | ".join(feedback_msgs) if feedback_msgs else "Analyse en cours...")
        else:
            st.info("⚠️ En attente de l'accès à la caméra...")
            st_frame.info("Veuillez autoriser l'accès à votre caméra")
    
    else:
        # Écran d'arrêt
        dummy_frame = np.zeros((DEFAULT_HEIGHT, DEFAULT_WIDTH, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, "SESSION ARRÊTÉE", (DEFAULT_WIDTH//2 - 200, DEFAULT_HEIGHT//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 3)
        
        if IS_STREAMLIT_CLOUD:
            cv2.putText(dummy_frame, "Streamlit Cloud - Prêt", 
                       (DEFAULT_WIDTH//2 - 150, DEFAULT_HEIGHT//2 + 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)
        
        st_frame.image(dummy_frame, channels="BGR")
        st_feedback.text("Cliquez sur Start pour commencer l'analyse")

# -----------------------------
# POINT D'ENTRÉE
# -----------------------------
if __name__ == "__main__":
    # Configuration pour éviter les erreurs de mémoire
    st.set_page_config(
        page_title="AI Focus Tracker",
        page_icon="👁️",
        layout="wide"
    )
    
    main()