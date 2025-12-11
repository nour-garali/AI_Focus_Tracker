# app_streamlit.py
import streamlit as st
import numpy as np
import mediapipe as mp
import math
import time
import os
import sys
from collections import deque
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# -----------------------------
# CONFIGURATION POUR STREAMLIT CLOUD
# -----------------------------

# CORRECTION : Importer OpenCV différemment pour Streamlit Cloud
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError as e:
    st.error(f"OpenCV import error: {e}")
    # Créer une simulation de cv2 pour le mode test
    class MockCV2:
        CAP_PROP_FRAME_WIDTH = 3
        CAP_PROP_FRAME_HEIGHT = 4
        CAP_PROP_FPS = 5
        FONT_HERSHEY_SIMPLEX = 0
        COLOR_BGR2RGB = 4
        
        @staticmethod
        def VideoCapture(*args):
            return None
            
        @staticmethod
        def cvtColor(img, code):
            return img
            
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
            return img
            
        @staticmethod
        def ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness):
            return img
            
        @staticmethod
        def circle(img, center, radius, color, thickness):
            return img
    
    cv2 = MockCV2()
    OPENCV_AVAILABLE = False

# -----------------------------
# GESTION DE TENSORFLOW POUR STREAMLIT CLOUD
# -----------------------------
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.losses import MeanSquaredError
    TENSORFLOW_AVAILABLE = True
except ImportError:
    st.warning("TensorFlow non disponible - Mode sans modèle activé")
    TENSORFLOW_AVAILABLE = False

# -----------------------------
# CONFIGURATION
# -----------------------------
MODEL_PATH = "best_gaze_model.h5"
DURATION_SECONDS = 30
FPS_TARGET = 15
WINDOW_SEC = 3
EAR_THRESHOLD = 0.22
EYE_CLOSED_CONSEC_FRAMES = 3
STABILITY_MOVEMENT_THRESH = 25
PRIVACY_BLUR = False  # Désactivé pour Streamlit Cloud
CALIBRATION_FRAMES = 10  # Réduit pour Streamlit Cloud
DASHBOARD_UPDATE_INTERVAL = 0.5

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

DEBUG = False

# -----------------------------
# UTILITAIRES
# -----------------------------
def euclidean(a, b):
    return math.dist(a, b)

def eye_aspect_ratio(landmarks, eye_idx, w, h):
    try:
        pts = [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in eye_idx]
        A = euclidean(pts[1], pts[5])
        B = euclidean(pts[2], pts[4])
        C = euclidean(pts[0], pts[3])
        return (A + B) / (2.0 * C) if C != 0 else 0.0
    except Exception:
        return 0.0

def angle_between_eyes(landmarks, left_idx, right_idx, w, h):
    try:
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
# LOAD MODEL
# -----------------------------
def load_gaze_model(path):
    if not TENSORFLOW_AVAILABLE:
        st.warning("TensorFlow non disponible - Fonctionnalité modèle désactivée")
        return None
    
    try:
        if not os.path.exists(path):
            st.warning(f"⚠️ Fichier modèle non trouvé : {path}")
            # Liste des fichiers pour débogage
            try:
                files = os.listdir(".")
                st.info(f"Fichiers disponibles: {files[:10]}")
            except:
                pass
            return None
        
        model_local = load_model(path, custom_objects={'mse': MeanSquaredError()})
        st.success("✅ Modèle gaze chargé.")
        return model_local
    except Exception as e:
        st.warning(f"❌ Erreur chargement modèle : {str(e)[:100]}... Modèle désactivé.")
        return None

model = load_gaze_model(MODEL_PATH)
model_enabled = True if (model is not None and TENSORFLOW_AVAILABLE) else False

# -----------------------------
# CAMERA & MEDIAPIPE
# -----------------------------
def initialize_camera():
    """Initialise la caméra pour Streamlit Cloud"""
    if not OPENCV_AVAILABLE:
        st.warning("OpenCV non disponible - Mode test forcé")
        return create_test_camera()
    
    # CORRECTION POUR STREAMLIT CLOUD : Utiliser une caméra de test
    # Sur Streamlit Cloud, l'accès à la caméra n'est pas autorisé
    st.info("🔍 Streamlit Cloud détecté - Mode démonstration activé")
    return create_test_camera()

def create_test_camera():
    """Crée une caméra de test pour Streamlit Cloud"""
    class TestCamera:
        def __init__(self):
            self.width = 640
            self.height = 480
            self.frame_count = 0
            self.face_position = [self.width//2, self.height//2]
            self.face_speed = [2, 1]
            
        def read(self):
            self.frame_count += 1
            # Créer une image de test avec un visage animé
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            
            # Animation du visage
            self.face_position[0] += self.face_speed[0]
            self.face_position[1] += self.face_speed[1]
            
            # Rebond sur les bords
            if self.face_position[0] < 100 or self.face_position[0] > self.width - 100:
                self.face_speed[0] *= -1
            if self.face_position[1] < 100 or self.face_position[1] > self.height - 100:
                self.face_speed[1] *= -1
                
            center_x, center_y = self.face_position
            
            # Dessiner un visage
            # Tête
            cv2.ellipse(frame, (center_x, center_y), (100, 120), 0, 0, 360, (100, 100, 255), -1)
            
            # Yeux
            eye_offset = 40 + 10 * math.sin(self.frame_count * 0.1)
            cv2.circle(frame, (center_x - int(eye_offset), center_y - 30), 20, (255, 255, 255), -1)
            cv2.circle(frame, (center_x + int(eye_offset), center_y - 30), 20, (255, 255, 255), -1)
            
            # Pupilles (qui bougent)
            pupil_offset = 5 * math.sin(self.frame_count * 0.2)
            cv2.circle(frame, (center_x - int(eye_offset + pupil_offset), center_y - 30), 8, (0, 0, 0), -1)
            cv2.circle(frame, (center_x + int(eye_offset + pupil_offset), center_y - 30), 8, (0, 0, 0), -1)
            
            # Bouche
            mouth_width = 40 + 10 * math.sin(self.frame_count * 0.15)
            cv2.ellipse(frame, (center_x, center_y + 40), (int(mouth_width), 15), 0, 0, 180, (50, 50, 200), 2)
            
            # Texte informatif
            cv2.putText(frame, "STREAMLIT CLOUD DEMO", (50, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Mode demonstration active", (50, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
            cv2.putText(frame, f"Frame: {self.frame_count}", (50, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            return True, frame
        
        def isOpened(self):
            return True
            
        def get(self, prop):
            if prop == cv2.CAP_PROP_FRAME_WIDTH:
                return self.width
            elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
                return self.height
            elif prop == cv2.CAP_PROP_FPS:
                return 30
            return 0
            
        def release(self):
            pass
    
    return TestCamera()

# Initialiser la caméra
cap = initialize_camera()
is_test_camera = True  # Toujours en mode test sur Streamlit Cloud

# Obtenir les dimensions
try:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
except:
    width, height = 640, 480

# Initialiser MediaPipe
try:
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False, 
        max_num_faces=1,
        refine_landmarks=True, 
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    MEDIAPIPE_AVAILABLE = True
except Exception as e:
    st.error(f"Erreur MediaPipe: {e}")
    MEDIAPIPE_AVAILABLE = False

# -----------------------------
# CALIBRATION TILT
# -----------------------------
def calibrate_tilt(frames=CALIBRATION_FRAMES):
    st.info("🔹 Calibration tilt (simulation)...")
    
    # Simulation pour Streamlit Cloud
    tilt_values = []
    for i in range(frames):
        # Générer des valeurs de tilt réalistes
        base_tilt = 0.0
        noise = np.random.normal(0, 2)  # Bruit aléatoire
        tilt_values.append(base_tilt + noise)
    
    center = float(np.mean(tilt_values)) if tilt_values else 0.0
    st.success(f"✅ Calibration terminée. Tilt_center={center:.2f} (simulé)")
    return center

tilt_center = calibrate_tilt()

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
    fig.update_layout(height=600, width=800, paper_bgcolor='#2b3e5c', plot_bgcolor='#2b3e5c',
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
# MAIN LOOP STREAMLIT
# -----------------------------
def main_loop(fig_dashboard=None, st_plot=None, st_frame=None, st_feedback=None):
    global model_enabled, is_test_camera
    
    fps_interval = 1.0 / FPS_TARGET
    gaze_queue = deque(maxlen=int(WINDOW_SEC * FPS_TARGET))
    center_queue = deque(maxlen=int(WINDOW_SEC * FPS_TARGET))
    consecutive_eye_closed = 0
    counters = {
        "center_gaze": 0, "left": 0, "right": 0,
        "eye_closed": 0, "head_tilt": 0, "unstable": 0, 
        "total": 0, "no_face": 0
    }
    
    ear_history = deque(maxlen=5)
    last_dashboard_update_local = 0.0
    
    # Simulation de données pour Streamlit Cloud
    simulation_counter = 0

    while st.session_state.running:
        loop_t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            continue
            
        counters["total"] += 1
        simulation_counter += 1
        
        # Copie pour affichage
        frame_display = frame.copy()
        
        # Traitement MediaPipe simulé pour Streamlit Cloud
        feedback_msgs = []
        
        # Simulation des résultats MediaPipe
        if simulation_counter % 30 > 2:  # 90% du temps, visage détecté
            # Visage "détecté" en mode simulation
            counters["no_face"] = max(0, counters["no_face"] - 1)
            
            # Simulation des métriques
            eye_closed_val = 10 + 5 * math.sin(simulation_counter * 0.05)
            face_detected_val = 95 + 3 * math.sin(simulation_counter * 0.03)
            unstable_val = 60 + 20 * math.sin(simulation_counter * 0.02)
            
            # Simulation du focus
            focus = 75 + 15 * math.sin(simulation_counter * 0.04)
            focus = max(0, min(100, focus))
            
            # Simulation de la direction du regard
            gaze_sin = math.sin(simulation_counter * 0.1)
            if gaze_sin > 0.3:
                gaze = "RIGHT"
                counters["right"] += 1
                pred = 0.7
            elif gaze_sin < -0.3:
                gaze = "LEFT"
                counters["left"] += 1
                pred = -0.7
            else:
                gaze = "CENTER"
                counters["center_gaze"] += 1
                pred = 0.1
                
            gaze_queue.append(pred)
            
            # Simulation yeux fermés
            if simulation_counter % 100 > 90:  # 10% du temps yeux fermés
                counters["eye_closed"] += 1
                feedback_msgs.append("Eyes Closed")
                
            # Simulation stabilité
            if simulation_counter % 150 > 140:  # Instable 10% du temps
                feedback_msgs.append("Too stable")
                
        else:
            # Pas de visage détecté
            counters["no_face"] += 1
            counters["eye_closed"] += 1
            eye_closed_val = 80
            face_detected_val = 30
            unstable_val = 0
            focus = 20
            gaze = "NONE"
            feedback_msgs.append("No face detected")
            
        # Mise à jour dashboard
        if time.time() - last_dashboard_update_local > DASHBOARD_UPDATE_INTERVAL:
            update_dashboard(fig_dashboard, 
                           round(focus, 2), 
                           round(eye_closed_val, 2), 
                           round(face_detected_val, 2), 
                           round(unstable_val, 2))
            last_dashboard_update_local = time.time()
            
        # Affichage dashboard
        st_plot.plotly_chart(fig_dashboard, use_container_width=True)
        
        # Ajouter du texte à la frame
        cv2.putText(frame_display, f"Gaze: {gaze}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame_display, f"Focus: {focus:.1f}%", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Ajouter les messages de feedback
        for idx, msg in enumerate(feedback_msgs):
            cv2.putText(frame_display, msg, (10, 90 + 30 * idx), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Indication mode démo
        cv2.putText(frame_display, "STREAMLIT CLOUD DEMO", (width - 250, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Afficher la frame
        st_frame.image(frame_display, channels="BGR", use_container_width=True)
        
        # Afficher feedback
        feedback_text = " | ".join(feedback_msgs) if feedback_msgs else "Analyse en cours..."
        st_feedback.text(feedback_text)
        
        # Contrôle FPS
        t_elapsed = time.time() - loop_t0
        if t_elapsed < fps_interval:
            time.sleep(max(0, fps_interval - t_elapsed))

# -----------------------------
# INTERFACE STREAMLIT
# -----------------------------
def main():
    st.set_page_config(
        page_title="AI Focus Tracker",
        page_icon="👁️",
        layout="wide"
    )
    
    st.title("👁️ AI Focus Tracker - Streamlit Cloud")
    
    # Informations système
    with st.sidebar:
        st.header("Informations système")
        st.write(f"📦 OpenCV: {'✅' if OPENCV_AVAILABLE else '❌'}")
        st.write(f"🧠 TensorFlow: {'✅' if TENSORFLOW_AVAILABLE else '❌'}")
        st.write(f"🤖 MediaPipe: {'✅' if MEDIAPIPE_AVAILABLE else '❌'}")
        st.write(f"📊 Modèle: {'✅ Chargé' if model_enabled else '❌ Non disponible'}")
        st.write(f"🎥 Mode: {'🔄 Démonstration' if is_test_camera else '📷 Caméra réelle'}")
        
        st.divider()
        
        st.info("""
        **Note Streamlit Cloud:**
        - Mode démonstration activé
        - Pas d'accès à la webcam
        - Données simulées pour la démo
        """)
    
    # Initialisation session state
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'fig_dashboard' not in st.session_state:
        st.session_state.fig_dashboard = make_dashboard()
    
    # Contrôles
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("▶️ Démarrer l'analyse", type="primary", use_container_width=True):
            st.session_state.running = True
            st.rerun()
    
    with col2:
        if st.button("⏹️ Arrêter", type="secondary", use_container_width=True):
            st.session_state.running = False
            st.rerun()
    
    # Statut
    status_col1, status_col2 = st.columns([3, 1])
    with status_col1:
        if st.session_state.running:
            st.success("🟢 Analyse en cours - Mode démonstration")
        else:
            st.warning("🔴 Analyse arrêtée")
    
    with status_col2:
        if st.button("🔄 Redémarrer calibration"):
            global tilt_center
            tilt_center = calibrate_tilt()
            st.rerun()
    
    # Placeholders
    st_plot = st.empty()
    st_frame = st.empty()
    st_feedback = st.empty()
    
    # Afficher dashboard initial
    st_plot.plotly_chart(st.session_state.fig_dashboard, use_container_width=True)
    
    # Démarrer ou arrêter l'analyse
    if st.session_state.running:
        main_loop(
            fig_dashboard=st.session_state.fig_dashboard,
            st_plot=st_plot,
            st_frame=st_frame,
            st_feedback=st_feedback
        )
    else:
        # Écran d'arrêt
        dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, "ANALYSE ARRÊTÉE", 
                   (width//2 - 150, height//2 - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 200, 200), 3)
        cv2.putText(dummy_frame, "Cliquez sur 'Démarrer' pour commencer", 
                   (width//2 - 200, height//2 + 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (150, 150, 255), 2)
        cv2.putText(dummy_frame, "Mode: Streamlit Cloud - Données simulées", 
                   (width//2 - 200, height//2 + 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 1)
        
        st_frame.image(dummy_frame, channels="BGR", use_container_width=True)
        st_feedback.info("Prêt à démarrer l'analyse de concentration")

if __name__ == "__main__":
    main()