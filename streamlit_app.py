# app_streamlit.py - VERSION CORRIGÉE
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
# IMPORTATION AVEC GESTION D'ERREUR
# -----------------------------
try:
    # streamlit-webrtc version compatible
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration
    import av
    WEBRTC_AVAILABLE = True
except ImportError as e:
    WEBRTC_AVAILABLE = False
    st.error(f"Erreur d'importation de streamlit-webrtc: {e}")
    st.info("Assurez-vous que 'streamlit-webrtc' est dans requirements.txt")

# Mode démo si WebRTC n'est pas disponible
DEMO_MODE = not WEBRTC_AVAILABLE or os.environ.get('STREAMLIT_CLOUD') is not None

# -----------------------------
# CHARGEMENT DES AUTRES BIBLIOTHÈQUES
# -----------------------------
if DEMO_MODE:
    st.warning("🔍 Mode démonstration activé")
    
    # Simulation minimale pour OpenCV
    class MockCV2:
        FONT_HERSHEY_SIMPLEX = 0
        
        @staticmethod
        def putText(img, text, org, fontFace, fontScale, color, thickness):
            return img
            
        @staticmethod
        def rectangle(img, pt1, pt2, color, thickness):
            return img
            
        @staticmethod
        def circle(img, center, radius, color, thickness):
            return img
            
        @staticmethod
        def ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness):
            return img
            
        @staticmethod
        def GaussianBlur(img, ksize, sigma):
            return img
            
        @staticmethod
        def cvtColor(img, code):
            return img
    
    cv2 = MockCV2()
else:
    try:
        import cv2
        import mediapipe as mp
        from tensorflow.keras.models import load_model
        from tensorflow.keras.losses import MeanSquaredError
    except ImportError as e:
        st.error(f"Erreur d'importation: {e}")
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
CALIBRATION_FRAMES = 50
DASHBOARD_UPDATE_INTERVAL = 0.5

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# -----------------------------
# FONCTIONS UTILITAIRES
# -----------------------------
def euclidean(a, b):
    return math.dist(a, b) if hasattr(math, 'dist') else np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

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

def get_eye_open_values(lm, width, height):
    """Calcule les valeurs eye_open_left et eye_open_right"""
    try:
        left_upper = (lm[159].x*width, lm[159].y*height)
        left_lower = (lm[145].x*width, lm[145].y*height)
        right_upper = (lm[386].x*width, lm[386].y*height)
        right_lower = (lm[374].x*width, lm[374].y*height)
        
        eye_open_left = euclidean(left_upper, left_lower)
        eye_open_right = euclidean(right_upper, right_lower)
        
        return eye_open_left, eye_open_right
    except:
        return 10.0, 10.0

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
    fig.update_layout(height=500, width=700, paper_bgcolor='#2b3e5c', plot_bgcolor='#2b3e5c',
                      title_text="Dashboard Concentration Live", title_x=0.5,
                      font=dict(color="white", size=12))
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
# MODE DÉMONSTRATION (pour Streamlit Cloud)
# -----------------------------
def run_demo_mode():
    """Exécute le mode démonstration"""
    st.title("AI Focus Tracker - Mode Démonstration")
    
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'dashboard' not in st.session_state:
        st.session_state.dashboard = make_dashboard()
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0
    
    # Contrôles
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Démarrer", type="primary"):
            st.session_state.running = True
    with col2:
        if st.button("⏹ Arrêter"):
            st.session_state.running = False
    
    # Dashboard
    st.plotly_chart(st.session_state.dashboard)
    
    # Zone vidéo
    video_placeholder = st.empty()
    status_placeholder = st.empty()
    
    if st.session_state.running:
        # Simulation
        for i in range(30):  # 30 secondes de simulation
            if not st.session_state.running:
                break
            
            st.session_state.frame_count += 1
            
            # Générer une frame simulée
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Animation
            offset = int(20 * math.sin(st.session_state.frame_count * 0.1))
            eye_offset = int(10 * math.sin(st.session_state.frame_count * 0.2))
            
            # Dessiner un visage
            cv2.ellipse(frame, (320, 240), (100 + offset, 130), 
                       0, 0, 360, (100, 100, 200), -1)
            
            # Yeux
            cv2.circle(frame, (280 + eye_offset, 210), 20, (255, 255, 255), -1)
            cv2.circle(frame, (360 + eye_offset, 210), 20, (255, 255, 255), -1)
            
            # Texte
            cv2.putText(frame, "MODE DÉMONSTRATION", (50, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Frame: {st.session_state.frame_count}", (50, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            # Métriques simulées
            focus = 50 + 30 * math.sin(st.session_state.frame_count * 0.05)
            eyes_open = 70 + 20 * math.sin(st.session_state.frame_count * 0.07)
            face_detected = 85 + 10 * math.sin(st.session_state.frame_count * 0.03)
            stability = 60 + 25 * math.sin(st.session_state.frame_count * 0.09)
            
            # Mettre à jour le dashboard
            update_dashboard(st.session_state.dashboard, 
                            round(focus, 2), 
                            round(100 - eyes_open, 2), 
                            round(face_detected, 2), 
                            round(stability, 2))
            
            # Afficher
            video_placeholder.image(frame, channels="BGR")
            status_placeholder.text(f"Focus: {focus:.1f}% | Yeux: {eyes_open:.1f}%")
            
            time.sleep(0.1)  # 10 FPS
        
        status_placeholder.success("✅ Analyse terminée !")
    else:
        # Écran d'attente
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(frame, "PRÊT À DÉMARRER", (150, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
        video_placeholder.image(frame, channels="BGR")
        status_placeholder.info("Cliquez sur 'Démarrer' pour commencer")

# -----------------------------
# MODE AVEC WEBCAM (local)
# -----------------------------
def run_webcam_mode():
    """Exécute le mode avec webcam réelle"""
    st.title("AI Focus Tracker - Mode Local avec Webcam")
    
    # Configuration WebRTC
    rtc_configuration = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    
    # Classe de traitement vidéo
    class VideoProcessor(VideoProcessorBase):
        def __init__(self):
            super().__init__()
            self.frame_count = 0
            self.counters = {
                "total": 0, "no_face": 0, "eye_closed": 0,
                "center_gaze": 0, "left": 0, "right": 0
            }
            
        def recv(self, frame):
            self.frame_count += 1
            self.counters["total"] += 1
            
            # Convertir la frame
            img = frame.to_ndarray(format="bgr24")
            
            # Simulation simple (remplacez par votre vraie logique)
            # Ici vous mettriez votre code MediaPipe, TensorFlow, etc.
            
            # Ajouter du texte
            cv2.putText(img, f"Webcam Active - Frame {self.frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")
    
    # Dashboard
    if 'dashboard' not in st.session_state:
        st.session_state.dashboard = make_dashboard()
    
    st.plotly_chart(st.session_state.dashboard)
    
    # Composant WebRTC
    webrtc_ctx = webrtc_streamer(
        key="webcam",
        video_processor_factory=VideoProcessor,
        rtc_configuration=rtc_configuration,
        media_stream_constraints={
            "video": {"width": 640, "height": 480},
            "audio": False
        },
    )
    
    # Contrôles
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Démarrer l'analyse", type="primary"):
            st.info("L'analyse commence lorsque la webcam est active")
    with col2:
        if st.button("📊 Mettre à jour les stats"):
            # Simulation de mise à jour
            focus = 65.5
            eyes_open = 78.2
            face_detected = 92.1
            stability = 71.8
            
            update_dashboard(st.session_state.dashboard,
                            focus, 100 - eyes_open, face_detected, stability)
            st.rerun()

# -----------------------------
# APPLICATION PRINCIPALE
# -----------------------------
def main():
    """Fonction principale de l'application"""
    
    st.sidebar.title("Configuration")
    
    # Sélection du mode
    mode = st.sidebar.selectbox(
        "Mode d'exécution",
        ["Démonstration", "Local avec Webcam"],
        index=0 if DEMO_MODE else 1
    )
    
    if DEMO_MODE and mode == "Local avec Webcam":
        st.sidebar.warning("⚠️ Mode Webcam non disponible sur Streamlit Cloud")
        mode = "Démonstration"
    
    # Informations
    st.sidebar.info("""
    **Instructions :**
    - **Démonstration** : Données simulées
    - **Local avec Webcam** : Nécessite une exécution locale
    """)
    
    # Exécuter le mode sélectionné
    if mode == "Démonstration":
        run_demo_mode()
    else:
        if WEBRTC_AVAILABLE:
            run_webcam_mode()
        else:
            st.error("streamlit-webrtc n'est pas disponible")
            run_demo_mode()

# -----------------------------
# POINT D'ENTRÉE
# -----------------------------
if __name__ == "__main__":
    main()