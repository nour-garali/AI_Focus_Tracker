# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import math
import time
from collections import deque
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# -----------------------------
# CONFIGURATION
# -----------------------------
MODEL_PATH = "best_gaze_model.h5"
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
    except:
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
    except:
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
    try:
        model_local = load_model(path, custom_objects={'mse': MeanSquaredError()})
        st.success("✅ Modèle gaze chargé.")
        return model_local
    except Exception as e:
        st.warning(f"❌ Erreur chargement modèle : {e}. Model désactivé.")
        return None

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
        fig.data[i].gauge.bar.color = color_bar_stability(val) if i==3 else color_bar(val)
    return fig

# -----------------------------
# FONCTION PRINCIPALE
# -----------------------------
def main():
    # IMPORTANT: Tous les imports sont déjà faits au début du fichier
    # pas d'appel à set_page_config() ici car fait plus bas
    
    # Initialisation session_state
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'fig_dashboard' not in st.session_state:
        st.session_state.fig_dashboard = make_dashboard()
    if 'model' not in st.session_state:
        st.session_state.model = load_gaze_model(MODEL_PATH)
    if 'tilt_center' not in st.session_state:
        st.session_state.tilt_center = 0.0
    
    model_enabled = st.session_state.model is not None
    
    # Titre principal
    st.title("🧠 AI Focus Tracker - Streamlit")
    
    # Boutons Start/Stop
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start"):
            st.session_state.running = True
    with col2:
        if st.button("⏹ Stop"):
            st.session_state.running = False
    
    st.info("Status: " + ("Running" if st.session_state.running else "Stopped"))
    
    # Placeholders
    st_plot = st.empty()
    st_frame = st.empty()
    st_feedback = st.empty()
    
    # Affichage du dashboard
    st_plot.plotly_chart(st.session_state.fig_dashboard)
    
    # -----------------------------
    # CALIBRATION (si nécessaire)
    # -----------------------------
    if st.session_state.running and st.session_state.tilt_center == 0:
        with st.spinner("🔹 Calibration en cours..."):
            try:
                mp_face_mesh = mp.solutions.face_mesh
                face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                                  refine_landmarks=True, min_detection_confidence=0.5,
                                                  min_tracking_confidence=0.5)
                
                # Pour Streamlit Cloud, on ne peut pas utiliser cv2.VideoCapture(0)
                # On va simuler ou utiliser une image de test
                st.warning("⚠️ Fonction caméra désactivée sur Streamlit Cloud")
                st.session_state.tilt_center = 0.0  # Valeur par défaut
                st.success("✅ Calibration simulée terminée.")
                
            except Exception as e:
                st.error(f"Erreur calibration: {e}")
                st.session_state.tilt_center = 0.0
    
    # -----------------------------
    # MAIN LOOP SIMPLIFIÉ (pour Streamlit Cloud)
    # -----------------------------
    if st.session_state.running:
        try:
            # Simulation des données pour Streamlit Cloud
            # (car cv2.VideoCapture(0) ne fonctionne pas)
            
            focus_simulated = np.random.uniform(70, 90)
            eye_closed_simulated = np.random.uniform(0, 10)
            face_detected_simulated = 95
            unstable_simulated = np.random.uniform(10, 30)
            
            # Mettre à jour le dashboard
            update_dashboard(st.session_state.fig_dashboard, 
                           focus_simulated, 
                           eye_closed_simulated, 
                           face_detected_simulated, 
                           unstable_simulated)
            
            st_plot.plotly_chart(st.session_state.fig_dashboard)
            
            # Afficher une image de test
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(test_image, "STREAMLIT CLOUD - MODE SIMULATION", 
                       (640//2 - 300, 480//2 - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(test_image, f"Focus: {focus_simulated:.1f}%", 
                       (640//2 - 100, 480//2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(test_image, "⚠️ Webcam non disponible en cloud", 
                       (640//2 - 200, 480//2 + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            
            st_frame.image(test_image, channels="BGR")
            st_feedback.text("Mode simulation activé | Test en cours")
            
            # Pause courte pour éviter les boucles trop rapides
            time.sleep(0.5)
            
        except Exception as e:
            st.error(f"Erreur dans la boucle principale: {e}")
            st.session_state.running = False
    else:
        # Mode arrêté
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(dummy_frame, "SESSION ARRÊTÉE", 
                   (640//2 - 200, 480//2),
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 3)
        st_frame.image(dummy_frame, channels="BGR")
        st_feedback.text("Session terminée. Cliquez sur Start pour lancer une analyse.")

# -----------------------------
# POINT D'ENTRÉE PRINCIPAL
# -----------------------------
# IMPORTANT: set_page_config() DOIT ÊTRE LE PREMIER APPEL STREAMLIT
st.set_page_config(page_title="AI Focus Tracker", layout="wide")

# Puis exécuter la fonction principale
if __name__ == "__main__":
    main()