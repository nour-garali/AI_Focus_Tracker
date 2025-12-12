# app_streamlit.py - VERSION CORRIGÉE AVEC streamlit-webrtc
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
# IMPORTATION DE streamlit-webrtc
# -----------------------------
try:
    import streamlit_webrtc as webrtc
    from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    st.error("streamlit-webrtc n'est pas installé. Ajoutez-le à requirements.txt")
    st.stop()

# -----------------------------
# CHARGEMENT DES BIBLIOTHÈQUES
# -----------------------------
# Suppression de la détection STREAMLIT_CLOUD et des simulations
try:
    import cv2
    import mediapipe as mp
    from tensorflow.keras.models import load_model
    from tensorflow.keras.losses import MeanSquaredError
    TENSORFLOW_AVAILABLE = True
except ImportError as e:
    st.error(f"Erreur d'importation: {e}")
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
CALIBRATION_FRAMES = 50
DASHBOARD_UPDATE_INTERVAL = 0.5

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

DEBUG = False

# -----------------------------
# INITIALISATION MEDIAPIPE
# -----------------------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

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
# LOAD MODEL (VERSION ROBUSTE)
# -----------------------------
def load_gaze_model(path):
    """Charge le modèle - version robuste"""
    try:
        if not TENSORFLOW_AVAILABLE:
            st.warning("TensorFlow non disponible - Mode simulation")
            return None
            
        if not os.path.exists(path):
            st.error(f"Fichier modèle {path} non trouvé")
            return None
        
        # Charge le modèle avec gestion d'erreur
        try:
            model_local = load_model(path, custom_objects={'mse': MeanSquaredError()})
        except:
            try:
                model_local = load_model(path, compile=False)
            except Exception as e:
                if "conv2d" in str(e).lower() or "conv1" in str(e).lower():
                    st.warning("⚠️ Modèle chargé en mode limité (couche Conv2D)")
                    model_local = load_model(path, compile=False)
                else:
                    raise e
        
        st.success("✅ Modèle gaze chargé.")
        return model_local
        
    except Exception as e:
        st.warning(f"🧪 Mode simulation: {str(e)[:60]}")
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
# CORRECTION : Variables manquantes
# -----------------------------
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
        return 10.0, 10.0  # Valeurs par défaut

# -----------------------------
# CLASSE PROCESSOR POUR streamlit-webrtc
# -----------------------------
class FocusTrackerProcessor(VideoProcessorBase):
    def __init__(self):
        super().__init__()
        # Initialiser MediaPipe
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Charger le modèle
        self.model = load_gaze_model(MODEL_PATH)
        self.model_enabled = True
        
        # Variables d'état (comme dans votre main_loop)
        self.width = 640
        self.height = 480
        self.fps_interval = 1.0 / FPS_TARGET
        self.gaze_queue = deque(maxlen=int(WINDOW_SEC * FPS_TARGET))
        self.center_queue = deque(maxlen=int(WINDOW_SEC * FPS_TARGET))
        self.consecutive_eye_closed = 0
        self.counters = {
            "center_gaze": 0, "left": 0, "right": 0,
            "eye_closed": 0, "head_tilt": 0, "unstable": 0, 
            "total": 0, "no_face": 0
        }
        self.ear_history = deque(maxlen=5)
        self.last_dashboard_update = 0.0
        self.tilt_center = 0.0
        self.demo_counter = 0
        
        # Pour partager les données avec le dashboard
        self.current_focus = 0.0
        self.current_eye_closed_val = 0.0
        self.current_face_detected_val = 0.0
        self.current_unstable_val = 0.0
        self.current_frame_display = None
        
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Méthode appelée pour chaque frame (remplace votre main_loop)"""
        # Convertir la frame en image OpenCV
        img = frame.to_ndarray(format="bgr24")
        self.width = img.shape[1]
        self.height = img.shape[0]
        
        # Démarrer le timer pour FPS
        loop_t0 = time.time()
        
        # Incrémenter les compteurs
        self.counters["total"] += 1
        self.demo_counter += 1
        
        # Copier l'image pour l'affichage
        frame_display = cv2.GaussianBlur(img, (51, 51), 0) if PRIVACY_BLUR else img.copy()
        
        # Traitement MediaPipe
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        feedback_msgs = []

        # ---------- No face detected
        if not res.multi_face_landmarks:
            self.counters["no_face"] += 1
            self.counters["eye_closed"] += 1
            eye_closed_val = min(100, (self.counters["eye_closed"] / self.counters["total"]) * 100)
            face_detected_val = min(100, ((self.counters["total"] - self.counters["no_face"]) / self.counters["total"]) * 100)
            unstable_val = 0
            focus = 0
            self.gaze_queue.append(0)
            feedback_msgs.append("No face detected")
            
            # Mettre à jour les données pour le dashboard
            self.current_focus = focus
            self.current_eye_closed_val = eye_closed_val
            self.current_face_detected_val = face_detected_val
            self.current_unstable_val = unstable_val
            self.current_frame_display = frame_display
            
            return frame.to_image()

        # ---------- Face detected
        lm = res.multi_face_landmarks[0].landmark
        xs_all = [lm[i].x * self.width for i in range(len(lm))]
        ys_all = [lm[i].y * self.height for i in range(len(lm))]
        x_min = max(0, int(min(xs_all) - 10))
        y_min = max(0, int(min(ys_all) - 10))
        x_max = min(self.width - 1, int(max(xs_all) + 10))
        y_max = min(self.height - 1, int(max(ys_all) + 10))
        face_roi = img[y_min:y_max, x_min:x_max]

        if PRIVACY_BLUR:
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(self.width, x_max)
            y_max = min(self.height, y_max)
            face_roi = img[y_min:y_max, x_min:x_max]
            h_roi, w_roi, _ = face_roi.shape
            h_disp = y_max - y_min
            w_disp = x_max - x_min

            h_min = min(h_roi, h_disp)
            w_min = min(w_roi, w_disp)
            face_roi = face_roi[:h_min, :w_min]
            frame_display[y_min:y_min + h_min, x_min:x_min + w_min] = face_roi

        # Gaze model
        pred = 0.0
        if self.model_enabled and self.model is not None:
            try:
                if face_roi.size > 0:
                    img_resized = cv2.resize(face_roi, (64, 64)) / 255.0
                    pred = float(self.model.predict(np.expand_dims(img_resized, 0), verbose=0)[0][0])
            except:
                pred = 0.0
        else:
            # Simulation si modèle manquant
            pred = math.sin(self.demo_counter * 0.1) * 0.8
            
        if pred > 0.5: 
            gaze = "RIGHT"
            self.counters["right"] += 1
        elif pred < -0.5: 
            gaze = "LEFT"
            self.counters["left"] += 1
        else: 
            gaze = "CENTER"
            self.counters["center_gaze"] += 1
        
        self.gaze_queue.append(pred)

        # Eyes
        ear_left = eye_aspect_ratio(lm, LEFT_EYE_IDX, self.width, self.height)
        ear_right = eye_aspect_ratio(lm, RIGHT_EYE_IDX, self.width, self.height)
        ear = (ear_left + ear_right) / 2.0
        
        current_tilt, _, _ = angle_between_eyes(lm, LEFT_EYE_IDX, RIGHT_EYE_IDX, self.width, self.height)
        tilt_delta = abs(current_tilt - self.tilt_center)
        dynamic_ear_threshold = EAR_THRESHOLD + min(0.07, tilt_delta * 0.003)
        
        eye_open_left, eye_open_right = get_eye_open_values(lm, self.width, self.height)
        
        iris_visible = False
        try:
            left_upper = (lm[159].x * self.width, lm[159].y * self.height)
            left_lower = (lm[145].x * self.width, lm[145].y * self.height)
            right_upper = (lm[386].x * self.width, lm[386].y * self.height)
            right_lower = (lm[374].x * self.width, lm[374].y * self.height)
            iris_left_y = lm[468].y * self.height if len(lm) > 468 else None
            iris_right_y = lm[473].y * self.height if len(lm) > 473 else None
            if iris_left_y is not None and iris_right_y is not None and eye_open_left > 2.5 and eye_open_right > 2.5:
                iris_visible = True
        except:
            iris_visible = False

        self.ear_history.append(ear)
        ear_smoothed = float(np.mean(self.ear_history)) if len(self.ear_history) > 0 else ear
        
        eyes_closed_detected = (ear_smoothed < dynamic_ear_threshold) and (not iris_visible)
        
        if eyes_closed_detected:
            self.consecutive_eye_closed += 1
        else:
            self.consecutive_eye_closed = 0
            
        eye_closed_flag = (self.consecutive_eye_closed >= EYE_CLOSED_CONSEC_FRAMES)
        
        if eye_closed_flag: 
            self.counters["eye_closed"] += 1
        if eye_closed_flag: 
            feedback_msgs.append("Eyes Closed")

        # Stability
        center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
        self.center_queue.append(center)
        
        unstable_flag = False
        instability_score = 0
        
        if len(self.center_queue) >= 3:
            var_x = np.var([p[0] for p in self.center_queue])
            var_y = np.var([p[1] for p in self.center_queue])
            movement = math.sqrt(var_x + var_y)
            
            if movement < 5: 
                instability_score = 20
                unstable_flag = True
                feedback_msgs.append("Too stable")
            elif movement > STABILITY_MOVEMENT_THRESH:
                instability_score = 100
            else:
                instability_score = int((movement / STABILITY_MOVEMENT_THRESH) * 100)
        
        unstable_val = instability_score

        # Focus calculation
        gaze_focus_smoothed = np.mean([1 if abs(g) < 0.5 else 0 for g in self.gaze_queue]) * 100
        eye_closed_val = min(100, (self.counters["eye_closed"] / self.counters["total"]) * 100)
        face_detected_val = min(100, ((self.counters["total"] - self.counters["no_face"]) / self.counters["total"]) * 100)
        
        focus = (0.4 * gaze_focus_smoothed + 0.2 * (100 - eye_closed_val) + 
                 0.2 * face_detected_val + 0.2 * (100 - unstable_val))
        focus = max(0.0, min(100.0, focus))

        # Stocker les données pour le dashboard
        self.current_focus = focus
        self.current_eye_closed_val = eye_closed_val
        self.current_face_detected_val = face_detected_val
        self.current_unstable_val = unstable_val

        # Draw feedback on frame
        cv2.rectangle(frame_display, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame_display, f"Gaze:{gaze} (Model {'ON' if self.model_enabled else 'OFF'})", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        for idx, msg in enumerate(feedback_msgs):
            cv2.putText(frame_display, msg, (10, 60 + 30 * idx), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Stocker la frame pour l'affichage
        self.current_frame_display = frame_display
        
        # Gestion FPS
        t_elapsed = time.time() - loop_t0
        if t_elapsed < self.fps_interval: 
            time.sleep(max(0, self.fps_interval - t_elapsed))
        
        # Retourner la frame modifiée
        return av.VideoFrame.from_ndarray(frame_display, format="bgr24")

# -----------------------------
# INTERFACE PRINCIPALE STREAMLIT
# -----------------------------
def main():
    st.title("AI Focus Tracker - Streamlit")
    
    # Initialisation session_state
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'fig_dashboard' not in st.session_state:
        st.session_state.fig_dashboard = make_dashboard()
    if 'processor_data' not in st.session_state:
        st.session_state.processor_data = {
            'focus': 0.0,
            'eye_closed': 0.0,
            'face_detected': 0.0,
            'unstable': 0.0,
            'frame': None
        }
    
    # Boutons Start/Stop
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start"):
            st.session_state.running = True
            st.rerun()
    with col2:
        if st.button("⏹ Stop"):
            st.session_state.running = False
            st.rerun()
    
    st.info("Status: " + ("Running" if st.session_state.running else "Stopped"))
    
    # Placeholders pour le dashboard et la vidéo
    st_plot = st.empty()
    st_frame = st.empty()
    st_feedback = st.empty()
    
    # Affichage initial du dashboard
    st_plot.plotly_chart(st.session_state.fig_dashboard)
    
    if st.session_state.running:
        # Créer le composant WebRTC
        ctx = webrtc_streamer(
            key="ai-focus-tracker",
            video_processor_factory=FocusTrackerProcessor,
            mode=WebRtcMode.SENDRECV,
            rtc_configuration={
                "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
            },
            media_stream_constraints={
                "video": {"width": 640, "height": 480},
                "audio": False
            },
        )
        
        # Mettre à jour le dashboard avec les données du processor
        if ctx.video_processor:
            processor = ctx.video_processor
            
            # Récupérer les données du processor
            focus = processor.current_focus
            eye_closed = processor.current_eye_closed_val
            face_detected = processor.current_face_detected_val
            unstable = processor.current_unstable_val
            
            # Mettre à jour le dashboard
            updated_fig = update_dashboard(
                st.session_state.fig_dashboard,
                round(focus, 2),
                round(eye_closed, 2),
                round(face_detected, 2),
                round(unstable, 2)
            )
            st_plot.plotly_chart(updated_fig)
            
            # Afficher la frame
            if processor.current_frame_display is not None:
                st_frame.image(processor.current_frame_display, channels="BGR")
            
            # Afficher le feedback
            st_feedback.text(f"Focus: {focus:.1f}% | Visage détecté: {face_detected:.1f}%")
    else:
        # Écran d'arrêt
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8) 
        cv2.putText(dummy_frame, "SESSION ARRÊTÉE", (640//2 - 200, 480//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 3)
        cv2.putText(dummy_frame, "Cliquez sur Start pour commencer", (640//2 - 250, 480//2 + 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 255), 2)
        
        st_frame.image(dummy_frame, channels="BGR")
        st_feedback.text("Session terminée. Cliquez sur Start pour lancer une nouvelle analyse.")

if __name__ == "__main__":
    main()