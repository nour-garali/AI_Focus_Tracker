import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import math
import time
from collections import deque
import av 
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os # Pour vérifier l'existence du modèle

# Import conditionnel pour TensorFlow pour éviter les erreurs si non installé
try:
    from tensorflow.keras.models import load_model
    from tensorflow.keras.losses import MeanSquaredError
except ImportError:
    # st.error("TensorFlow n'est pas installé.") # En cloud, ceci s'affiche une fois
    load_model = None
    MeanSquaredError = None

# -----------------------------
# 1. CONFIGURATION
# -----------------------------
MODEL_PATH = "best_gaze_model.h5"
FPS_TARGET = 15
WINDOW_SEC = 3
EAR_THRESHOLD = 0.22
EYE_CLOSED_CONSEC_FRAMES = 3
STABILITY_MOVEMENT_THRESH = 25
PRIVACY_BLUR = False 
TILT_CENTER = 0.0 # Valeur par défaut pour le WebRTC

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

# -----------------------------
# 2. UTILITAIRES & DASHBOARD
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

# Fonctions de couleur réintégrées
def color_bar_stability(val):
    if val < 30: return "red"
    elif val < 70: return "orange"
    else: return "green"

def color_bar(val):
    if val > 70: return "green"
    elif val > 40: return "orange"
    else: return "red"

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

# Fonction pour mettre à jour l'état session sans erreur de clé
def update_metrics_in_session(focus, eye_closed_val, face_detected_val, unstable_val, feedback_str):
    # La stabilité est l'inverse de unstable_val dans le calcul focus
    eyes_open = 100 - eye_closed_val
    stable = 100 - unstable_val
    
    st.session_state.dashboard_metrics = {
        "Focus": round(focus, 2),
        "EyesOpen": round(eyes_open, 2),
        "FaceDetected": round(face_detected_val, 2),
        "Stability": round(stable, 2), 
        "Feedback": feedback_str
    }

# -----------------------------
# 3. CHARGEMENT MODÈLE (Cache)
# -----------------------------
@st.cache_resource
def load_gaze_model(path):
    if not os.path.exists(path):
        st.warning(f"❌ Fichier modèle introuvable: {path}. Modèle désactivé.")
        return None
    if load_model is None: 
        st.error("❌ TensorFlow n'est pas disponible. Modèle désactivé.")
        return None
    try:
        model_local = load_model(path, custom_objects={'mse': MeanSquaredError()})
        print("✅ Modèle gaze chargé.")
        return model_local
    except Exception as e:
        st.error(f"❌ Erreur chargement modèle : {e}. Modèle désactivé.")
        return None

GAZE_MODEL = load_gaze_model(MODEL_PATH)


# ----------------------------------------------------
# 4. CLASSE DE TRAITEMENT VIDÉO (Logique WebRTC)
# ----------------------------------------------------
class FocusTracker(VideoTransformerBase):
    def __init__(self, model_gaze, tilt_center_val):
        self.model = model_gaze
        self.tilt_center = tilt_center_val
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )

        # Files d'attente et compteurs (Logique originale)
        self.gaze_queue = deque(maxlen=int(WINDOW_SEC * FPS_TARGET))
        self.center_queue = deque(maxlen=int(WINDOW_SEC * FPS_TARGET))
        self.consecutive_eye_closed = 0
        self.counters = {"center_gaze":0, "left":0, "right":0,
                         "eye_closed":0, "head_tilt":0, "unstable":0, "total":0, "no_face":0}
        self.ear_history = deque(maxlen=5)

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr")
        h, w = img.shape[:2]

        self.counters["total"] += 1
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        feedback_msgs = []
        
        # Initialisation locale
        face_detected_val = 0
        eye_closed_val = 100
        unstable_val = 100
        focus = 0

        # ---------- No face detected
        if not res.multi_face_landmarks:
            self.counters["no_face"] += 1
            self.counters["eye_closed"] += 1
            
            if self.counters["total"] > 0:
                 eye_closed_val = min(100,(self.counters["eye_closed"]/self.counters["total"])*100)
                 face_detected_val = min(100,((self.counters["total"]-self.counters["no_face"])/self.counters["total"])*100)
            
            self.gaze_queue.append(0)
            feedback_msgs.append("No face detected")
            
        else:
            # ---------- Face detected
            lm = res.multi_face_landmarks[0].landmark
            
            # 1. Bounding Box & ROI
            xs_all = [lm[i].x*w for i in range(len(lm))]
            ys_all = [lm[i].y*h for i in range(len(lm))]
            x_min, y_min = max(0,int(min(xs_all)-10)), max(0,int(min(ys_all)-10))
            x_max, y_max = min(w-1,int(max(xs_all)+10)), min(h-1,int(max(ys_all)+10))
            
            # Clamp coordinates
            x_min, x_max = max(0, x_min), min(w, x_max)
            y_min, y_max = max(0, y_min), min(h, y_max)
            
            face_roi = img[y_min:y_max, x_min:x_max].copy()

            # 2. Gaze model
            gaze = "CENTER"
            pred = 0.0
            if self.model is not None and face_roi.size > 0:
                try:
                    img_resized = cv2.resize(face_roi, (64, 64)) / 255.0
                    pred = float(self.model.predict(np.expand_dims(img_resized, 0), verbose=0)[0][0])
                except Exception:
                    pred = 0.0

            if pred > 0.5: gaze = "RIGHT"; self.counters["right"] += 1
            elif pred < -0.5: gaze = "LEFT"; self.counters["left"] += 1
            else: gaze = "CENTER"; self.counters["center_gaze"] += 1
            self.gaze_queue.append(pred)

            # 3. Eyes (EAR)
            ear = (eye_aspect_ratio(lm, LEFT_EYE_IDX, w, h) + eye_aspect_ratio(lm, RIGHT_EYE_IDX, w, h)) / 2.0
            current_tilt, _, _ = angle_between_eyes(lm, LEFT_EYE_IDX, RIGHT_EYE_IDX, w, h)
            tilt_delta = abs(current_tilt - self.tilt_center)
            dynamic_ear_threshold = EAR_THRESHOLD + min(0.07, tilt_delta * 0.003)
            
            # Iris logic (simplifiée)
            iris_visible = False
            try:
                if len(lm) > 473:
                    iris_visible = True
            except:
                iris_visible = False

            self.ear_history.append(ear)
            ear_smoothed = float(np.mean(self.ear_history)) if self.ear_history else ear

            eyes_closed_detected = (ear_smoothed < dynamic_ear_threshold)
            
            if eyes_closed_detected:
                self.consecutive_eye_closed += 1
            else:
                self.consecutive_eye_closed = 0
            
            eye_closed_flag = (self.consecutive_eye_closed >= EYE_CLOSED_CONSEC_FRAMES)
            if eye_closed_flag: 
                self.counters["eye_closed"] += 1
                feedback_msgs.append("Eyes Closed")

            # 4. Stability
            center = ((x_min+x_max)/2, (y_min+y_max)/2)
            self.center_queue.append(center)
            unstable_val = 0
            if len(self.center_queue) >= 3:
                var_x = np.var([p[0] for p in self.center_queue])
                var_y = np.var([p[1] for p in self.center_queue])
                movement = math.sqrt(var_x + var_y)
                
                if movement < 5: 
                    unstable_val = 20
                    feedback_msgs.append("Too stable")
                elif movement > STABILITY_MOVEMENT_THRESH:
                    unstable_val = 100
                else:
                    unstable_val = int((movement / STABILITY_MOVEMENT_THRESH) * 100)

            # 5. Calcul des métriques
            gaze_focus_smoothed = np.mean([1 if abs(g)<0.5 else 0 for g in self.gaze_queue])*100
            
            eye_closed_val = min(100,(self.counters["eye_closed"]/self.counters["total"])*100)
            face_detected_val = min(100,((self.counters["total"]-self.counters["no_face"])/self.counters["total"])*100)
            
            focus = (0.4*gaze_focus_smoothed + 0.2*(100-eye_closed_val) + 0.2*face_detected_val +0.2*(100-unstable_val))
            focus = max(0.0, min(100.0, focus))

            # Dessin feedback
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            gaze_status_text = f"Gaze:{gaze} (Model {'ON' if self.model is not None else 'OFF'})"
            cv2.putText(img, gaze_status_text, (10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            for idx, msg in enumerate(feedback_msgs):
                cv2.putText(img, msg, (10, 60 + 30 * idx), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # --- Mise à jour de Session State ---
        update_metrics_in_session(focus, eye_closed_val, face_detected_val, unstable_val, " | ".join(feedback_msgs))
        
        return av.VideoFrame.from_ndarray(img, format="bgr")

# -----------------------------
# 5. STREAMLIT APP
# -----------------------------
st.set_page_config(layout="wide", page_title="AI Focus Tracker")
st.title("AI Focus Tracker - Streamlit Cloud")

# Initialisation de Session State 
if 'dashboard_metrics' not in st.session_state:
    st.session_state.dashboard_metrics = {
        "Focus": 0, "EyesOpen": 0, "FaceDetected": 0, "Stability": 0, "Feedback": "En attente du démarrage..."
    }
if 'fig_dashboard' not in st.session_state:
    st.session_state.fig_dashboard = make_dashboard()

# Placeholders
st_plot = st.empty()
st_feedback = st.empty()

# --------------------
# WEBRTC STREAMER
# --------------------
col_webrtc, col_metrics = st.columns([1, 1])

with col_webrtc:
    st.markdown("### 🎥 Flux Vidéo")
    webrtc_ctx = webrtc_streamer(
        key="focus-tracker-key",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_processor_factory=lambda: FocusTracker(GAZE_MODEL, TILT_CENTER), 
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

# --------------------
# MISE À JOUR DASHBOARD (Dans la colonne Metrics)
# --------------------
with col_metrics:
    st.markdown("### 📊 Tableau de Bord Concentration")
    
    # Affichage du dashboard
    metrics = st.session_state.dashboard_metrics
    fig_updated = st.session_state.fig_dashboard
    
    # Mise à jour des valeurs et couleurs
    values = [metrics["Focus"], metrics["EyesOpen"], metrics["FaceDetected"], metrics["Stability"]]
    
    for i in range(4):
        # Mise à jour des valeurs
        fig_updated.data[i].value = values[i]
        
        # Mise à jour des couleurs (Basé sur la métrique affichée)
        if i == 0: # Focus
            fig_updated.data[i].gauge.bar.color = color_bar(values[i])
        elif i == 1: # Yeux ouverts
            fig_updated.data[i].gauge.bar.color = color_bar(values[i])
        elif i == 2: # Visage détecté
            fig_updated.data[i].gauge.bar.color = color_bar(values[i])
        elif i == 3: # Stabilité (Notez que la métrique en session state est déjà 100-unstable)
            fig_updated.data[i].gauge.bar.color = color_bar_stability(values[i])

    # Affichage du graphique mis à jour
    # L'utilisation de st.empty() n'est pas nécessaire ici, car nous sommes dans une colonne
    st_plot.plotly_chart(fig_updated, use_container_width=True)

# --------------------
# AFFICHAGE FEEDBACK
# --------------------
if webrtc_ctx.state.playing:
    st_feedback.info(f"Status: **En cours** | Feedback: {metrics['Feedback']}")
else:
    st_feedback.warning("Status: **Stopped** | Cliquez sur **Start** ci-dessus pour lancer l'analyse.")