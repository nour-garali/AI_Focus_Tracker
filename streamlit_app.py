# app_streamlit.py
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
import traceback
# streamlit_app.py
import streamlit as st
import av
import numpy as np
import mediapipe as mp
import math
import time
from collections import deque
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode

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
CALIBRATION_FRAMES = 30

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
@st.cache_resource
def load_gaze_model(path):
    try:
        model_local = load_model(path, custom_objects={'mse': MeanSquaredError()})
        return model_local
    except Exception as e:
        st.warning(f"❌ Erreur chargement modèle : {e}. Mode simulé activé.")
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
# CLASSE DE TRANSFORMATION VIDÉO
# -----------------------------
class FaceAnalyzerTransformer(VideoTransformerBase):
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False, 
            max_num_faces=1,
            refine_landmarks=True, 
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.model = load_gaze_model(MODEL_PATH)
        self.model_enabled = self.model is not None
        
        # Initialisation des états
        self.gaze_queue = deque(maxlen=45)
        self.center_queue = deque(maxlen=15)
        self.consecutive_eye_closed = 0
        self.ear_history = deque(maxlen=5)
        self.tilt_center = 0.0
        self.calibrated = False
        self.calibration_counter = 0
        self.calibration_values = []
        
        # Compteurs
        self.counters = {
            "center_gaze": 0, "left": 0, "right": 0,
            "eye_closed": 0, "head_tilt": 0, "unstable": 0, 
            "total": 0, "no_face": 0
        }
        
        self.feedback_msgs = []
        self.focus_value = 0
        self.eye_closed_val = 0
        self.face_detected_val = 0
        self.unstable_val = 0

    def calibrate_tilt(self, landmarks, width, height):
        if self.calibration_counter < CALIBRATION_FRAMES:
            tilt, _, _ = angle_between_eyes(landmarks, LEFT_EYE_IDX, RIGHT_EYE_IDX, width, height)
            self.calibration_values.append(tilt)
            self.calibration_counter += 1
            return False
        
        if not self.calibrated and self.calibration_values:
            self.tilt_center = float(np.mean(self.calibration_values))
            self.calibrated = True
            st.success(f"✅ Calibration terminée. Tilt_center={self.tilt_center:.2f}")
        return True

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        height, width = img.shape[:2]
        
        # Copie pour affichage
        display_img = img.copy()
        if PRIVACY_BLUR:
            display_img = cv2.GaussianBlur(display_img, (51, 51), 0)
        
        self.feedback_msgs = []
        self.counters["total"] += 1
        
        # Traitement avec MediaPipe
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = self.face_mesh.process(rgb)
        
        if not res.multi_face_landmarks:
            self.counters["no_face"] += 1
            self.counters["eye_closed"] += 1
            self.feedback_msgs.append("Pas de visage détecté")
            gaze = "AUCUN"
        else:
            lm = res.multi_face_landmarks[0].landmark
            
            # Calibration si nécessaire
            if not self.calibrated:
                if self.calibrate_tilt(lm, width, height):
                    # Dessiner le message de calibration
                    cv2.putText(display_img, "CALIBRATION EN COURS...", 
                               (width//2 - 150, height//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    return av.VideoFrame.from_ndarray(display_img, format="bgr24")
            
            # Extraction ROI du visage
            xs_all = [lm[i].x * width for i in range(len(lm))]
            ys_all = [lm[i].y * height for i in range(len(lm))]
            x_min, y_min = max(0, int(min(xs_all) - 10)), max(0, int(min(ys_all) - 10))
            x_max, y_max = min(width - 1, int(max(xs_all) + 10)), min(height - 1, int(max(ys_all) + 10))
            
            # Détection du regard
            pred = 0.0
            if self.model_enabled and x_max > x_min and y_max > y_min:
                try:
                    face_roi = img[y_min:y_max, x_min:x_max]
                    if face_roi.size > 0:
                        img_resized = cv2.resize(face_roi, (64, 64)) / 255.0
                        pred = float(self.model.predict(np.expand_dims(img_resized, 0), verbose=0)[0][0])
                except:
                    pred = 0.0
            
            # Détermination de la direction du regard
            if pred > 0.5: 
                gaze = "DROITE"
                self.counters["right"] += 1
            elif pred < -0.5: 
                gaze = "GAUCHE"
                self.counters["left"] += 1
            else: 
                gaze = "CENTRE"
                self.counters["center_gaze"] += 1
            
            self.gaze_queue.append(pred)
            
            # Calcul EAR (Eye Aspect Ratio)
            ear_left = eye_aspect_ratio(lm, LEFT_EYE_IDX, width, height)
            ear_right = eye_aspect_ratio(lm, RIGHT_EYE_IDX, width, height)
            ear = (ear_left + ear_right) / 2.0
            self.ear_history.append(ear)
            ear_smoothed = float(np.mean(self.ear_history))
            
            # Détection yeux fermés
            current_tilt, _, _ = angle_between_eyes(lm, LEFT_EYE_IDX, RIGHT_EYE_IDX, width, height)
            tilt_delta = abs(current_tilt - self.tilt_center)
            dynamic_ear_threshold = EAR_THRESHOLD + min(0.07, tilt_delta * 0.003)
            
            if ear_smoothed < dynamic_ear_threshold:
                self.consecutive_eye_closed += 1
            else:
                self.consecutive_eye_closed = 0
            
            if self.consecutive_eye_closed >= EYE_CLOSED_CONSEC_FRAMES:
                self.counters["eye_closed"] += 1
                self.feedback_msgs.append("Yeux fermés")
            
            # Calcul stabilité
            center = ((x_min + x_max) / 2, (y_min + y_max) / 2)
            self.center_queue.append(center)
            
            if len(self.center_queue) >= 3:
                var_x = np.var([p[0] for p in self.center_queue])
                var_y = np.var([p[1] for p in self.center_queue])
                movement = math.sqrt(var_x + var_y)
                
                if movement < 5:
                    self.unstable_val = 20
                    self.feedback_msgs.append("Trop stable")
                elif movement > STABILITY_MOVEMENT_THRESH:
                    self.unstable_val = 100
                    self.feedback_msgs.append("Trop mouvementé")
                else:
                    self.unstable_val = int((movement / STABILITY_MOVEMENT_THRESH) * 100)
            
            # Calcul focus
            gaze_focus_smoothed = np.mean([1 if abs(g) < 0.5 else 0 for g in self.gaze_queue]) * 100
            self.eye_closed_val = min(100, (self.counters["eye_closed"] / self.counters["total"]) * 100)
            self.face_detected_val = min(100, ((self.counters["total"] - self.counters["no_face"]) / self.counters["total"]) * 100)
            
            self.focus_value = (0.4 * gaze_focus_smoothed + 
                               0.2 * (100 - self.eye_closed_val) + 
                               0.2 * self.face_detected_val + 
                               0.2 * (100 - self.unstable_val))
            self.focus_value = max(0.0, min(100.0, self.focus_value))
            
            # Dessiner sur l'image
            cv2.rectangle(display_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(display_img, f"Regard: {gaze}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_img, f"Focus: {self.focus_value:.1f}%", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Ajouter les messages de feedback
            for idx, msg in enumerate(self.feedback_msgs):
                cv2.putText(display_img, msg, (10, 90 + 30 * idx),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        return av.VideoFrame.from_ndarray(display_img, format="bgr24")

# -----------------------------
# INTERFACE STREAMLIT
# -----------------------------
st.set_page_config(page_title="AI Focus Tracker", layout="wide")

st.title("🧠 AI Focus Tracker - Surveillance de Concentration")

# Section informations
with st.expander("ℹ️ À propos de cette application"):
    st.write("""
    Cette application analyse votre concentration en temps réel en utilisant :
    - **Détection du regard** (via modèle d'IA)
    - **Ouverture des yeux** (EAR - Eye Aspect Ratio)
    - **Stabilité du visage**
    - **Position de la tête**
    
    🔒 **Confidentialité** : L'analyse se fait localement, aucune donnée n'est enregistrée.
    """)

# Initialisation du state
if 'transformer' not in st.session_state:
    st.session_state.transformer = None

if 'dashboard_fig' not in st.session_state:
    st.session_state.dashboard_fig = make_dashboard()

# Layout
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 Flux Vidéo Live")
    
    # Création du transformer
    if st.session_state.transformer is None:
        st.session_state.transformer = FaceAnalyzerTransformer()
    
    # Configuration WebRTC
    ctx = webrtc_streamer(
        key="face-analysis",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        video_transformer_factory=FaceAnalyzerTransformer,
        media_stream_constraints={"video": True, "audio": False},
        async_transform=False,
    )
    
    if ctx.state.playing:
        st.success("✅ Caméra activée - Analyse en cours")
    else:
        st.warning("⏸️ Cliquez sur 'START' ci-dessous pour démarrer la caméra")
        st.info("L'analyse démarre automatiquement après calibration (30 images)")

with col2:
    st.subheader("📊 Dashboard de Concentration")
    
    # Affichage des métriques
    if st.session_state.transformer:
        transformer = st.session_state.transformer
        
        # Mise à jour du dashboard
        update_dashboard(
            st.session_state.dashboard_fig,
            transformer.focus_value,
            transformer.eye_closed_val,
            transformer.face_detected_val,
            transformer.unstable_val
        )
        
        st.plotly_chart(st.session_state.dashboard_fig, use_container_width=True)
        
        # Statistiques textuelles
        st.metric("Focus Actuel", f"{transformer.focus_value:.1f}%")
        
        with st.expander("📈 Détails des statistiques"):
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Regard Centre", f"{transformer.counters.get('center_gaze', 0)}")
                st.metric("Yeux Fermés", f"{transformer.counters.get('eye_closed', 0)}")
            with col_b:
                st.metric("Regard Gauche", f"{transformer.counters.get('left', 0)}")
                st.metric("Regard Droite", f"{transformer.counters.get('right', 0)}")
        
        # Messages de feedback
        if transformer.feedback_msgs:
            st.warning("⚠️ **Alertes** : " + " | ".join(transformer.feedback_msgs))
        else:
            st.success("✅ Bonne concentration détectée")

# Section contrôle
st.sidebar.title("⚙️ Contrôles")

if st.sidebar.button("🔄 Réinitialiser l'analyse"):
    st.session_state.transformer = FaceAnalyzerTransformer()
    st.session_state.dashboard_fig = make_dashboard()
    st.rerun()

# Paramètres
st.sidebar.subheader("Paramètres")
PRIVACY_BLUR = st.sidebar.toggle("Floutage de confidentialité", value=True)
EAR_THRESHOLD = st.sidebar.slider("Seuil yeux fermés", 0.1, 0.3, 0.22, 0.01)
STABILITY_THRESH = st.sidebar.slider("Seuil stabilité", 10, 50, 25, 5)

# Instructions
st.sidebar.subheader("📋 Instructions")
st.sidebar.info("""
1. Cliquez sur **START** dans le flux vidéo
2. Autorisez l'accès à la caméra
3. Placez-vous face à la caméra
4. L'analyse démarre automatiquement après calibration
""")

# Pied de page
st.sidebar.markdown("---")
st.sidebar.caption("AI Focus Tracker v1.0 | Développé avec Streamlit & MediaPipe")