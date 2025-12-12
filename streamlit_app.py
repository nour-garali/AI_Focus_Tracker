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
# CHARGEMENT CONDITIONNEL DES BIBLIOTHÈQUES
# -----------------------------
if IS_STREAMLIT_CLOUD:
    # Mode démonstration pour Streamlit Cloud
    st.info("🔍 **Mode démonstration activé** - Streamlit Cloud détecté")
    
    # Simulation d'OpenCV
    class MockCV2:
        CAP_PROP_FRAME_WIDTH = 3
        CAP_PROP_FRAME_HEIGHT = 4
        CAP_PROP_FPS = 5
        FONT_HERSHEY_SIMPLEX = 0
        COLOR_BGR2RGB = 4
        
        @staticmethod
        def VideoCapture(*args):
            class MockCamera:
                def __init__(self):
                    self.is_opened_value = True
                    self.width = 640
                    self.height = 480
                    self.frame_count = 0
                    
                def isOpened(self):
                    return self.is_opened_value
                    
                def get(self, prop):
                    if prop == 3:  # CAP_PROP_FRAME_WIDTH
                        return self.width
                    elif prop == 4:  # CAP_PROP_FRAME_HEIGHT
                        return self.height
                    return 0
                    
                def read(self):
                    self.frame_count += 1
                    # Créer une image de test
                    frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                    
                    # Dessiner un visage simulé
                    center_x, center_y = self.width // 2, self.height // 2
                    
                    # Animation simple
                    offset = int(20 * math.sin(self.frame_count * 0.1))
                    
                    # Tête
                    cv2.ellipse(frame, (center_x, center_y), (100 + offset, 150), 
                               0, 0, 360, (100, 100, 255), -1)
                    
                    # Yeux
                    cv2.circle(frame, (center_x - 40, center_y - 30), 20, (255, 255, 255), -1)
                    cv2.circle(frame, (center_x + 40, center_y - 30), 20, (255, 255, 255), -1)
                    
                    # Pupilles (qui bougent)
                    pupil_offset = int(5 * math.sin(self.frame_count * 0.2))
                    cv2.circle(frame, (center_x - 40 + pupil_offset, center_y - 30), 10, (0, 0, 0), -1)
                    cv2.circle(frame, (center_x + 40 + pupil_offset, center_y - 30), 10, (0, 0, 0), -1)
                    
                    # Bouche
                    mouth_width = 40 + int(10 * math.sin(self.frame_count * 0.15))
                    cv2.ellipse(frame, (center_x, center_y + 40), (mouth_width, 15), 
                               0, 0, 180, (50, 50, 200), 2)
                    
                    # Texte
                    cv2.putText(frame, "DEMO MODE", (50, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Frame: {self.frame_count}", (50, 60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
                    
                    return True, frame
                    
                def release(self):
                    self.is_opened_value = False
                    
            return MockCamera()
            
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
    
    # Simulation de MediaPipe
    class MockMediaPipe:
        class solutions:
            class face_mesh:
                class FaceMesh:
                    def __init__(self, **kwargs):
                        pass
                        
                    def process(self, image):
                        class Result:
                            def __init__(self):
                                self.multi_face_landmarks = [MockLandmarks()]
                        return Result()
    
    class MockLandmarks:
        def __init__(self):
            self.landmark = []
            for i in range(478):
                landmark = type('Landmark', (), {})()
                landmark.x = 0.5 + 0.1 * math.sin(i * 0.1)
                landmark.y = 0.5 + 0.1 * math.cos(i * 0.1)
                self.landmark.append(landmark)
    
    mp = MockMediaPipe()
    
    # Simulation de TensorFlow
    try:
        from tensorflow.keras.models import load_model
        from tensorflow.keras.losses import MeanSquaredError
        TENSORFLOW_AVAILABLE = True
    except:
        TENSORFLOW_AVAILABLE = False
        st.warning("TensorFlow non disponible en mode démo")
        
else:
    # Mode normal (local)
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
# VOTRE CODE ORIGINAL COMMENCE ICI
# -----------------------------
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
    try:
        if not TENSORFLOW_AVAILABLE:
            st.warning("TensorFlow non disponible - Mode simulation")
            return None
            
        if not os.path.exists(path):
            if IS_STREAMLIT_CLOUD:
                st.warning("Modèle non trouvé - Mode simulation activé")
                return None
            else:
                st.error(f"Fichier modèle {path} non trouvé")
                return None
                
        model_local = load_model(path, custom_objects={'mse': MeanSquaredError()})
        st.success("✅ Modèle gaze chargé.")
        return model_local
    except Exception as e:
        st.warning(f"❌ Erreur chargement modèle : {e}. Mode simulation activé.")
        return None

model = load_gaze_model(MODEL_PATH)
model_enabled = True if model is not None else False

# -----------------------------
# CAMERA & MEDIAPIPE
# -----------------------------
try:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        if IS_STREAMLIT_CLOUD:
            # Créer une caméra simulée pour Streamlit Cloud
            cap = cv2.VideoCapture(0)  # Utilise la classe Mock déjà créée
        else:
            st.error("Impossible d'ouvrir la caméra")
            st.stop()
except:
    if IS_STREAMLIT_CLOUD:
        cap = cv2.VideoCapture(0)  # Utilise la classe Mock
    else:
        st.error("Erreur d'accès à la caméra")
        st.stop()

try:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
except:
    width, height = 640, 480

try:
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                      refine_landmarks=True, min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)
except:
    st.warning("MediaPipe non disponible - Utilisation du mode simulation")

# -----------------------------
# CALIBRATION TILT
# -----------------------------
def calibrate_tilt(frames=CALIBRATION_FRAMES):
    st.info("🔹 Calibration tilt...")
    tilt_values = []
    count = 0
    while count < frames:
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            continue
        lm = res.multi_face_landmarks[0].landmark
        tilt, _, _ = angle_between_eyes(lm, LEFT_EYE_IDX, RIGHT_EYE_IDX, width, height)
        tilt_values.append(tilt)
        count += 1
    center = float(np.mean(tilt_values)) if tilt_values else 0.0
    st.success(f"✅ Calibration terminée. Tilt_center={center:.2f}")
    return center

tilt_center = 0.0  # Valeur par défaut
try:
    tilt_center = calibrate_tilt()
except:
    st.warning("Calibration simulée - Tilt_center=0.0")

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

fig_dashboard = make_dashboard()
st_plot = st.empty()
st_frame = st.empty()
st_feedback = st.empty()

# -----------------------------
# CORRECTION : Variables manquantes
# -----------------------------
# Ajouter ces variables qui étaient manquantes dans votre code
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
# MAIN LOOP STREAMLIT
# -----------------------------
def main_loop(fig_dashboard=None, st_plot=None, st_frame=None, st_feedback=None):
    global model_enabled, DEBUG, IS_STREAMLIT_CLOUD
    
    fps_interval = 1.0 / FPS_TARGET
    gaze_queue = deque(maxlen=int(WINDOW_SEC * FPS_TARGET))
    center_queue = deque(maxlen=int(WINDOW_SEC * FPS_TARGET))
    consecutive_eye_closed = 0
    counters = {"center_gaze":0, "left":0, "right":0,
                "eye_closed":0, "head_tilt":0, "unstable":0, "total":0, "no_face":0}
    ear_history = deque(maxlen=5)
    last_dashboard_update_local = 0.0

    # Ajout pour Streamlit Cloud
    demo_counter = 0

    while st.session_state.running:
        loop_t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            continue
        counters["total"] += 1
        demo_counter += 1
        
        # Pour Streamlit Cloud, ajuster le frame
        if IS_STREAMLIT_CLOUD:
            # Ajouter un indicateur de mode démo
            cv2.putText(frame, "DEMO MODE - STREAMLIT CLOUD", (50, height - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        frame_display = cv2.GaussianBlur(frame,(51,51),0) if PRIVACY_BLUR else frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        feedback_msgs = []

        # ---------- No face detected
        if not res.multi_face_landmarks:
            counters["no_face"] += 1
            counters["eye_closed"] +=1
            eye_closed_val = min(100,(counters["eye_closed"]/counters["total"])*100)
            face_detected_val = min(100,((counters["total"]-counters["no_face"])/counters["total"])*100)
            unstable_val = 0
            focus = 0
            gaze_queue.append(0)
            feedback_msgs.append("No face detected")
            update_dashboard(fig_dashboard, focus, eye_closed_val, face_detected_val, unstable_val)
            st_plot.plotly_chart(fig_dashboard)
            st_frame.image(frame_display, channels="BGR")
            st_feedback.text(" | ".join(feedback_msgs))
            continue

        # ---------- Face detected
        lm = res.multi_face_landmarks[0].landmark
        xs_all = [lm[i].x*width for i in range(len(lm))]
        ys_all = [lm[i].y*height for i in range(len(lm))]
        x_min, y_min = max(0,int(min(xs_all)-10)), max(0,int(min(ys_all)-10))
        x_max, y_max = min(width-1,int(max(xs_all)+10)), min(height-1,int(max(ys_all)+10))
        face_roi = frame[y_min:y_max, x_min:x_max]

        if PRIVACY_BLUR:
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(width, x_max)
            y_max = min(height, y_max)
            face_roi = frame[y_min:y_max, x_min:x_max]
            h_roi, w_roi, _ = face_roi.shape
            h_disp = y_max - y_min
            w_disp = x_max - x_min

            h_min = min(h_roi, h_disp)
            w_min = min(w_roi, w_disp)
            face_roi = face_roi[:h_min, :w_min]
            frame_display[y_min:y_min+h_min, x_min:x_min+w_min] = face_roi

        # Gaze model
        pred = 0.0
        if model_enabled and model is not None:
            try:
                img = cv2.resize(face_roi,(64,64))/255.0
                pred = float(model.predict(np.expand_dims(img,0), verbose=0)[0][0])
            except:
                pred = 0.0
        else:
            # Simulation pour Streamlit Cloud ou modèle manquant
            pred = math.sin(demo_counter * 0.1) * 0.8
            
        if pred > 0.5: gaze = "RIGHT"; counters["right"] += 1
        elif pred < -0.5: gaze = "LEFT"; counters["left"] += 1
        else: gaze = "CENTER"; counters["center_gaze"] += 1
        gaze_queue.append(pred)

        # Eyes
        ear_left = eye_aspect_ratio(lm, LEFT_EYE_IDX, width, height)
        ear_right = eye_aspect_ratio(lm, RIGHT_EYE_IDX, width, height)
        ear = (ear_left + ear_right)/2.0
        current_tilt, _, _ = angle_between_eyes(lm, LEFT_EYE_IDX, RIGHT_EYE_IDX, width, height)
        tilt_delta = abs(current_tilt - tilt_center)
        dynamic_ear_threshold = EAR_THRESHOLD + min(0.07, tilt_delta * 0.003)
        iris_visible = False
        
        # CORRECTION : Utiliser la fonction get_eye_open_values
        eye_open_left, eye_open_right = get_eye_open_values(lm, width, height)
        
        try:
            left_upper = (lm[159].x*width, lm[159].y*height)
            left_lower = (lm[145].x*width, lm[145].y*height)
            right_upper = (lm[386].x*width, lm[386].y*height)
            right_lower = (lm[374].x*width, lm[374].y*height)
            iris_left_y = lm[468].y*height if len(lm)>468 else None
            iris_right_y = lm[473].y*height if len(lm)>473 else None
            if iris_left_y is not None and iris_right_y is not None and eye_open_left>2.5 and eye_open_right>2.5:
                iris_visible=True
        except:
            iris_visible=False

        ear_history.append(ear)
        ear_smoothed = float(np.mean(ear_history)) if len(ear_history)>0 else ear
        eyes_closed_detected = (ear_smoothed < dynamic_ear_threshold) and (not iris_visible)
        if eyes_closed_detected:
            consecutive_eye_closed += 1
        else:
            consecutive_eye_closed = 0
        eye_closed_flag = (consecutive_eye_closed >= EYE_CLOSED_CONSEC_FRAMES)
        if eye_closed_flag: counters["eye_closed"] +=1
        if eye_closed_flag: feedback_msgs.append("Eyes Closed")

        # Stability
        center = ((x_min+x_max)/2, (y_min+y_max)/2)
        center_queue.append(center)
        unstable_flag=False
        instability_score=0
        if len(center_queue)>=3:
            var_x=np.var([p[0] for p in center_queue])
            var_y=np.var([p[1] for p in center_queue])
            movement=math.sqrt(var_x + var_y)
            if movement<5: 
                instability_score=20
                unstable_flag=True
                feedback_msgs.append("Too stable")
            elif movement>STABILITY_MOVEMENT_THRESH:
                instability_score=100
            else:
                instability_score=int((movement/STABILITY_MOVEMENT_THRESH)*100)
        unstable_val = instability_score

        # Focus calculation
        gaze_focus_smoothed = np.mean([1 if abs(g)<0.5 else 0 for g in gaze_queue])*100
        eye_closed_val = min(100,(counters["eye_closed"]/counters["total"])*100)
        face_detected_val = min(100,((counters["total"]-counters["no_face"])/counters["total"])*100)
        focus = (0.4*gaze_focus_smoothed + 0.2*(100-eye_closed_val) + 0.2*face_detected_val +0.2*(100-unstable_val))
        focus = max(0.0, min(100.0, focus))

        # Dashboard
        if time.time()-last_dashboard_update_local > DASHBOARD_UPDATE_INTERVAL:
            update_dashboard(fig_dashboard, round(focus,2), round(eye_closed_val,2), round(face_detected_val,2), round(unstable_val,2))
            last_dashboard_update_local = time.time()
        st_plot.plotly_chart(fig_dashboard)

        # Draw feedback on frame
        cv2.rectangle(frame_display, (x_min,y_min), (x_max,y_max), (0,255,0), 2)
        cv2.putText(frame_display,f"Gaze:{gaze} (Model {'ON' if model_enabled else 'OFF'})",(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
        
        # Ajout mode démo
        if IS_STREAMLIT_CLOUD:
            cv2.putText(frame_display, "STREAMLIT CLOUD - DEMO", (10, height - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        for idx,msg in enumerate(feedback_msgs):
            cv2.putText(frame_display,msg,(10,60+30*idx),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

        st_frame.image(frame_display, channels="BGR")
        st_feedback.text(" | ".join(feedback_msgs))

        # Gestion FPS
        t_elapsed = time.time()-loop_t0
        if t_elapsed<fps_interval: 
            time.sleep(max(0,fps_interval-t_elapsed))

if __name__=="__main__":
    st.title("AI Focus Tracker - Streamlit")
    
    # Avertissement pour Streamlit Cloud
    if IS_STREAMLIT_CLOUD:
        st.warning("""
        ⚠️ **Mode démonstration activé**
        Cette application fonctionne en mode simulation sur Streamlit Cloud.
        Pour utiliser toutes les fonctionnalités (webcam, modèle réel), exécutez l'application localement.
        """)

    # Initialisation session_state
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'fig_dashboard' not in st.session_state:
        st.session_state.fig_dashboard = make_dashboard()

    # Boutons Start/Stop
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start"):
            st.session_state.running = True
    with col2:
        if st.button("⏹ Stop"):
            st.session_state.running = False

    st.info("Status: " + ("Running" if st.session_state.running else "Stopped"))

    # Placeholders pour le dashboard et la vidéo
    st_plot = st.empty()
    st_frame = st.empty()
    st_feedback = st.empty()

    # Affichage initial du dashboard
    st_plot.plotly_chart(st.session_state.fig_dashboard)

    if st.session_state.running:
        main_loop(fig_dashboard=st.session_state.fig_dashboard,
                  st_plot=st_plot,
                  st_frame=st_frame,
                  st_feedback=st_feedback)
    else:
        # Écran d'arrêt
        try:
            dummy_frame = np.zeros((height, width, 3), dtype=np.uint8) 
            cv2.putText(dummy_frame, "SESSION ARRÊTÉE", (width//2 - 200, height//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 3)
            cv2.putText(dummy_frame, "Cliquez sur Start pour commencer", (width//2 - 250, height//2 + 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 255), 2)
            
            # Ajout pour Streamlit Cloud
            if IS_STREAMLIT_CLOUD:
                cv2.putText(dummy_frame, "Mode démonstration - Données simulées", 
                          (width//2 - 300, height//2 + 100), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 1)
            
            st_frame.image(dummy_frame, channels="BGR")
            
        except NameError:
             st_frame.text("Vidéo arrêtée")
             if IS_STREAMLIT_CLOUD:
                 st_frame.text("Mode démonstration activé")

        # Message de feedback
        st_feedback.text("Session terminée. Cliquez sur Start pour lancer une nouvelle analyse.")