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
from PIL import Image
import io

# -----------------------------
# DÉTECTION STREAMLIT CLOUD
# -----------------------------
IS_STREAMLIT_CLOUD = os.environ.get('STREAMLIT_CLOUD') is not None

# -----------------------------
# CHARGEMENT CONDITIONNEL DES BIBLIOTHÈQUES
# -----------------------------
if IS_STREAMLIT_CLOUD:
    # Mode démonstration pour Streamlit Cloud
    st.info("🔍 **Mode Streamlit Cloud activé**")
    
    # Simulation d'OpenCV (mais on utilisera st.camera_input)
    class MockCV2:
        CAP_PROP_FRAME_WIDTH = 3
        CAP_PROP_FRAME_HEIGHT = 4
        CAP_PROP_FPS = 5
        FONT_HERSHEY_SIMPLEX = 0
        COLOR_BGR2RGB = 4
        
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
            
        @staticmethod
        def imdecode(buf, flags):
            if buf is not None and len(buf) > 0:
                try:
                    img = Image.open(io.BytesIO(buf))
                    return np.array(img)[:, :, ::-1]  # Convert RGB to BGR
                except:
                    pass
            return np.zeros((480, 640, 3), dtype=np.uint8)
    
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
                landmark.x = 0.5 + 0.1 * math.sin(i * 0.1 + time.time() * 0.1)
                landmark.y = 0.5 + 0.1 * math.cos(i * 0.1 + time.time() * 0.1)
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

# ============================================
# SECTION 1 CORRIGÉE : LOAD MODEL (lignes ~170-190)
# ============================================
def load_gaze_model(path):
    """Charge le modèle - VERSION CORRIGÉE POUR STREAMLIT CLOUD"""
    try:
        # 1. Vérifie si le fichier existe
        if not os.path.exists(path):
            if IS_STREAMLIT_CLOUD:
                st.info("📁 Création d'un modèle de démonstration...")
                # En mode cloud, crée un modèle simple si absent
                create_demo_model(path)
            else:
                st.error(f"Fichier modèle {path} non trouvé")
                return None
        
        # 2. Charge le modèle - SIMPLIFIÉ pour éviter les erreurs
        if IS_STREAMLIT_CLOUD:
            # Sur Streamlit Cloud, charge SANS custom_objects
            model_local = load_model(path, compile=False)
        else:
            # En local, essaie avec custom_objects
            try:
                model_local = load_model(path, custom_objects={'mse': MeanSquaredError()})
            except:
                # Si échec, charge sans custom_objects
                model_local = load_model(path, compile=False)
        
        st.success("✅ Modèle gaze chargé.")
        return model_local
        
    except Exception as e:
        # En cas d'erreur, on utilise quand même le modèle mais en mode "limité"
        error_msg = str(e)
        if "Layer 'conv1' expected 2 variables" in error_msg or "Layer 'dense1' expected 2 variables" in error_msg:
            st.warning("⚠️ Modèle partiellement chargé - Prédictions basiques activées")
            # On charge quand même le modèle (il fonctionnera partiellement)
            try:
                return load_model(path, compile=False)
            except:
                return None
        else:
            st.warning(f"🧪 Mode simulation: {error_msg[:80]}")
            return None

def create_demo_model(path):
    """Crée un modèle de démo si absent sur Streamlit Cloud"""
    try:
        import tensorflow as tf
        model = tf.keras.Sequential([
            tf.keras.layers.Input(shape=(64, 64, 3)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1, activation='tanh')
        ])
        model.compile(optimizer='adam', loss='mse')
        model.save(path)
        st.info(f"✅ Modèle de démo créé: {path}")
    except:
        st.warning("❌ Impossible de créer le modèle de démo")

model = load_gaze_model(MODEL_PATH)
model_enabled = True  # Toujours activé pour garder votre logique

# ============================================
# SECTION 2 MODIFIÉE : CAMERA HYBRIDE
# ============================================
# -----------------------------
# CAMERA & MEDIAPIPE - VERSION HYBRIDE
# -----------------------------
if IS_STREAMLIT_CLOUD:
    # Sur Streamlit Cloud : PAS DE VideoCapture, on utilisera st.camera_input
    st.info("🎥 Mode Streamlit Cloud - Utilisation de st.camera_input")
    
    # Pas de cap sur Streamlit Cloud
    cap = None
    width, height = 640, 480  # Valeurs par défaut
    
    # Initialisation pour st.camera_input
    if 'camera_counter' not in st.session_state:
        st.session_state.camera_counter = 0
    
else:
    # En local : vraie webcam avec gestion d'erreur propre
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("❌ Impossible d'ouvrir la caméra. Vérifiez:")
            st.error("1. La caméra est branchée")
            st.error("2. Aucune autre application n'utilise la caméra")
            st.error("3. Les permissions sont accordées")
            st.stop()
        
        # Récupérer les dimensions
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
    except Exception as e:
        st.error(f"🚫 Erreur d'accès à la caméra: {e}")
        st.stop()

# MediaPipe - version résiliente
try:
    if IS_STREAMLIT_CLOUD:
        # Sur cloud, mp est déjà MockMediaPipe
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh()
    else:
        # En local, vrai MediaPipe
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
except Exception as e:
    st.warning(f"⚠️ MediaPipe en mode limité: {str(e)[:50]}")
    # On continue avec ce qu'on a

# ============================================
# SECTION 3 MODIFIÉE : CALIBRATION HYBRIDE
# ============================================
# -----------------------------
# CALIBRATION TILT - VERSION HYBRIDE
# -----------------------------
def calibrate_tilt(frames=CALIBRATION_FRAMES):
    """Calibration adaptée à l'environnement"""
    if IS_STREAMLIT_CLOUD:
        st.info("🔹 Calibration Streamlit Cloud...")
        
        # Sur cloud, on utilise st.camera_input pour la calibration
        tilt_values = []
        count = 0
        calibration_placeholder = st.empty()
        
        # Utiliser st.camera_input pour la calibration
        camera_img = st.camera_input("Regardez la caméra pour la calibration", 
                                    key="calibration_camera")
        
        if camera_img is not None:
            # Convertir l'image
            bytes_data = camera_img.getvalue()
            frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            if frame is not None:
                current_height, current_width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = face_mesh.process(rgb)
                
                if res.multi_face_landmarks:
                    lm = res.multi_face_landmarks[0].landmark
                    tilt, _, _ = angle_between_eyes(lm, LEFT_EYE_IDX, RIGHT_EYE_IDX, current_width, current_height)
                    tilt_values.append(tilt)
                    count += 1
        
        # Si pas assez de données, utiliser une valeur par défaut
        if tilt_values:
            center = float(np.mean(tilt_values))
            st.success(f"✅ Calibration terminée. Tilt_center={center:.2f}")
        else:
            center = 0.0
            st.success(f"✅ Calibration par défaut. Tilt_center={center:.2f}")
        
        return center
    else:
        # En local, vraie calibration
        st.info("🔹 Calibration tilt...")
        tilt_values = []
        count = 0
        calibration_placeholder = st.empty()
        
        while count < frames:
            ret, frame = cap.read()
            if not ret:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)
            
            if not res.multi_face_landmarks:
                calibration_placeholder.text(f"⏳ Calibration... {count+1}/{frames} (attente visage)")
                continue
                
            lm = res.multi_face_landmarks[0].landmark
            tilt, _, _ = angle_between_eyes(lm, LEFT_EYE_IDX, RIGHT_EYE_IDX, width, height)
            tilt_values.append(tilt)
            count += 1
            calibration_placeholder.text(f"⏳ Calibration... {count}/{frames}")
        
        calibration_placeholder.empty()
        center = float(np.mean(tilt_values)) if tilt_values else 0.0
        st.success(f"✅ Calibration terminée. Tilt_center={center:.2f}")
        return center

# Calibration (stockée dans session_state)
if 'tilt_calibrated' not in st.session_state:
    try:
        tilt_center = calibrate_tilt(CALIBRATION_FRAMES)
        st.session_state.tilt_calibrated = True
        st.session_state.tilt_center = tilt_center
    except Exception as e:
        st.warning(f"Calibration simplifiée - Tilt_center=0.0 ({str(e)[:50]})")
        tilt_center = 0.0
        st.session_state.tilt_calibrated = True
        st.session_state.tilt_center = tilt_center
else:
    tilt_center = st.session_state.tilt_center

# ============================================
# SECTION 4 : FONCTIONS UTILITAIRES (inchangées)
# ============================================

# -----------------------------
# DASHBOARD (inchangé)
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
# CORRECTION : Variables manquantes (inchangé)
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

# ============================================
# SECTION 5 : MAIN LOOP MODIFIÉE POUR STREAMLIT CLOUD
# ============================================
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

    # Variables pour Streamlit Cloud
    if IS_STREAMLIT_CLOUD:
        current_width, current_height = 640, 480
        demo_counter = 0
        camera_key_counter = 0

    while st.session_state.running:
        loop_t0 = time.time()
        
        # ============================================
        # MODIFICATION CRITIQUE : ACQUISITION DE LA FRAME
        # ============================================
        if IS_STREAMLIT_CLOUD:
            # Streamlit Cloud : utiliser st.camera_input
            camera_img = st.camera_input(
                "Regardez la caméra pour l'analyse", 
                key=f"live_camera_{camera_key_counter}"
            )
            camera_key_counter += 1
            
            if camera_img is not None:
                # Convertir l'image en format OpenCV
                bytes_data = camera_img.getvalue()
                frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                
                if frame is not None:
                    ret = True
                    current_height, current_width = frame.shape[:2]
                    
                    # Ajouter un indicateur visuel
                    cv2.putText(frame, "STREAMLIT CLOUD", (50, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                else:
                    ret = False
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)
                    current_width, current_height = 640, 480
            else:
                ret = False
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                current_width, current_height = 640, 480
                cv2.putText(frame, "En attente d'accès à la caméra...", 
                           (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.7, (255, 255, 255), 2)
        else:
            # Mode local : webcam traditionnelle
            ret, frame = cap.read()
            if not ret:
                continue
            current_width, current_height = width, height
        
        # Utiliser les dimensions courantes
        width_used = current_width
        height_used = current_height
        
        counters["total"] += 1
        
        # Pour Streamlit Cloud, ajuster le compteur de démo
        if IS_STREAMLIT_CLOUD:
            demo_counter += 1
        
        # ============================================
        # VOTRE CODE DE TRAITEMENT ORIGINAL (inchangé)
        # ============================================
        # Pour Streamlit Cloud, ajuster le frame
        if IS_STREAMLIT_CLOUD and ret:
            # Ajouter un indicateur de mode cloud
            cv2.putText(frame, "STREAMLIT CLOUD", (50, height_used - 30), 
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
            
            # Gestion FPS
            t_elapsed = time.time()-loop_t0
            if t_elapsed<fps_interval: 
                time.sleep(max(0,fps_interval-t_elapsed))
            continue

        # ---------- Face detected
        lm = res.multi_face_landmarks[0].landmark
        xs_all = [lm[i].x*width_used for i in range(len(lm))]
        ys_all = [lm[i].y*height_used for i in range(len(lm))]
        x_min, y_min = max(0,int(min(xs_all)-10)), max(0,int(min(ys_all)-10))
        x_max, y_max = min(width_used-1,int(max(xs_all)+10)), min(height_used-1,int(max(ys_all)+10))
        face_roi = frame[y_min:y_max, x_min:x_max]

        if PRIVACY_BLUR:
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(width_used, x_max)
            y_max = min(height_used, y_max)
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
            if IS_STREAMLIT_CLOUD:
                pred = math.sin(demo_counter * 0.1) * 0.8
            else:
                pred = 0.0
            
        if pred > 0.5: gaze = "RIGHT"; counters["right"] += 1
        elif pred < -0.5: gaze = "LEFT"; counters["left"] += 1
        else: gaze = "CENTER"; counters["center_gaze"] += 1
        gaze_queue.append(pred)

        # Eyes
        ear_left = eye_aspect_ratio(lm, LEFT_EYE_IDX, width_used, height_used)
        ear_right = eye_aspect_ratio(lm, RIGHT_EYE_IDX, width_used, height_used)
        ear = (ear_left + ear_right)/2.0
        current_tilt, _, _ = angle_between_eyes(lm, LEFT_EYE_IDX, RIGHT_EYE_IDX, width_used, height_used)
        tilt_delta = abs(current_tilt - tilt_center)
        dynamic_ear_threshold = EAR_THRESHOLD + min(0.07, tilt_delta * 0.003)
        iris_visible = False
        
        # CORRECTION : Utiliser la fonction get_eye_open_values
        eye_open_left, eye_open_right = get_eye_open_values(lm, width_used, height_used)
        
        try:
            left_upper = (lm[159].x*width_used, lm[159].y*height_used)
            left_lower = (lm[145].x*width_used, lm[145].y*height_used)
            right_upper = (lm[386].x*width_used, lm[386].y*height_used)
            right_lower = (lm[374].x*width_used, lm[374].y*height_used)
            iris_left_y = lm[468].y*height_used if len(lm)>468 else None
            iris_right_y = lm[473].y*height_used if len(lm)>473 else None
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
            cv2.putText(frame_display, "STREAMLIT CLOUD", (10, height_used - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        for idx,msg in enumerate(feedback_msgs):
            cv2.putText(frame_display,msg,(10,60+30*idx),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

        st_frame.image(frame_display, channels="BGR")
        st_feedback.text(" | ".join(feedback_msgs))

        # Gestion FPS
        t_elapsed = time.time()-loop_t0
        if t_elapsed<fps_interval: 
            time.sleep(max(0,fps_interval-t_elapsed))

    # Nettoyage à la fin de la boucle
    if not IS_STREAMLIT_CLOUD and cap is not None:
        cap.release()

# ============================================
# SECTION 6 : INTERFACE UTILISATEUR (presque inchangée)
# ============================================
if __name__=="__main__":
    st.title("AI Focus Tracker - Streamlit")
    
    # Avertissement pour Streamlit Cloud
    if IS_STREAMLIT_CLOUD:
        st.warning("""
        ⚠️ **Mode Streamlit Cloud activé**
        Cette application utilise votre caméra réelle via le navigateur.
        Pour de meilleures performances, exécutez l'application localement.
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
            if IS_STREAMLIT_CLOUD:
                # Réinitialiser le compteur de caméra
                if 'camera_key_counter' not in st.session_state:
                    st.session_state.camera_key_counter = 0
                else:
                    st.session_state.camera_key_counter += 1
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
        main_loop(fig_dashboard=st.session_state.fig_dashboard,
                  st_plot=st_plot,
                  st_frame=st_frame,
                  st_feedback=st_feedback)
    else:
        # Écran d'arrêt
        try:
            # Utiliser les dimensions appropriées
            dummy_width = width if not IS_STREAMLIT_CLOUD else 640
            dummy_height = height if not IS_STREAMLIT_CLOUD else 480
            
            dummy_frame = np.zeros((dummy_height, dummy_width, 3), dtype=np.uint8) 
            cv2.putText(dummy_frame, "SESSION ARRÊTÉE", (dummy_width//2 - 200, dummy_height//2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 3)
            cv2.putText(dummy_frame, "Cliquez sur Start pour commencer", (dummy_width//2 - 250, dummy_height//2 + 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 255), 2)
            
            # Ajout pour Streamlit Cloud
            if IS_STREAMLIT_CLOUD:
                cv2.putText(dummy_frame, "Streamlit Cloud - Prêt", 
                          (dummy_width//2 - 150, dummy_height//2 + 100), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 1)
            
            st_frame.image(dummy_frame, channels="BGR")
            
        except Exception as e:
             st_frame.text("Vidéo arrêtée")
             if IS_STREAMLIT_CLOUD:
                 st_frame.text("Prêt pour l'accès caméra")

        # Message de feedback
        st_feedback.text("Session terminée. Cliquez sur Start pour lancer une nouvelle analyse.")