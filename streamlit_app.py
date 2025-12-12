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
# CHARGEMENT UNIFIÉ DES BIBLIOTHÈQUES
# -----------------------------
# Toujours importer les vraies bibliothèques, elles seront mockées si nécessaire
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
# GESTIONNAIRE DE CAMÉRA UNIFIÉ
# -----------------------------
class CameraManager:
    """Gestionnaire de caméra qui fonctionne sur Streamlit Cloud et en local"""
    
    def __init__(self):
        self.is_streamlit_cloud = IS_STREAMLIT_CLOUD
        self.local_camera = None
        self.width = 640
        self.height = 480
        self.frame_count = 0
        
        if self.is_streamlit_cloud:
            st.info("🔍 **Mode caméra navigateur activé** - Utilisez la caméra de votre navigateur")
    
    def get_frame(self):
        """Obtenir un frame depuis la source appropriée"""
        if self.is_streamlit_cloud:
            return self._get_frame_from_browser()
        else:
            return self._get_frame_from_local_camera()
    
    def _get_frame_from_browser(self):
        """Obtenir un frame depuis le navigateur (Streamlit Cloud)"""
        # Utiliser st.camera_input pour obtenir une image du navigateur
        camera_image = st.camera_input("Capturez votre visage", key=f"camera_{self.frame_count}")
        
        if camera_image is not None:
            self.frame_count += 1
            
            # Convertir l'image Streamlit en format OpenCV
            bytes_data = camera_image.getvalue()
            nparr = np.frombuffer(bytes_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            # Redimensionner si nécessaire
            if frame is not None:
                frame = cv2.resize(frame, (self.width, self.height))
                return True, frame
        
        return False, None
    
    def _get_frame_from_local_camera(self):
        """Obtenir un frame depuis la webcam locale"""
        # Initialiser la caméra locale si ce n'est pas déjà fait
        if self.local_camera is None:
            self.local_camera = cv2.VideoCapture(0)
            if not self.local_camera.isOpened():
                st.error("❌ Impossible d'ouvrir la caméra. Vérifiez:")
                st.error("1. La caméra est branchée")
                st.error("2. Aucune autre application n'utilise la caméra")
                st.error("3. Les permissions sont accordées")
                return False, None
            
            # Définir les dimensions
            self.width = int(self.local_camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.local_camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Lire le frame
        ret, frame = self.local_camera.read()
        return ret, frame
    
    def get_dimensions(self):
        """Obtenir les dimensions de la caméra"""
        return self.width, self.height
    
    def release(self):
        """Libérer les ressources de la caméra"""
        if self.local_camera is not None:
            self.local_camera.release()
            self.local_camera = None

# -----------------------------
# CONFIGURATION (identique à votre code)
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
# UTILITAIRES (identique à votre code)
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
# SECTION 1 : LOAD MODEL (adaptée)
# ============================================
def load_gaze_model(path):
    """Charge le modèle - VERSION UNIFIÉE"""
    try:
        # Vérifie si le fichier existe
        if not os.path.exists(path):
            if IS_STREAMLIT_CLOUD:
                st.info("📁 Création d'un modèle de démonstration...")
                # En mode cloud, crée un modèle simple si absent
                create_demo_model(path)
            else:
                st.error(f"Fichier modèle {path} non trouvé")
                return None
        
        # Charge le modèle
        try:
            model_local = load_model(path, custom_objects={'mse': MeanSquaredError()})
        except:
            # Si échec, charge sans custom_objects
            model_local = load_model(path, compile=False)
        
        st.success("✅ Modèle gaze chargé.")
        return model_local
        
    except Exception as e:
        error_msg = str(e)
        if "Layer 'conv1' expected 2 variables" in error_msg:
            st.warning("⚠️ Modèle partiellement chargé - Prédictions basiques activées")
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

# ============================================
# INITIALISATION DES COMPOSANTS
# ============================================
# Initialiser le gestionnaire de caméra
camera_manager = CameraManager()
width, height = camera_manager.get_dimensions()

# Charger le modèle
model = load_gaze_model(MODEL_PATH)
model_enabled = True

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
except Exception as e:
    st.warning(f"⚠️ MediaPipe en mode limité: {str(e)[:50]}")

# ============================================
# CALIBRATION (adaptée pour le nouveau système)
# ============================================
def calibrate_tilt(frames=CALIBRATION_FRAMES):
    """Calibration adaptée au nouveau système de caméra"""
    if IS_STREAMLIT_CLOUD:
        st.info("🔹 Calibration via navigateur...")
        st.info("Veuillez vous positionner face à la caméra pour la calibration")
        
        tilt_values = []
        count = 0
        calibration_placeholder = st.empty()
        
        # Sur Streamlit Cloud, on utilise des captures individuelles
        while count < frames:
            calibration_placeholder.text(f"⏳ Calibration... {count+1}/{frames} (capture en cours)")
            
            # Obtenir un frame via le gestionnaire
            ret, frame = camera_manager.get_frame()
            
            if not ret:
                # Attendre un peu avant de réessayer
                time.sleep(0.1)
                continue
            
            # Traiter le frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)
            
            if not res.multi_face_landmarks:
                calibration_placeholder.text(f"⏳ Calibration... {count+1}/{frames} (attente visage)")
                time.sleep(0.1)
                continue
                
            lm = res.multi_face_landmarks[0].landmark
            tilt, _, _ = angle_between_eyes(lm, LEFT_EYE_IDX, RIGHT_EYE_IDX, width, height)
            tilt_values.append(tilt)
            count += 1
        
        calibration_placeholder.empty()
        center = float(np.mean(tilt_values)) if tilt_values else 0.0
        st.success(f"✅ Calibration terminée. Tilt_center={center:.2f}")
        return center
    
    else:
        # En local, calibration en temps réel
        st.info("🔹 Calibration tilt en cours...")
        tilt_values = []
        count = 0
        calibration_placeholder = st.empty()
        
        while count < frames:
            ret, frame = camera_manager.get_frame()
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

# Initialiser la calibration
tilt_center = 0.0
try:
    tilt_center = calibrate_tilt(CALIBRATION_FRAMES)
except Exception as e:
    st.warning(f"Calibration simplifiée - Tilt_center=0.0 ({str(e)[:50]})")
    tilt_center = 0.0

# ============================================
# DASHBOARD (identique à votre code)
# ============================================
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

# ============================================
# FONCTIONS AUXILIAIRES (identique à votre code)
# ============================================
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
# BOUCLE PRINCIPALE (adaptée au nouveau système)
# ============================================
def main_loop(fig_dashboard=None, st_plot=None, st_frame=None, st_feedback=None):
    global model_enabled, DEBUG
    
    fps_interval = 1.0 / FPS_TARGET
    gaze_queue = deque(maxlen=int(WINDOW_SEC * FPS_TARGET))
    center_queue = deque(maxlen=int(WINDOW_SEC * FPS_TARGET))
    consecutive_eye_closed = 0
    counters = {"center_gaze":0, "left":0, "right":0,
                "eye_closed":0, "head_tilt":0, "unstable":0, "total":0, "no_face":0}
    ear_history = deque(maxlen=5)
    last_dashboard_update_local = 0.0
    demo_counter = 0

    while st.session_state.running:
        loop_t0 = time.time()
        
        # Obtenir un frame via le gestionnaire de caméra
        ret, frame = camera_manager.get_frame()
        
        if not ret:
            # Sur Streamlit Cloud, on attend une capture
            if IS_STREAMLIT_CLOUD:
                time.sleep(0.5)  # Petit délai avant de réessayer
                continue
            else:
                # En local, continuer normalement
                continue
        
        counters["total"] += 1
        demo_counter += 1
        
        # Indicateur de mode
        if IS_STREAMLIT_CLOUD:
            cv2.putText(frame, "STREAMLIT CLOUD - MODE NAVIGATEUR", (50, height - 30), 
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
        
        # Utiliser la fonction get_eye_open_values
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
        
        # Ajout mode
        if IS_STREAMLIT_CLOUD:
            cv2.putText(frame_display, "MODE NAVIGATEUR", (10, height - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        for idx,msg in enumerate(feedback_msgs):
            cv2.putText(frame_display,msg,(10,60+30*idx),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

        st_frame.image(frame_display, channels="BGR")
        st_feedback.text(" | ".join(feedback_msgs))

        # Gestion FPS
        t_elapsed = time.time()-loop_t0
        if t_elapsed<fps_interval: 
            time.sleep(max(0,fps_interval-t_elapsed))

# ============================================
# INTERFACE STREAMLIT
# ============================================
if __name__=="__main__":
    st.title("AI Focus Tracker - Streamlit")
    
    # Avertissement pour Streamlit Cloud
    if IS_STREAMLIT_CLOUD:
        st.warning("""
        ⚠️ **Mode navigateur activé**
        Cette application utilise la caméra de votre navigateur sur Streamlit Cloud.
        - Cliquez sur "Allow" pour autoriser l'accès à la caméra
        - L'analyse se fait en temps réel via des captures régulières
        - Pour une expérience optimale, exécutez l'application localement
        """)
    else:
        st.success("✅ Mode local activé - Utilisation de la webcam système")

    # Initialisation session_state
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'fig_dashboard' not in st.session_state:
        st.session_state.fig_dashboard = make_dashboard()

    # Boutons Start/Stop
    col1, col2 = st.columns(2)
    with col1:
        if st.button("▶️ Start", type="primary", use_container_width=True):
            st.session_state.running = True
            st.rerun()
    with col2:
        if st.button("⏹ Stop", type="secondary", use_container_width=True):
            st.session_state.running = False
            camera_manager.release()
            st.rerun()

    st.info("Status: " + ("**Running**" if st.session_state.running else "**Stopped**"))

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
                cv2.putText(dummy_frame, "Utilisez la caméra de votre navigateur", 
                          (width//2 - 300, height//2 + 100), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 200, 255), 1)
            
            st_frame.image(dummy_frame, channels="BGR")
            
        except NameError:
             st_frame.text("Vidéo arrêtée")
             if IS_STREAMLIT_CLOUD:
                 st_frame.text("Mode navigateur - Prêt à capturer")

        # Message de feedback
        st_feedback.text("Session terminée. Cliquez sur Start pour lancer une nouvelle analyse.")
    
    # Nettoyage à la fermeture
    st.markdown("---")
    st.caption("AI Focus Tracker v1.0 | Compatible Streamlit Cloud & Local")