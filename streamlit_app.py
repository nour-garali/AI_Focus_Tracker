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

model = load_gaze_model(MODEL_PATH)
model_enabled = model is not None

# -----------------------------
# CAMERA & MEDIAPIPE
# -----------------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    st.error("Impossible d'ouvrir la caméra")
    st.stop()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                  refine_landmarks=True, min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

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
        fig.data[i].gauge.bar.color = color_bar_stability(val) if i==3 else color_bar(val)
    return fig

# -----------------------------
# STREAMLIT APP
# -----------------------------
st.title("AI Focus Tracker - Streamlit")

# Session state
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

# Placeholders
st_plot = st.empty()
st_frame = st.empty()
st_feedback = st.empty()
st_plot.plotly_chart(st.session_state.fig_dashboard)

# -----------------------------
# MAIN LOOP OPTIMIZED
# -----------------------------
def main_loop():
    fps_interval = 1.0 / FPS_TARGET
    gaze_queue = deque(maxlen=int(WINDOW_SEC * FPS_TARGET))
    center_queue = deque(maxlen=int(WINDOW_SEC * FPS_TARGET))
    consecutive_eye_closed = 0
    counters = {"center_gaze":0, "left":0, "right":0,
                "eye_closed":0, "head_tilt":0, "unstable":0, "total":0, "no_face":0}
    ear_history = deque(maxlen=5)
    last_dashboard_update = 0

    frame_count = 0

    while st.session_state.running:
        loop_t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            continue
        frame_count += 1
        counters["total"] += 1
        frame_display = cv2.GaussianBlur(frame,(51,51),0) if PRIVACY_BLUR else frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        feedback_msgs = []

        # ----------- No face
        if not res.multi_face_landmarks:
            counters["no_face"] += 1
            counters["eye_closed"] +=1
            eye_closed_val = min(100,(counters["eye_closed"]/counters["total"])*100)
            face_detected_val = min(100,((counters["total"]-counters["no_face"])/counters["total"])*100)
            unstable_val = 0
            focus = 0
            gaze_queue.append(0)
            feedback_msgs.append("No face detected")
        else:
            lm = res.multi_face_landmarks[0].landmark
            xs_all = [lm[i].x*width for i in range(len(lm))]
            ys_all = [lm[i].y*height for i in range(len(lm))]
            x_min, y_min = max(0,int(min(xs_all)-10)), max(0,int(min(ys_all)-10))
            x_max, y_max = min(width-1,int(max(xs_all)+10)), min(height-1,int(max(ys_all)+10))
            face_roi = frame[y_min:y_max, x_min:x_max]

            if PRIVACY_BLUR:
                h_roi, w_roi, _ = face_roi.shape
                h_disp = y_max - y_min
                w_disp = x_max - x_min
                h_min = min(h_roi, h_disp)
                w_min = min(w_roi, w_disp)
                face_roi = face_roi[:h_min, :w_min]
                frame_display[y_min:y_min+h_min, x_min:x_min+w_min] = face_roi

            # Gaze
            pred = 0.0
            if model_enabled:
                try:
                    img = cv2.resize(face_roi,(64,64))/255.0
                    pred = float(model.predict(np.expand_dims(img,0), verbose=0)[0][0])
                except:
                    pred = 0.0
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
            ear_history.append(ear)
            ear_smoothed = float(np.mean(ear_history))
            eyes_closed_detected = (ear_smoothed < dynamic_ear_threshold)
            if eyes_closed_detected:
                consecutive_eye_closed += 1
            else:
                consecutive_eye_closed = 0
            eye_closed_flag = (consecutive_eye_closed >= EYE_CLOSED_CONSEC_FRAMES)
            if eye_closed_flag:
                counters["eye_closed"] += 1
                feedback_msgs.append("Eyes Closed")

            # Stability
            center = ((x_min+x_max)/2, (y_min+y_max)/2)
            center_queue.append(center)
            unstable_val = 0
            if len(center_queue)>=3:
                var_x=np.var([p[0] for p in center_queue])
                var_y=np.var([p[1] for p in center_queue])
                movement=math.sqrt(var_x + var_y)
                if movement<5: 
                    unstable_val=20
                    feedback_msgs.append("Too stable")
                elif movement>STABILITY_MOVEMENT_THRESH:
                    unstable_val=100
                else:
                    unstable_val=int((movement/STABILITY_MOVEMENT_THRESH)*100)

            # Focus calculation
            gaze_focus_smoothed = np.mean([1 if abs(g)<0.5 else 0 for g in gaze_queue])*100
            eye_closed_val = min(100,(counters["eye_closed"]/counters["total"])*100)
            face_detected_val = min(100,((counters["total"]-counters["no_face"])/counters["total"])*100)
            focus = (0.4*gaze_focus_smoothed + 0.2*(100-eye_closed_val) + 0.2*face_detected_val +0.2*(100-unstable_val))
            focus = max(0.0, min(100.0, focus))

            # Draw feedback on frame
            cv2.rectangle(frame_display, (x_min,y_min), (x_max,y_max), (0,255,0), 2)
            cv2.putText(frame_display,f"Gaze:{gaze} (Model {'ON' if model_enabled else 'OFF'})",(10,30),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            for idx,msg in enumerate(feedback_msgs):
                cv2.putText(frame_display,msg,(10,60+30*idx),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

        # -------------------------
        # Update dashboard every DASHBOARD_UPDATE_INTERVAL
        # -------------------------
        if time.time() - last_dashboard_update > DASHBOARD_UPDATE_INTERVAL:
            update_dashboard(st.session_state.fig_dashboard, round(focus,2), round(eye_closed_val,2), round(face_detected_val,2), round(unstable_val,2))
            st_plot.plotly_chart(st.session_state.fig_dashboard)
            last_dashboard_update = time.time()

        # Update frame & feedback every 2 frames
        if frame_count % 2 == 0:
            st_frame.image(frame_display, channels="BGR")
            st_feedback.text(" | ".join(feedback_msgs))

        # Control FPS
        t_elapsed = time.time()-loop_t0
        if t_elapsed < fps_interval:
            time.sleep(fps_interval - t_elapsed)

# -----------------------------
# LANCEMENT DU MAIN LOOP
# -----------------------------
if st.session_state.running:
    main_loop()
else:
    # Affichage image session arrêtée
    dummy_frame = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.putText(dummy_frame, "SESSION ARRÊTÉE", (width//2 - 200, height//2),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 200, 200), 3)
    st_frame.image(dummy_frame, channels="BGR")
    st_feedback.text(" | Session terminée. Cliquez sur Start pour lancer une nouvelle analyse.")
