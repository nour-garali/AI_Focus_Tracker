# app_streamlit.py
import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import math
import time
from collections import deque
import traceback

# streamlit-webrtc
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, ClientSettings

# TensorFlow/Keras (chargement modèle)
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# -----------------------------
# CONFIGURATION (modifiable)
# -----------------------------
MODEL_PATH = "best_gaze_model.keras"
DURATION_SECONDS = 30
FPS_TARGET = 15
WINDOW_SEC = 3
EAR_THRESHOLD = 0.22
EYE_CLOSED_CONSEC_FRAMES = 3
STABILITY_MOVEMENT_THRESH = 25
PRIVACY_BLUR = True       # tu as demandé garder le flou
CALIBRATION_FRAMES = 50
DASHBOARD_UPDATE_INTERVAL = 0.5

LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]

DEBUG = False

# Client settings for webRTC - allows camera capture from browser
CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
)

# -----------------------------
# UTILITAIRES (identiques à ton code)
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
# DASHBOARD (Plotly) - identique
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
# Load model (non bloquant)
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_gaze_model_cached(path):
    try:
        model_local = load_model(path, custom_objects={'mse': MeanSquaredError()})
        return model_local
    except Exception as e:
        st.warning(f"Erreur chargement modèle : {e}. Mode modèle désactivé.")
        return None

# charger modèle (sera accessible pour l'instance transformer aussi)
model = load_gaze_model_cached(MODEL_PATH)
model_enabled_global = True if model is not None else False

# -----------------------------
# VideoTransformer : place ici TOUTE TA LOGIQUE (adaptée à traitement par frame)
# -----------------------------
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        # initialise Mediapipe et files/datas similaires à ta boucle
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                                   refine_landmarks=True, min_detection_confidence=0.5,
                                                   min_tracking_confidence=0.5)
        # Stats & queues
        self.gaze_queue = deque(maxlen=int(WINDOW_SEC * FPS_TARGET))
        self.center_queue = deque(maxlen=int(WINDOW_SEC * FPS_TARGET))
        self.ear_history = deque(maxlen=5)
        self.counters = {"center_gaze":0, "left":0, "right":0,
                         "eye_closed":0, "head_tilt":0, "unstable":0, "total":0, "no_face":0}
        self.consecutive_eye_closed = 0
        self.model = model  # peut être None
        self.model_enabled = model_enabled_global
        # metrics exposées pour UI polling
        self.focus = 0.0
        self.eye_closed_val = 0.0
        self.face_detected_val = 0.0
        self.unstable_val = 0.0
        # tilt center calibré (sera calibré par une première phase)
        self.tilt_center = 0.0
        self.calibrated = False
        self.last_dashboard_update = 0.0
        self.demo_counter = 0

    def calibrate(self, frames=CALIBRATION_FRAMES):
        # Effectue calibration sur les premières frames captées automatiquement
        vals = []
        got = 0
        # on attend que le pipeline appelle recv plusieurs fois ; on se contente de marquer calibré plus tard
        self.tilt_values_tmp = {"needed": frames, "vals": [], "got": 0}
        # la logique de remplissage se fait dans recv lors du premier N frames

    def recv(self, frame):
        """
        frame: av.VideoFrame
        retourne: av.VideoFrame
        """
        try:
            img = frame.to_ndarray(format="bgr24")  # numpy array BGR
            self.demo_counter += 1
            height, width, _ = img.shape

            # --- Privacy blur if enabled
            frame_display = cv2.GaussianBlur(img, (51,51), 0) if PRIVACY_BLUR else img.copy()

            # Convert to RGB for MediaPipe
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            res = self.face_mesh.process(rgb)

            # If we are in calibration filling mode, collect tilt values
            if not self.calibrated:
                # try to collect CALIBRATION_FRAMES tilt samples
                if res.multi_face_landmarks:
                    lm = res.multi_face_landmarks[0].landmark
                    tilt, _, _ = angle_between_eyes(lm, LEFT_EYE_IDX, RIGHT_EYE_IDX, width, height)
                    if not hasattr(self, "tilt_values_tmp"):
                        self.tilt_values_tmp = {"needed": CALIBRATION_FRAMES, "vals": [], "got": 0}
                    self.tilt_values_tmp["vals"].append(tilt)
                    self.tilt_values_tmp["got"] += 1
                    if self.tilt_values_tmp["got"] >= self.tilt_values_tmp["needed"]:
                        self.tilt_center = float(np.mean(self.tilt_values_tmp["vals"])) if self.tilt_values_tmp["vals"] else 0.0
                        self.calibrated = True
                # overlay text during calibration
                cv2.putText(frame_display, "Calibration...", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                # return frame while calibrating
                return_frame = frame_display
            else:
                # --- No face
                if not res.multi_face_landmarks:
                    self.counters["total"] += 1
                    self.counters["no_face"] += 1
                    self.counters["eye_closed"] += 1
                    self.gaze_queue.append(0)
                    self.eye_closed_val = min(100, (self.counters["eye_closed"]/self.counters["total"])*100 if self.counters["total"]>0 else 0.0)
                    self.face_detected_val = min(100, ((self.counters["total"]-self.counters["no_face"])/self.counters["total"])*100 if self.counters["total"]>0 else 0.0)
                    self.unstable_val = 0
                    self.focus = 0.0
                    cv2.putText(frame_display, "No face detected", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                    return_frame = frame_display
                else:
                    # Face detected -> perform full processing (keeps your logic)
                    self.counters["total"] += 1
                    lm = res.multi_face_landmarks[0].landmark
                    xs_all = [lm[i].x*width for i in range(len(lm))]
                    ys_all = [lm[i].y*height for i in range(len(lm))]
                    x_min, y_min = max(0,int(min(xs_all)-10)), max(0,int(min(ys_all)-10))
                    x_max, y_max = min(width-1,int(max(xs_all)+10)), min(height-1,int(max(ys_all)+10))
                    # crop face_roi safely
                    x_min = max(0, x_min); y_min = max(0, y_min)
                    x_max = min(width, x_max); y_max = min(height, y_max)
                    face_roi = img[y_min:y_max, x_min:x_max]
                    if face_roi.size == 0:
                        # fallback if empty ROI
                        face_roi = img.copy()

                    # Model predict (same as ton code)
                    pred = 0.0
                    if self.model_enabled and self.model is not None:
                        try:
                            img_res = cv2.resize(face_roi, (64,64)) / 255.0
                            pred = float(self.model.predict(np.expand_dims(img_res,0), verbose=0)[0][0])
                        except Exception:
                            pred = 0.0
                    else:
                        # simulation if no model
                        pred = math.sin(self.demo_counter * 0.1) * 0.8

                    if pred > 0.5:
                        gaze = "RIGHT"; self.counters["right"] += 1
                    elif pred < -0.5:
                        gaze = "LEFT"; self.counters["left"] += 1
                    else:
                        gaze = "CENTER"; self.counters["center_gaze"] += 1
                    self.gaze_queue.append(pred)

                    # Eyes (EAR logic)
                    ear_left = eye_aspect_ratio(lm, LEFT_EYE_IDX, width, height)
                    ear_right = eye_aspect_ratio(lm, RIGHT_EYE_IDX, width, height)
                    ear = (ear_left + ear_right)/2.0
                    current_tilt, _, _ = angle_between_eyes(lm, LEFT_EYE_IDX, RIGHT_EYE_IDX, width, height)
                    tilt_delta = abs(current_tilt - self.tilt_center)
                    dynamic_ear_threshold = EAR_THRESHOLD + min(0.07, tilt_delta * 0.003)
                    iris_visible = False
                    try:
                        left_upper = (lm[159].x*width, lm[159].y*height)
                        left_lower = (lm[145].x*width, lm[145].y*height)
                        right_upper = (lm[386].x*width, lm[386].y*height)
                        right_lower = (lm[374].x*width, lm[374].y*height)
                        eye_open_left = euclidean(left_upper, left_lower)
                        eye_open_right = euclidean(right_upper, right_lower)
                        iris_left_y = lm[468].y*height if len(lm)>468 else None
                        iris_right_y = lm[473].y*height if len(lm)>473 else None
                        if iris_left_y is not None and iris_right_y is not None and eye_open_left>2.5 and eye_open_right>2.5:
                            iris_visible=True
                    except:
                        iris_visible=False

                    self.ear_history.append(ear)
                    ear_smoothed = float(np.mean(self.ear_history)) if len(self.ear_history)>0 else ear
                    eyes_closed_detected = (ear_smoothed < dynamic_ear_threshold) and (not iris_visible)
                    if eyes_closed_detected:
                        self.consecutive_eye_closed += 1
                    else:
                        self.consecutive_eye_closed = 0
                    eye_closed_flag = (self.consecutive_eye_closed >= EYE_CLOSED_CONSEC_FRAMES)
                    if eye_closed_flag:
                        self.counters["eye_closed"] += 1

                    # Stability
                    center_pt = ((x_min+x_max)/2, (y_min+y_max)/2)
                    self.center_queue.append(center_pt)
                    unstable_flag=False
                    instability_score=0
                    if len(self.center_queue)>=3:
                        var_x=np.var([p[0] for p in self.center_queue])
                        var_y=np.var([p[1] for p in self.center_queue])
                        movement=math.sqrt(var_x + var_y)
                        if movement<5:
                            instability_score=20
                            unstable_flag=True
                        elif movement>STABILITY_MOVEMENT_THRESH:
                            instability_score=100
                        else:
                            instability_score=int((movement/STABILITY_MOVEMENT_THRESH)*100)
                    self.unstable_val = instability_score

                    # Focus
                    gaze_focus_smoothed = np.mean([1 if abs(g)<0.5 else 0 for g in self.gaze_queue])*100
                    self.eye_closed_val = min(100,(self.counters["eye_closed"]/self.counters["total"])*100 if self.counters["total"]>0 else 0.0)
                    self.face_detected_val = min(100,((self.counters["total"]-self.counters["no_face"])/self.counters["total"])*100 if self.counters["total"]>0 else 0.0)
                    self.focus = (0.4*gaze_focus_smoothed + 0.2*(100-self.eye_closed_val) + 0.2*self.face_detected_val +0.2*(100-self.unstable_val))
                    self.focus = max(0.0, min(100.0, self.focus))

                    # Draw annotations (identical style)
                    cv2.rectangle(frame_display, (x_min,y_min), (x_max,y_max), (0,255,0), 2)
                    cv2.putText(frame_display, f"Gaze:{gaze} (Model {'ON' if self.model_enabled else 'OFF'})", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0),2)
                    if eye_closed_flag:
                        cv2.putText(frame_display, "EYES CLOSED", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
                    if unstable_flag:
                        cv2.putText(frame_display, "TOO STABLE", (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)

                    # Feedback msgs (small list)
                    # (we don't maintain the exact list variable, but overlay same messages)
                    return_frame = frame_display

            # Return av.VideoFrame
            import av
            return av.VideoFrame.from_ndarray(return_frame, format="bgr24")

        except Exception as e:
            # On exception, return original frame to avoid breaking the stream
            try:
                import av
                return av.VideoFrame.from_ndarray(frame.to_ndarray(format="bgr24"), format="bgr24")
            except Exception:
                # last resort: black frame
                black = np.zeros((480,640,3), dtype=np.uint8)
                import av
                return av.VideoFrame.from_ndarray(black, format="bgr24")

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="AI Focus Tracker", layout="wide")
st.title("AI Focus Tracker - Streamlit (WebRTC)")

# placeholders
st_plot = st.empty()
st_frame = st.empty()
st_feedback = st.empty()

# dashboard initial
fig_dashboard = make_dashboard()
st_plot.plotly_chart(fig_dashboard, use_container_width=True)

# Start/Stop controls (session state)
if 'webrtc_ctx' not in st.session_state:
    st.session_state.webrtc_ctx = None
if 'running' not in st.session_state:
    st.session_state.running = False

col1, col2 = st.columns(2)
with col1:
    if st.button("▶️ Start"):
        st.session_state.running = True
with col2:
    if st.button("⏹ Stop"):
        st.session_state.running = False
        # stop webRTC if exists
        if st.session_state.webrtc_ctx is not None:
            try:
                st.session_state.webrtc_ctx.stop()
            except:
                pass

# When starting, create the webrtc streamer
if st.session_state.running and (st.session_state.webrtc_ctx is None or not st.session_state.webrtc_ctx.state.playing):
    # create a fresh transformer instance
    transformer_factory = VideoTransformer
    ctx = webrtc_streamer(
        key="ai-focus-tracker",
        client_settings=CLIENT_SETTINGS,
        video_transformer_factory=transformer_factory,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True
    )
    st.session_state.webrtc_ctx = ctx

# Poll the transformer's metrics and update dashboard
def ui_poll_and_update():
    ctx = st.session_state.get("webrtc_ctx", None)
    if ctx is None:
        return
    # poll while playing
    poll_t0 = time.time()
    try:
        transformer = ctx.video_transformer
    except Exception:
        transformer = None
    if transformer is None:
        return
    # update dashboard at a rate (non-blocking for streamlit)
    # We'll run a short polling loop but avoid blocking main thread too long.
    # This function is triggered on each rerun; to keep UI responsive we only update once per call.
    try:
        # read metrics
        focus = getattr(transformer, "focus", 0.0)
        eye_closed_val = getattr(transformer, "eye_closed_val", 0.0)
        face_detected_val = getattr(transformer, "face_detected_val", 0.0)
        unstable_val = getattr(transformer, "unstable_val", 0.0)
        # update chart object
        update_dashboard(fig_dashboard, round(focus,2), round(eye_closed_val,2), round(face_detected_val,2), round(unstable_val,2))
        st_plot.plotly_chart(fig_dashboard, use_container_width=True)
        # Feedback text
        st_feedback.text(f"Focus: {focus:.1f}% | EyesClosed%: {eye_closed_val:.1f}% | FaceDetected%: {face_detected_val:.1f}% | Stable%: {unstable_val:.1f}%")
    except Exception as e:
        # ignore transient errors
        pass

# Poll UI on each rerun
ui_poll_and_update()

# Footer message
if not st.session_state.running:
    st.info("Status: Stopped. Click Start to begin (la caméra sera demandée par le navigateur).")
else:
    st.success("Status: Running (autorisez la caméra dans le navigateur)")

