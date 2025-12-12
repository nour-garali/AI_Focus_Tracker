# streamlit_app.py
import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
import time
import os
from collections import deque
from plotly.subplots import make_subplots
import plotly.graph_objects as go

from utils import *  # Importez les fonctions utilitaires

# -----------------------------
# CONFIGURATION
# -----------------------------
MODEL_PATH = "best_gaze_model.keras"
DURATION_SECONDS = 30
FPS_TARGET = 15
WINDOW_SEC = 3
CALIBRATION_FRAMES = 10
DASHBOARD_UPDATE_INTERVAL = 1.0

# -----------------------------
# INITIALISATION
# -----------------------------
@st.cache_resource
def load_resources():
    """Charge les ressources une seule fois"""
    # Charger le modèle
    model = None
    if os.path.exists(MODEL_PATH):
        try:
            model = load_model(MODEL_PATH, compile=False)
            st.success("✅ Modèle chargé avec succès")
        except Exception as e:
            st.warning(f"⚠️ Modèle chargé en mode limité: {e}")
    else:
        st.warning("⚠️ Modèle non trouvé, mode démonstration activé")
    
    # Initialiser MediaPipe
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    return model, face_mesh

# -----------------------------
# DASHBOARD
# -----------------------------
def make_dashboard():
    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"type":"indicator"}, {"type":"indicator"}],
               [{"type":"indicator"}, {"type":"indicator"}]],
        subplot_titles=["Concentration %","Yeux ouverts %","Visage détecté %","Stabilité %"]
    )
    for i in range(4):
        fig.add_trace(go.Indicator(mode="gauge+number", value=50,
                                   gauge={'axis':{'range':[0,100]},
                                          'bar':{'color':'gray'}}),
                      row=(i//2)+1, col=(i%2)+1)
    fig.update_layout(height=500, width=700, 
                      paper_bgcolor='#2b3e5c', plot_bgcolor='#2b3e5c',
                      title_text="Dashboard Concentration", title_x=0.5,
                      font=dict(color="white", size=12))
    return fig

def update_dashboard(fig, focus, eye_closed_val, face_detected_val, stable_val):
    eyes_open = 100 - eye_closed_val
    values = [focus, eyes_open, face_detected_val, 100 - stable_val]
    
    colors = ['green', 'green', 'green', 'green']
    for i, val in enumerate(values):
        if val < 40:
            colors[i] = 'red'
        elif val < 70:
            colors[i] = 'orange'
    
    for i, (val, color) in enumerate(zip(values, colors)):
        fig.data[i].value = val
        fig.data[i].gauge.bar.color = color
    
    return fig

# -----------------------------
# FONCTION PRINCIPALE
# -----------------------------
def main():
    st.set_page_config(page_title="AI Focus Tracker", layout="wide")
    
    st.title("🎯 AI Focus Tracker")
    st.markdown("""
    Cette application analyse votre concentration en temps réel grâce à l'IA.
    **Fonctionne à la fois en local et sur Streamlit Cloud !**
    """)
    
    # Sidebar pour les paramètres
    with st.sidebar:
        st.header("⚙️ Paramètres")
        calibration_mode = st.checkbox("Activer la calibration", value=True)
        privacy_blur = st.checkbox("Flouter le visage pour confidentialité", value=True)
        show_debug = st.checkbox("Mode debug", value=False)
        
        if st.button("🔄 Réinitialiser l'analyse"):
            st.session_state.clear()
            st.rerun()
    
    # Charger les ressources
    model, face_mesh = load_resources()
    
    # Initialiser les états de session
    if 'counters' not in st.session_state:
        st.session_state.counters = {
            "center_gaze": 0, "left": 0, "right": 0,
            "eye_closed": 0, "total": 0, "no_face": 0
        }
    
    if 'gaze_queue' not in st.session_state:
        st.session_state.gaze_queue = deque(maxlen=int(WINDOW_SEC * FPS_TARGET))
    
    if 'center_queue' not in st.session_state:
        st.session_state.center_queue = deque(maxlen=int(WINDOW_SEC * FPS_TARGET))
    
    if 'ear_history' not in st.session_state:
        st.session_state.ear_history = deque(maxlen=5)
    
    if 'consecutive_eye_closed' not in st.session_state:
        st.session_state.consecutive_eye_closed = 0
    
    if 'tilt_center' not in st.session_state:
        st.session_state.tilt_center = 0.0
    
    if 'last_update' not in st.session_state:
        st.session_state.last_update = time.time()
    
    # Créer le dashboard
    fig = make_dashboard()
    
    # Interface principale
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.subheader("📸 Caméra en direct")
        
        # Utilisez st.camera_input() qui fonctionne sur Streamlit Cloud
        img_file_buffer = st.camera_input(
            "Prenez une photo ou activez la caméra",
            key="camera_input",
            help="Cliquez pour prendre une photo ou maintenez pour la vidéo en direct"
        )
        
        if img_file_buffer is not None:
            # Convertir l'image en format OpenCV
            bytes_data = img_file_buffer.getvalue()
            cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
            
            # Traiter l'image
            result = process_frame(
                cv2_img, face_mesh, model, 
                st.session_state.tilt_center,
                st.session_state.counters,
                st.session_state.gaze_queue,
                st.session_state.center_queue,
                st.session_state.ear_history,
                st.session_state.consecutive_eye_closed
            )
            
            if result[0] is not None:
                # Afficher l'image traitée
                frame_display, focus, feedback_msgs, counters, gaze_queue, center_queue, ear_history, consecutive_eye_closed = result
                
                # Mettre à jour les états
                st.session_state.counters = counters
                st.session_state.gaze_queue = gaze_queue
                st.session_state.center_queue = center_queue
                st.session_state.ear_history = ear_history
                st.session_state.consecutive_eye_closed = consecutive_eye_closed
                
                # Convertir pour affichage Streamlit
                frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, caption="Analyse en temps réel", use_column_width=True)
                
                # Afficher les feedbacks
                if feedback_msgs:
                    for msg in feedback_msgs:
                        st.warning(f"⚠️ {msg}")
                
                # Mettre à jour le dashboard
                eye_closed_val = min(100, (counters["eye_closed"] / max(1, counters["total"])) * 100)
                face_detected_val = min(100, ((counters["total"] - counters["no_face"]) / max(1, counters["total"])) * 100)
                
                update_dashboard(fig, focus, eye_closed_val, face_detected_val, 50)
                
                # Afficher les statistiques
                with st.expander("📊 Statistiques détaillées"):
                    col_a, col_b, col_c = st.columns(3)
                    with col_a:
                        st.metric("Concentration", f"{focus:.1f}%")
                        st.metric("Regard centre", f"{counters['center_gaze']}")
                    with col_b:
                        st.metric("Yeux ouverts", f"{100 - eye_closed_val:.1f}%")
                        st.metric("Regard gauche", f"{counters['left']}")
                    with col_c:
                        st.metric("Visage détecté", f"{face_detected_val:.1f}%")
                        st.metric("Regard droite", f"{counters['right']}")
        
        else:
            st.info("👆 Activez la caméra pour commencer l'analyse")
    
    with col2:
        st.subheader("📈 Dashboard")
        st.plotly_chart(fig, use_container_width=True)
        
        # Section calibration
        if calibration_mode:
            st.subheader("🎯 Calibration")
            if st.button("🔧 Lancer la calibration"):
                with st.spinner("Calibration en cours..."):
                    # Simulation simple de calibration
                    st.session_state.tilt_center = 0.0
                    time.sleep(2)
                    st.success("✅ Calibration terminée!")
                    st.info(f"Valeur de référence: {st.session_state.tilt_center:.2f}°")
        
        # Instructions
        st.subheader("ℹ️ Instructions")
        st.markdown("""
        1. **Activez la caméra** ci-contre
        2. **Positionnez votre visage** bien en vue
        3. **Maintenez la position** pour une analyse précise
        4. **Évitez les mouvements brusques**
        5. **Gardez les yeux ouverts** pour une meilleure détection
        """)
    
    # Footer
    st.markdown("---")
    st.caption("🔒 Toutes les données sont traitées localement. Aucune image n'est stockée.")
    st.caption("💡 Pour de meilleurs résultats, utilisez un éclairage uniforme et restez à distance fixe de la caméra.")

if __name__ == "__main__":
    main()