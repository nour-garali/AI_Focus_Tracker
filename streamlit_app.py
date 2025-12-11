# streamlit_app.py
import streamlit as st
import numpy as np
import math
import time
import os
from collections import deque
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import random
from PIL import Image, ImageDraw

# -----------------------------
# CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="AI Focus Tracker",
    page_icon="👁️",
    layout="wide"
)

# Paramètres
DURATION_SECONDS = 30
FPS_TARGET = 10  # Réduit pour Streamlit Cloud
WINDOW_SEC = 3
EAR_THRESHOLD = 0.22
EYE_CLOSED_CONSEC_FRAMES = 3
STABILITY_MOVEMENT_THRESH = 25
CALIBRATION_FRAMES = 5  # Très réduit
DASHBOARD_UPDATE_INTERVAL = 1.0  # Plus lent

# Dimensions
WIDTH = 640
HEIGHT = 480

# -----------------------------
# FONCTIONS UTILITAIRES SIMULÉES
# -----------------------------
def euclidean(a, b):
    return math.dist(a, b)

def color_bar_stability(val):
    if val < 30: return "red"
    elif val < 70: return "orange"
    else: return "green"

def color_bar(val):
    if val > 70: return "green"
    elif val > 40: return "orange"
    else: return "red"

# -----------------------------
# SIMULATION DE DONNÉES
# -----------------------------
class DataSimulator:
    def __init__(self):
        self.frame_count = 0
        self.face_position = [WIDTH//2, HEIGHT//2]
        self.face_speed = [1.5, 1.0]
        self.eye_state = "open"
        self.eye_timer = 0
        self.gaze_history = []
        self.concentration = 75
        
    def update(self):
        self.frame_count += 1
        
        # Mettre à jour la position du visage (animation)
        self.face_position[0] += self.face_speed[0]
        self.face_position[1] += self.face_speed[1]
        
        # Rebond sur les bords
        if self.face_position[0] < 100 or self.face_position[0] > WIDTH - 100:
            self.face_speed[0] *= -1
        if self.face_position[1] < 100 or self.face_position[1] > HEIGHT - 100:
            self.face_speed[1] *= -1
        
        # Simulation d'ouverture/fermeture des yeux
        self.eye_timer += 1
        if self.eye_timer > 100:  # Toutes les 100 frames
            self.eye_state = "closed" if self.eye_state == "open" else "open"
            self.eye_timer = 0
        
        # Simulation de la concentration (varie de manière réaliste)
        noise = random.uniform(-0.5, 0.5)
        drift = 0.1 if self.concentration < 50 else -0.1
        self.concentration += drift + noise
        self.concentration = max(20, min(95, self.concentration))
        
        # Simulation de la direction du regard
        gaze_sin = math.sin(self.frame_count * 0.08)
        if gaze_sin > 0.4:
            gaze = "RIGHT"
            gaze_val = 0.8
        elif gaze_sin < -0.4:
            gaze = "LEFT"
            gaze_val = -0.8
        else:
            gaze = "CENTER"
            gaze_val = 0.1
        
        self.gaze_history.append(gaze_val)
        if len(self.gaze_history) > 30:
            self.gaze_history.pop(0)
        
        return {
            "face_detected": random.random() > 0.1,  # 90% du temps
            "eye_state": self.eye_state,
            "gaze": gaze,
            "gaze_val": gaze_val,
            "concentration": self.concentration,
            "face_position": self.face_position.copy(),
            "frame_count": self.frame_count
        }

# -----------------------------
# GÉNÉRATION D'IMAGE SIMULÉE
# -----------------------------
def generate_simulation_frame(sim_data):
    """Génère une image de simulation sans OpenCV"""
    # Créer une image PIL
    img = Image.new('RGB', (WIDTH, HEIGHT), color='black')
    draw = ImageDraw.Draw(img)
    
    face_x, face_y = sim_data["face_position"]
    
    # Dessiner le visage
    # Tête
    head_radius = 100
    draw.ellipse([face_x - head_radius, face_y - 120, 
                  face_x + head_radius, face_y + 120], 
                 fill=(100, 100, 200), outline=(150, 150, 255))
    
    # Yeux
    eye_offset = 40
    eye_radius = 20
    pupil_radius = 8
    
    if sim_data["eye_state"] == "open":
        # Yeux ouverts
        draw.ellipse([face_x - eye_offset - eye_radius, face_y - 30 - eye_radius,
                      face_x - eye_offset + eye_radius, face_y - 30 + eye_radius],
                     fill='white')
        draw.ellipse([face_x + eye_offset - eye_radius, face_y - 30 - eye_radius,
                      face_x + eye_offset + eye_radius, face_y - 30 + eye_radius],
                     fill='white')
        
        # Pupilles (qui suivent le regard)
        pupil_x_offset = 5 * math.sin(sim_data["frame_count"] * 0.2)
        draw.ellipse([face_x - eye_offset + pupil_x_offset - pupil_radius, face_y - 30 - pupil_radius,
                      face_x - eye_offset + pupil_x_offset + pupil_radius, face_y - 30 + pupil_radius],
                     fill='black')
        draw.ellipse([face_x + eye_offset + pupil_x_offset - pupil_radius, face_y - 30 - pupil_radius,
                      face_x + eye_offset + pupil_x_offset + pupil_radius, face_y - 30 + pupil_radius],
                     fill='black')
    else:
        # Yeux fermés
        eye_height = 5
        draw.rectangle([face_x - eye_offset - eye_radius, face_y - 30 - eye_height,
                       face_x - eye_offset + eye_radius, face_y - 30 + eye_height],
                      fill=(150, 150, 150))
        draw.rectangle([face_x + eye_offset - eye_radius, face_y - 30 - eye_height,
                       face_x + eye_offset + eye_radius, face_y - 30 + eye_height],
                      fill=(150, 150, 150))
    
    # Bouche
    mouth_width = 40
    mouth_height = 20
    draw.arc([face_x - mouth_width, face_y + 30 - mouth_height,
              face_x + mouth_width, face_y + 30 + mouth_height],
             0, 180, fill=(200, 100, 100), width=3)
    
    # Ajouter du texte avec PIL
    from PIL import ImageFont
    try:
        font = ImageFont.load_default()
        # Titre
        draw.text((10, 10), "AI FOCUS TRACKER - DEMO", fill=(255, 255, 0), font=font)
        # Informations
        draw.text((10, 30), f"Gaze: {sim_data['gaze']}", fill=(0, 255, 0), font=font)
        draw.text((10, 50), f"Concentration: {sim_data['concentration']:.1f}%", fill=(0, 255, 255), font=font)
        draw.text((10, 70), f"Eyes: {sim_data['eye_state'].upper()}", 
                 fill=(255, 0, 0) if sim_data['eye_state'] == 'closed' else (0, 255, 0), 
                 font=font)
        draw.text((WIDTH - 200, 10), "STREAMLIT CLOUD", fill=(255, 200, 0), font=font)
        draw.text((WIDTH - 200, 30), "Mode: Simulation", fill=(255, 150, 150), font=font)
    except:
        # Fallback si la police ne charge pas
        pass
    
    # Convertir PIL Image en numpy array pour Streamlit
    return np.array(img)

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
        fig.add_trace(go.Indicator(
            mode="gauge+number", 
            value=50,
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "green"},
                'steps': [
                    {'range': [0, 40], 'color': "red"},
                    {'range': [40, 70], 'color': "orange"},
                    {'range': [70, 100], 'color': "green"}
                ]
            }
        ), row=(i//2)+1, col=(i%2)+1)
    
    fig.update_layout(
        height=500, 
        width=700, 
        paper_bgcolor='#2b3e5c', 
        plot_bgcolor='#2b3e5c',
        title_text="Dashboard Concentration Live", 
        title_x=0.5,
        font=dict(color="white", size=12),
        margin=dict(l=20, r=20, t=60, b=20)
    )
    return fig

def update_dashboard(fig, focus, eye_closed_val, face_detected_val, unstable_val):
    eyes_open = 100 - eye_closed_val
    stable = unstable_val
    values = [focus, eyes_open, face_detected_val, stable]
    
    for i, val in enumerate(values):
        fig.data[i].value = val
        
    return fig

# -----------------------------
# BOUCLE PRINCIPALE
# -----------------------------
def main_loop(fig_dashboard, st_plot, st_frame, st_feedback, st_metrics):
    """Boucle principale de simulation"""
    
    # Initialiser le simulateur
    simulator = DataSimulator()
    
    # Historiques
    gaze_queue = deque(maxlen=20)
    concentration_history = deque(maxlen=50)
    
    # Compteurs
    counters = {
        "total_frames": 0,
        "eyes_closed_frames": 0,
        "face_detected_frames": 0,
        "center_gaze_frames": 0,
        "left_gaze_frames": 0,
        "right_gaze_frames": 0
    }
    
    last_update = time.time()
    
    while st.session_state.running:
        start_time = time.time()
        
        # Générer des données simulées
        sim_data = simulator.update()
        counters["total_frames"] += 1
        
        # Mettre à jour les compteurs
        if sim_data["eye_state"] == "closed":
            counters["eyes_closed_frames"] += 1
        
        if sim_data["face_detected"]:
            counters["face_detected_frames"] += 1
        
        if sim_data["gaze"] == "CENTER":
            counters["center_gaze_frames"] += 1
        elif sim_data["gaze"] == "LEFT":
            counters["left_gaze_frames"] += 1
        elif sim_data["gaze"] == "RIGHT":
            counters["right_gaze_frames"] += 1
        
        gaze_queue.append(1 if sim_data["gaze"] == "CENTER" else 0)
        concentration_history.append(sim_data["concentration"])
        
        # Calculer les métriques
        face_detected_pct = (counters["face_detected_frames"] / counters["total_frames"]) * 100
        eyes_open_pct = 100 - (counters["eyes_closed_frames"] / counters["total_frames"]) * 100
        
        # Stabilité simulée
        stability = 70 + 20 * math.sin(sim_data["frame_count"] * 0.03)
        stability = max(30, min(95, stability))
        
        # Focus calculé
        gaze_focus = np.mean(list(gaze_queue)) * 100 if gaze_queue else 50
        focus = (
            0.4 * gaze_focus +
            0.3 * eyes_open_pct +
            0.2 * face_detected_pct +
            0.1 * stability
        )
        
        # Générer l'image de simulation
        frame = generate_simulation_frame(sim_data)
        
        # Mettre à jour le dashboard périodiquement
        current_time = time.time()
        if current_time - last_update > DASHBOARD_UPDATE_INTERVAL:
            update_dashboard(
                fig_dashboard,
                round(focus, 1),
                round(100 - eyes_open_pct, 1),
                round(face_detected_pct, 1),
                round(stability, 1)
            )
            last_update = current_time
        
        # Afficher les éléments
        st_plot.plotly_chart(fig_dashboard, use_container_width=True)
        st_frame.image(frame, caption="Simulation en temps réel")
        
        # Afficher les métriques
        metrics_text = f"""
        **Analyse en cours:**
        - 📊 Focus: {focus:.1f}%
        - 👁️ Yeux ouverts: {eyes_open_pct:.1f}%
        - 🎯 Visage détecté: {face_detected_pct:.1f}%
        - 🎭 Stabilité: {stability:.1f}%
        - 👁️ Regard: {sim_data['gaze']}
        """
        st_metrics.markdown(metrics_text)
        
        # Feedback textuel
        feedback_messages = []
        if sim_data["eye_state"] == "closed":
            feedback_messages.append("⚠️ Yeux fermés détectés")
        if focus < 40:
            feedback_messages.append("📉 Concentration basse")
        if stability < 50:
            feedback_messages.append("🎯 Mouvements détectés")
        
        feedback_text = " | ".join(feedback_messages) if feedback_messages else "✅ Toutes les métriques sont bonnes"
        st_feedback.info(feedback_text)
        
        # Contrôle du FPS
        elapsed = time.time() - start_time
        if elapsed < 1.0 / FPS_TARGET:
            time.sleep(1.0 / FPS_TARGET - elapsed)
        
        # Vérifier la fin de la session
        if st.session_state.session_duration > 0:
            st.session_state.session_duration -= elapsed
            if st.session_state.session_duration <= 0:
                st.session_state.running = False
                st.success("Session terminée!")

# -----------------------------
# INTERFACE PRINCIPALE
# -----------------------------
def main():
    st.title("👁️ AI Focus Tracker - Streamlit Cloud Edition")
    
    st.markdown("""
    <div style='background-color: #2b3e5c; padding: 15px; border-radius: 10px;'>
    <h4 style='color: white; margin: 0;'>⚠️ Mode Démonstration Activé</h4>
    <p style='color: #ccc; margin: 5px 0 0 0;'>
    Cette version fonctionne sur Streamlit Cloud avec des données simulées.<br>
    Pour une version complète avec webcam, exécutez l'application localement.
    </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialiser l'état de session
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'fig_dashboard' not in st.session_state:
        st.session_state.fig_dashboard = make_dashboard()
    if 'session_duration' not in st.session_state:
        st.session_state.session_duration = DURATION_SECONDS
    
    # Sidebar avec contrôles
    with st.sidebar:
        st.header("🎮 Contrôles")
        
        # Sélecteur de durée
        session_duration = st.slider(
            "Durée de la session (secondes)",
            min_value=10,
            max_value=300,
            value=DURATION_SECONDS,
            step=10
        )
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ Démarrer", type="primary", use_container_width=True):
                st.session_state.running = True
                st.session_state.session_duration = session_duration
                st.rerun()
        
        with col2:
            if st.button("⏹️ Arrêter", type="secondary", use_container_width=True):
                st.session_state.running = False
                st.rerun()
        
        st.divider()
        
        st.header("📊 Paramètres")
        
        # Paramètres simulés
        st.slider("Niveau de difficulté", 1, 10, 5)
        st.checkbox("Notifications sonores", False)
        st.checkbox("Enregistrer les données", True)
        
        st.divider()
        
        st.header("ℹ️ À propos")
        st.info("""
        Cette application simule un tracker de concentration:
        
        - 👁️ Suivi du regard (gaze tracking)
        - 😴 Détection de fatigue
        - 📊 Analyse de concentration
        - 🎭 Détection de mouvement
        
        *Les données sont simulées pour la démonstration.*
        """)
    
    # Zone principale
    if not st.session_state.running:
        # Écran d'accueil
        st.markdown("---")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("🎯 Prêt à analyser votre concentration?")
            st.markdown("""
            Cliquez sur **Démarrer** pour lancer une session d'analyse.
            
            **Fonctionnalités simulées:**
            1. 🔍 Détection du visage
            2. 👁️ Suivi du regard
            3. 😴 Détection de clignement
            4. 📊 Score de concentration
            5. 🎭 Analyse de stabilité
            """)
        
        with col2:
            st.image("https://img.icons8.com/color/240/000000/eye-tracking.png", 
                    caption="AI Eye Tracking")
        
        # Dashboard initial
        st.subheader("📈 Dashboard (Données statiques)")
        st.plotly_chart(st.session_state.fig_dashboard, use_container_width=True)
        
        # Image de démonstration
        demo_frame = generate_simulation_frame({
            "face_position": [WIDTH//2, HEIGHT//2],
            "eye_state": "open",
            "gaze": "CENTER",
            "frame_count": 0,
            "concentration": 75
        })
        st.image(demo_frame, caption="Aperçu de la simulation", use_container_width=True)
    
    else:
        # Session en cours
        st.markdown("---")
        
        # Créer les placeholders
        st_plot = st.empty()
        st_frame = st.empty()
        st_metrics = st.empty()
        st_feedback = st.empty()
        
        # Timer
        time_placeholder = st.empty()
        
        # Lancer la boucle principale
        try:
            main_loop(
                st.session_state.fig_dashboard,
                st_plot,
                st_frame,
                st_feedback,
                st_metrics
            )
        except Exception as e:
            st.error(f"Erreur lors de l'exécution: {str(e)}")
            st.session_state.running = False
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
    <p>AI Focus Tracker v1.0 • Streamlit Cloud Edition • Données simulées</p>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------
# POINT D'ENTRÉE
# -----------------------------
if __name__ == "__main__":
    main()