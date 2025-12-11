# Dockerfile
FROM python:3.11-slim

# Installer dépendances système pour OpenCV et Mediapipe
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copier app
WORKDIR /app
COPY . /app

# Installer Python packages
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Exposer le port Streamlit
EXPOSE 8501

# Commande pour lancer Streamlit
CMD ["streamlit", "run", "app_streamlit.py", "--server.port=8501", "--server.address=0.0.0.0"]
