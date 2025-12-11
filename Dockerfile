# Utilise l'image Python 3.11 (la version qui fonctionne chez vous)
FROM python:3.11-slim

# Définit le répertoire de travail
WORKDIR /app

# Copie le requirements.txt et l'installe
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copie le reste de votre application
COPY . .

# Commande pour lancer Streamlit
ENTRYPOINT ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.enableCORS=false"]