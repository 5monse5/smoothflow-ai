# SmoothFlow AI 🎧

DJ Virtual con Machine Learning: genera transiciones suaves entre géneros/moods
en tus playlists de Spotify, interpolando tempo y energía entre bloques musicales.

## ⚠️ Nota importante sobre la API de Spotify

Spotify deprecó el endpoint `audio-features` en noviembre de 2024 para apps
nuevas. Por eso este proyecto NO pide tempo/energy/etc directamente a la API:
usa un dataset público de referencia (`data/reference_tracks.csv`, ~32.800
canciones) y cruza tus canciones contra él por nombre + artista.

## Estado del proyecto

- [x] Fase 1: Conexión con Spotify + dataset de referencia
- [ ] Fase 2: Clustering (K-Means) y motor de búsqueda (KNN)
- [ ] Fase 3: Lógica de la rampa de transición
- [ ] Fase 4: Interfaz en Streamlit

## Setup

```bash
# 1. Cloná/descomprimí el proyecto y entrá a la carpeta
cd smoothflow-ai

# 2. Creá un entorno virtual (recomendado)
python3 -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# 3. Instalá las dependencias
pip install -r requirements.txt

# 4. Configurá tus credenciales de Spotify
cp .env.example .env
# Editá .env y completá SPOTIPY_CLIENT_ID y SPOTIPY_CLIENT_SECRET
# (los sacás de https://developer.spotify.com/dashboard)
```

## Uso - Fase 1

```bash
# 1. Traé tus canciones desde Spotify (top tracks + playlists guardadas)
python src/data/connect_spotify.py
# Esto genera data/my_tracks.csv

# 2. Cruzalas con el dataset de referencia para obtener audio features
python src/data/match_with_reference_dataset.py
# Esto genera data/my_tracks_with_features.csv
```

## Estructura del proyecto

```
smoothflow-ai/
├── data/
│   └── reference_tracks.csv   # Dataset público con audio features precalculados
├── src/
│   ├── data/
│   │   ├── connect_spotify.py              # Fase 1: conexión y extracción
│   │   └── match_with_reference_dataset.py # Fase 1B: asignar audio features
│   ├── models/    # Fase 2 (clustering, KNN) - próximamente
│   └── app/       # Fase 4 (Streamlit) - próximamente
├── .env.example
├── requirements.txt
└── README.md
```
