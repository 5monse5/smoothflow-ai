# SmoothFlow AI 🎧

DJ Virtual con Machine Learning: genera transiciones suaves entre géneros/moods
en tus playlists de Spotify, interpolando tempo y energía entre bloques musicales,
y te deja crear la playlist resultante directamente en tu cuenta de Spotify.

🔗 **Probalo en vivo:** [smoothflow-ai.streamlit.app](https://smoothflow-ai.streamlit.app/)
(la generación de rutas es libre para cualquiera; crear la playlist real en
Spotify está protegido con contraseña, ver sección "Deploy" más abajo)

## ⚠️ Notas importantes sobre la API de Spotify

- Spotify deprecó el endpoint `audio-features` en noviembre de 2024 para apps
  nuevas. Por eso este proyecto NO pide tempo/energy/etc directamente a la API:
  usa un dataset público de referencia (`data/reference_tracks.csv`, ~26.230
  canciones) y cruza tus canciones contra él por nombre + artista (matching
  difuso con `rapidfuzz`).
- Spotify migró varios endpoints en febrero de 2026 para apps en modo
  Development (incluido el de creación de playlists). El proyecto ya usa los
  endpoints nuevos (`spotipy`'s `current_user_playlist_create`, etc.), así que
  esto no debería requerir ninguna acción de tu parte.

## Estado del proyecto

- [x] Fase 1: Conexión con Spotify + dataset de referencia
- [x] Fase 1B: Cruce de tus canciones con el dataset (audio features)
- [x] Fase 2: Clustering (K-Means, K=10) y motor de búsqueda (KNN)
- [x] Fase 3: Lógica de la rampa de transición
- [x] Fase 4: Interfaz en Streamlit + creación de playlist real en Spotify

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

## Uso

```bash
# 1. Traé tus canciones desde Spotify (top tracks + tus playlists propias)
python src/data/connect_spotify.py
# Esto genera data/my_tracks.csv

# 2. Cruzalas con el dataset de referencia para obtener audio features
python src/data/match_with_reference_dataset.py
# Esto genera data/my_tracks_with_features.csv

# 3. Entrená el clustering (K-Means, K=10) y el motor de búsqueda KNN
python src/models/train_clustering.py
# Esto genera data/models/*.pkl y data/reference_tracks_clustered.csv

# 4. Levantá la interfaz
python -m streamlit run src/app/app.py
# Se abre en http://localhost:8501
```

Desde la interfaz podés:
- Elegir tus "estaciones" (moods) en el orden que quieras.
- Configurar cuántas canciones va cada bloque de permanencia y cada transición.
- Generar la ruta, ver la tabla y el gráfico de energía/tempo.
- Descargar la ruta en CSV.
- Crear la playlist resultante directamente en tu cuenta de Spotify (privada
  por default) con un botón.

## Cómo funciona (resumen técnico)

- **K-Means (K=10)** agrupa las ~26.230 canciones del catálogo en 10
  "micro-moods", entrenado sobre audio features estandarizados
  (danceability, energy, valence, acousticness, tempo, loudness, speechiness).
- Cada mood se etiqueta automáticamente con nombres descriptivos usando el
  **modelo de Russell** (circumplejo valencia-activación), el estándar en
  psicología de la música.
- **KNN** busca, para cada punto interpolado entre dos moods, la canción real
  más cercana en todo el catálogo — así se arman las "canciones puente" de
  las transiciones, favoreciendo el descubrimiento de música nueva.
- Los bloques de permanencia priorizan tus canciones conocidas con un
  "descuento de familiaridad" (no absoluto), para balancear lo conocido con
  el descubrimiento.

## Deploy

La app está desplegada en Streamlit Community Cloud. Como la API de Spotify
limita a pocos usuarios autorizados por app en modo Development, el deploy
público usa siempre la cuenta del dueño del proyecto para crear playlists
reales (protegido con contraseña) — cualquier visitante puede generar y ver
rutas libremente, pero solo el dueño puede efectivamente crear la playlist
en Spotify. Si cloná este repo y corrés la app en tu propia máquina con tus
propias credenciales (ver "Setup" arriba), no tenés esa restricción: podés
crear playlists en tu cuenta sin ninguna contraseña.

## Estructura del proyecto

```
smoothflow-ai/
├── data/
│   ├── reference_tracks.csv            # Dataset público con audio features precalculados
│   ├── reference_tracks_clustered.csv  # Igual al anterior + columna 'cluster'
│   ├── my_tracks.csv                   # Tus canciones (de connect_spotify.py)
│   ├── my_tracks_with_features.csv     # Tus canciones + audio features
│   └── models/                         # scaler.pkl, kmeans.pkl, knn.pkl
├── src/
│   ├── data/
│   │   ├── connect_spotify.py              # Fase 1: conexión y extracción
│   │   └── match_with_reference_dataset.py # Fase 1B: asignar audio features
│   ├── models/
│   │   ├── train_clustering.py    # Fase 2: K-Means (K=10) + KNN
│   │   └── build_route.py         # Fase 3: lógica de la ruta/transiciones
│   └── app/
│       └── app.py                 # Fase 4: interfaz Streamlit
├── .env.example
├── requirements.txt
└── README.md
```