"""
FASE 4: Interfaz de Usuario - SmoothFlow AI
------------------------------------------------
Interfaz web en Streamlit: elegís tus "estaciones" (moods), cuántas
canciones por bloque y por transición querés, y la app arma la ruta
completa con el motor de la Fase 3, mostrando la curva de energía/tempo.

Cómo correrlo:
    streamlit run src/app/app.py
(Se abre solo en tu navegador en http://localhost:8501)
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import spotipy
import streamlit as st
from spotipy.cache_handler import MemoryCacheHandler
from spotipy.oauth2 import SpotifyOAuth

# Permite importar las funciones de build_route.py y connect_spotify.py sin duplicar código.
sys.path.append(str(Path(__file__).resolve().parents[2]))
from src.models.build_route import (  # noqa: E402
    FEATURE_COLUMNS,
    build_route,
    load_data as load_route_data,
    load_models,
)
from src.data.connect_spotify import SCOPE, get_spotify_client  # noqa: E402

st.set_page_config(page_title="SmoothFlow AI", page_icon="🎧", layout="centered")


# ---------- Carga de datos y modelos (cacheada para no recalcular en cada click) ----------

@st.cache_resource
def get_models():
    return load_models()


@st.cache_data
def get_data():
    return load_route_data()


@st.cache_resource
def get_spotify():
    """Cliente de Spotify autenticado, SIEMPRE con TU cuenta (Opción A: en
    el deploy público nadie más puede crear playlists, solo vos).

    - Local (streamlit run en tu máquina): reusa el login de la Fase 1,
      leyendo el .cache que ya generó connect_spotify.py.
    - Deploy público (Streamlit Community Cloud): no hay .cache en el
      servidor, así que usamos un token guardado a mano en Secrets
      (ver instrucciones en el README, sección "Deploy"). MemoryCacheHandler
      le pasa ese token a spotipy sin necesitar un archivo en disco.
    """
    if "SPOTIFY_TOKEN_INFO" in st.secrets:
        token_info = json.loads(st.secrets["SPOTIFY_TOKEN_INFO"])
        auth_manager = SpotifyOAuth(
            client_id=st.secrets["SPOTIPY_CLIENT_ID"],
            client_secret=st.secrets["SPOTIPY_CLIENT_SECRET"],
            redirect_uri=st.secrets["SPOTIPY_REDIRECT_URI"],
            scope=SCOPE,
            cache_handler=MemoryCacheHandler(token_info=token_info),
            open_browser=False,  # nunca intentar abrir navegador/servidor local acá
        )
        return spotipy.Spotify(auth_manager=auth_manager)

    if len(st.secrets) > 0:
        # Hay Secrets configurados (estamos en un deploy, no en tu compu),
        # pero falta justo SPOTIFY_TOKEN_INFO o está mal pegado. NO caemos
        # al flujo local (que intentaría abrir un navegador/servidor y
        # rompe con errores crípticos tipo "Address already in use") -
        # avisamos claro en vez de eso.
        raise RuntimeError(
            "Falta o está mal pegado el secret SPOTIFY_TOKEN_INFO en "
            "Settings → Secrets de Streamlit Cloud. Revisá que el JSON "
            "esté completo, en una sola línea, y entre comillas SIMPLES "
            "rectas ('...'), no comillas tipográficas ('...')."
        )

    return get_spotify_client()


def create_spotify_playlist(route: pd.DataFrame, playlist_name: str) -> str:
    """Crea una playlist privada en tu cuenta de Spotify con las canciones
    de la ruta generada, en el mismo orden. Devuelve la URL de la playlist."""
    sp = get_spotify()

    # Usamos current_user_playlist_create() (POST /me/playlists) y no
    # user_playlist_create() (POST /users/{id}/playlists): Spotify eliminó
    # ese endpoint viejo para apps en modo Development en su migración de
    # febrero 2026. current_user_playlist_create() es el reemplazo oficial.
    playlist = sp.current_user_playlist_create(
        name=playlist_name,
        public=False,
        description="Generada con SmoothFlow AI: transiciones suaves de mood/tempo/energía.",
    )

    track_ids = route["track_id"].dropna().tolist()
    # Spotify solo acepta hasta 100 canciones por llamada, así que las mandamos en tandas.
    for i in range(0, len(track_ids), 100):
        chunk = track_ids[i:i + 100]
        sp.playlist_add_items(playlist["id"], chunk)

    return playlist["external_urls"]["spotify"]


def label_clusters(kmeans, scaler) -> dict[int, str]:
    """Genera un nombre descriptivo para cada cluster usando el modelo de
    Russell (circumplejo de valencia-activación), el estándar en psicología
    de la música para clasificar estados de ánimo con 2 ejes: valence
    (positivo/negativo) y energy (activación alta/baja):

        Valence alto + Energy alto  -> Feliz / Enérgico
        Valence alto + Energy bajo  -> Relajado / Sereno
        Valence bajo + Energy alto  -> Tenso / Agresivo
        Valence bajo + Energy bajo  -> Melancólico / Triste

    Se le agrega un modificador secundario (Acústico/Electrónico, Bailable,
    o Rap si el speechiness es muy alto) para diferenciar mejor los clusters.
    """
    centers = kmeans.cluster_centers_  # ya están en escala estandarizada (media 0)
    quadrants = {
        (True, True): "Feliz / Enérgico",
        (True, False): "Relajado / Sereno",
        (False, True): "Tenso / Agresivo",
        (False, False): "Melancólico / Triste",
    }

    labels = {}
    for i, center in enumerate(centers):
        scores = dict(zip(FEATURE_COLUMNS, center))
        base = quadrants[(scores["valence"] > 0, scores["energy"] > 0)]

        modifier = None
        if scores["speechiness"] > 0.8:  # muy distintivo cuando aparece
            modifier = "Rap/Hablado"
        elif abs(scores["acousticness"]) > abs(scores["danceability"]):
            modifier = "Acústico" if scores["acousticness"] > 0 else "Electrónico"
        else:
            modifier = "Bailable" if scores["danceability"] > 0 else "Poco Bailable"

        labels[i] = f"{base} · {modifier}"
    return labels


# ---------- Interfaz ----------

st.title("🎧 SmoothFlow AI")
st.caption("Armá una ruta musical con transiciones suaves entre moods.")

scaler, kmeans, knn = get_models()
catalog, my_tracks = get_data()
cluster_labels = label_clusters(kmeans, scaler)
label_to_cluster = {v: k for k, v in cluster_labels.items()}

st.subheader("1. Elegí tus estaciones (en orden)")
selected_labels = st.multiselect(
    "¿Por qué moods querés pasar, en orden?",
    options=list(cluster_labels.values()),
    default=list(cluster_labels.values())[:2] if len(cluster_labels) >= 2 else None,
    help="El orden en que las elijas es el orden en que van a sonar.",
)

st.subheader("2. Configurá el largo de cada tramo")
col1, col2 = st.columns(2)
with col1:
    block_size = st.slider("Canciones por bloque (permanencia)", 2, 8, 4)
with col2:
    bridge_size = st.slider("Canciones por transición (puente)", 1, 5, 2)

generate = st.button("🎼 Generar ruta", type="primary", disabled=len(selected_labels) < 2)

if len(selected_labels) < 2:
    st.info("Elegí al menos 2 estaciones para poder armar una transición entre ellas.")

if "route" not in st.session_state:
    st.session_state.route = None
    st.session_state.stations = None

if generate:
    stations = [label_to_cluster[label] for label in selected_labels]

    with st.spinner("Armando tu ruta..."):
        route = build_route(stations, block_size, bridge_size, catalog, my_tracks, scaler, kmeans, knn)

    st.session_state.route = route
    st.session_state.stations = stations

if st.session_state.route is not None:
    route = st.session_state.route
    stations = st.session_state.stations

    st.subheader("🎵 Tu ruta")

    # Tabla con nombres lindos por fila (bloque N: nombre del mood / transición X→Y)
    def row_label(row):
        if row["role"] == "bloque":
            return f"Bloque: {cluster_labels[stations[int(row['station']) - 1]]}"
        return f"Transición {row['station']}"

    display = route.copy()
    display["Tramo"] = display.apply(row_label, axis=1)
    display["¿Ya la conocías?"] = display["is_known"].map({True: "✅ Sí", False: "🆕 Nueva"}).fillna("🆕 Nueva")
    display = display.rename(columns={"track_name": "Canción", "track_artist": "Artista"})
    st.dataframe(
        display[["Tramo", "Canción", "Artista", "¿Ya la conocías?"]],
        width="stretch",
        hide_index=True,
    )

    # Gráfico de energía y tempo a lo largo de la ruta.
    st.subheader("📈 Curva de energía y tempo")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=route["energy"], mode="lines+markers", name="Energy",
        line=dict(color="#1DB954", width=3),
    ))
    fig.add_trace(go.Scatter(
        y=route["tempo"] / route["tempo"].max(), mode="lines+markers", name="Tempo (normalizado)",
        line=dict(color="#FFA500", width=2, dash="dot"),
    ))
    fig.update_layout(
        xaxis_title="Orden en la playlist",
        yaxis_title="Valor (0-1)",
        template="plotly_dark",
        height=350,
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig, width="stretch")

    csv = route.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Descargar ruta (CSV)", csv, "smoothflow_route.csv", "text/csv")

    n_known = int(display["¿Ya la conocías?"].eq("✅ Sí").sum())
    n_total = len(display)
    st.caption(f"{n_known} de {n_total} canciones ya las conocías, "
               f"{n_total - n_known} son descubrimientos nuevos.")

    st.subheader("3. Creá la playlist en tu cuenta de Spotify")

    # En el deploy público (Streamlit Community Cloud) se configura el
    # secret APP_PASSWORD para que SOLO el dueño del proyecto pueda crear
    # playlists reales (si no, cualquier visitante podría llenar tu cuenta
    # de Spotify de playlists). Corriendo local, sin ese secret, no pide nada.
    owner_password = st.secrets.get("APP_PASSWORD")
    if owner_password:
        entered_password = st.text_input(
            "Esta demo pública solo permite crear la playlist con la "
            "contraseña del dueño del proyecto:",
            type="password",
        )
        unlocked = entered_password != "" and entered_password == owner_password
        if entered_password and not unlocked:
            st.error("Contraseña incorrecta.")
    else:
        unlocked = True

    playlist_name = st.text_input(
        "Nombre de la playlist",
        value="SmoothFlow AI - " + " → ".join(selected_labels),
        disabled=not unlocked,
    )
    create = st.button("🚀 Crear en Spotify", disabled=not unlocked)

    if create:
        try:
            with st.spinner("Creando la playlist en tu cuenta..."):
                playlist_url = create_spotify_playlist(route, playlist_name)
            st.success("¡Listo! Playlist creada en tu cuenta de Spotify.")
            st.markdown(f"👉 [Abrir '{playlist_name}' en Spotify]({playlist_url})")
        except Exception as e:
            st.error(
                "No se pudo crear la playlist. Revisá que tu .env tenga "
                "SPOTIPY_CLIENT_ID, SPOTIPY_CLIENT_SECRET y SPOTIPY_REDIRECT_URI "
                f"configurados. Detalle del error: {e}"
            )