"""
FASE 1: Conexión y Extracción de Datos - SmoothFlow AI
--------------------------------------------------------
IMPORTANTE: Spotify deprecó el endpoint /audio-features en noviembre de 2024
para apps nuevas. Por eso este script YA NO pide tempo/energy/etc directamente
a la API - eso lo resolvemos en la Fase 1B (match_with_reference_dataset.py)
cruzando tus canciones con un dataset público de referencia.

Este script:
1. Se autentica con tu cuenta de Spotify (OAuth 2.0).
2. Extrae tus Top Tracks (en 3 rangos de tiempo) y las canciones de TUS
   playlists (las que vos creaste - las de otros curadores no son accesibles).
3. Guarda la metadata básica (nombre, artista, id) en data/my_tracks.csv.

Cómo correrlo:
    1. Completá el archivo .env con tus credenciales (ver .env.example).
    2. pip install -r requirements.txt
    3. python src/data/connect_spotify.py
    4. Se va a abrir tu navegador pidiéndote loguearte y autorizar la app.
       Después de autorizar, te va a redirigir a una URL localhost que
       fallará al cargar (es normal) - copiá esa URL completa y pegala
       en la consola cuando te la pida spotipy.
"""

import os
import time
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv

load_dotenv()

# Permisos que necesitamos: leer top tracks, leer playlists guardadas,
# y (para más adelante, Fase 4) crear playlists en tu cuenta.
SCOPE = "user-top-read playlist-read-private playlist-modify-public playlist-modify-private"


def get_spotify_client() -> spotipy.Spotify:
    """Crea y devuelve un cliente autenticado de Spotify."""
    auth_manager = SpotifyOAuth(
        client_id=os.getenv("SPOTIPY_CLIENT_ID"),
        client_secret=os.getenv("SPOTIPY_CLIENT_SECRET"),
        redirect_uri=os.getenv("SPOTIPY_REDIRECT_URI"),
        scope=SCOPE,
    )
    return spotipy.Spotify(auth_manager=auth_manager)


def get_top_tracks(sp: spotipy.Spotify, limit: int = 50) -> list[dict]:
    """Trae tus canciones más escuchadas en los 3 rangos de tiempo que ofrece Spotify
    (últimas ~4 semanas, ~6 meses, y varios años), para maximizar la cantidad de
    canciones únicas. A diferencia de las playlists de otros curadores, esto
    siempre es accesible porque es 100% tu propia data.
    """
    tracks = []
    for time_range in ["short_term", "medium_term", "long_term"]:
        results = sp.current_user_top_tracks(limit=limit, time_range=time_range)
        tracks.extend(results["items"])
    return tracks


def get_saved_playlist_tracks(sp: spotipy.Spotify, max_playlists: int = 20) -> list[dict]:
    """Trae canciones de TUS playlists (las que vos creaste), hasta max_playlists.

    Nota: las playlists de otros curadores que solo seguís (no creaste) no se
    pueden leer por API en apps nuevas desde los cambios de Spotify de nov. 2024,
    así que las filtramos de antemano en vez de intentar y fallar.
    """
    tracks = []
    me = sp.current_user()
    my_user_id = me["id"]

    playlists = sp.current_user_playlists(limit=max_playlists)
    own_playlists = [p for p in playlists["items"] if p["owner"]["id"] == my_user_id]

    print(f"   ({len(own_playlists)} de {len(playlists['items'])} playlists guardadas son tuyas)")

    for playlist in own_playlists:
        playlist_id = playlist["id"]
        playlist_name = playlist.get("name", playlist_id)
        try:
            results = sp.playlist_items(playlist_id, additional_types=["track"])
        except spotipy.exceptions.SpotifyException as e:
            print(f"   ⚠️  Salteando playlist '{playlist_name}' (no accesible por API: {e.http_status})")
            continue

        for item in results["items"]:
            track = item.get("track")
            if track and track.get("id"):
                tracks.append(track)
        time.sleep(0.1)  # para no saturar la API

    return tracks


def build_dataframe(tracks: list[dict]) -> pd.DataFrame:
    """Convierte la lista de tracks en un DataFrame con metadata básica."""
    rows = []
    seen_ids = set()

    for track in tracks:
        track_id = track.get("id")
        if not track_id or track_id in seen_ids:
            continue
        seen_ids.add(track_id)

        rows.append({
            "id": track_id,
            "name": track["name"],
            "artist": track["artists"][0]["name"] if track["artists"] else "Desconocido",
            "album": track["album"]["name"] if track.get("album") else "",
        })

    return pd.DataFrame(rows)


def main():
    print("🎵 Conectando con Spotify...")
    sp = get_spotify_client()

    print("📥 Extrayendo tus Top Tracks...")
    top_tracks = get_top_tracks(sp)

    print("📥 Extrayendo canciones de tus playlists guardadas...")
    playlist_tracks = get_saved_playlist_tracks(sp)

    all_tracks = top_tracks + playlist_tracks
    df = build_dataframe(all_tracks)
    print(f"✅ {len(df)} canciones únicas encontradas.")

    os.makedirs("data", exist_ok=True)
    output_path = "data/my_tracks.csv"
    df.to_csv(output_path, index=False)
    print(f"💾 Metadata guardada en {output_path} ({len(df)} canciones).")
    print(df.head())
    print("\n👉 Siguiente paso: correr src/data/match_with_reference_dataset.py")
    print("   para asignarle audio features a estas canciones.")


if __name__ == "__main__":
    main()