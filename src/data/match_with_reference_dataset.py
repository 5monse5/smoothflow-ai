"""
FASE 1B: Cruce con Dataset de Referencia - SmoothFlow AI
----------------------------------------------------------
Como Spotify ya no entrega audio features por API para apps nuevas,
usamos un dataset público (data/reference_tracks.csv, ~32.800 canciones)
que ya los tiene calculados, y le asignamos esos valores a TUS canciones
haciendo match por nombre de canción + artista.

Usa MATCHING DIFUSO (fuzzy matching) en vez de coincidencia exacta, porque
el dataset de referencia tiene muchas variantes de título ("Beat It - Single
Version", "Bad - 2012 Remaster", etc.) que un match exacto no encuentra.

También DEDUPLICA tus canciones: si tenés dos versiones del mismo tema
("Beat It" y "Beat It - Live"), se queda con una sola para que la playlist
final no repita la misma canción.

Input:  data/my_tracks.csv         (de connect_spotify.py)
        data/reference_tracks.csv  (dataset de referencia)
Output: data/my_tracks_with_features.csv

Cómo correrlo:
    python src/data/match_with_reference_dataset.py
"""

import re
import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process

FEATURE_COLUMNS = ["danceability", "energy", "valence", "acousticness", "tempo", "loudness", "speechiness"]

# Umbral mínimo de similitud (0-100) para aceptar un match difuso.
# Más alto = más estricto (menos falsos positivos, pero también menos matches).
ARTIST_THRESHOLD = 85
TRACK_THRESHOLD = 80


def normalize(text: str) -> str:
    """Normaliza texto: minúsculas, sin acentos raros ni símbolos, sin sufijos
    de versión comunes ("- Remastered", "(feat. X)", "- Single Version", etc.)
    """
    text = str(text).lower().strip()
    text = re.sub(r"\(.*?\)|\[.*?\]", "", text)  # saca "(feat. X)", "(Remix)"
    text = re.sub(r"-\s*((\d{4}\s*)?remaster(ed)?(\s*\d{4})?|"
                   r"single version|radio edit|live|mono|stereo|version|edit).*$", "", text)
    text = re.sub(r"[^a-z0-9 ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def deduplicate_my_tracks(my_tracks: pd.DataFrame) -> pd.DataFrame:
    """Si tenés dos versiones del mismo tema (ej: 'Beat It' y 'Beat It - Live'),
    nos quedamos con una sola para que la playlist final no repita canciones.
    """
    my_tracks = my_tracks.copy()
    my_tracks["_dedup_key"] = (
        my_tracks["artist"].apply(normalize) + "||" + my_tracks["name"].apply(normalize)
    )
    before = len(my_tracks)
    my_tracks = my_tracks.drop_duplicates(subset="_dedup_key", keep="first")
    removed = before - len(my_tracks)
    if removed > 0:
        print(f"🔁 {removed} versiones duplicadas del mismo tema removidas "
              f"(ej: 'Beat It' vs 'Beat It - Live').")
    return my_tracks.drop(columns="_dedup_key")


def match_tracks(my_tracks: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    reference = reference.dropna(subset=["track_name", "track_artist"]).copy()
    reference["norm_artist"] = reference["track_artist"].apply(normalize)
    reference["norm_track"] = reference["track_name"].apply(normalize)

    # Agrupamos las canciones de referencia por artista normalizado, para no
    # tener que comparar cada canción tuya contra las 32.800 de una (lento).
    unique_artists = reference["norm_artist"].unique().tolist()

    empty_features = pd.Series({col: np.nan for col in FEATURE_COLUMNS})
    results = []

    for _, row in my_tracks.iterrows():
        query_artist = normalize(row["artist"])
        query_track = normalize(row["name"])

        # 1. Encontrar el artista más parecido en el dataset de referencia.
        artist_match = process.extractOne(
            query_artist, unique_artists, scorer=fuzz.ratio, score_cutoff=ARTIST_THRESHOLD
        )
        if artist_match is None:
            results.append(empty_features)
            continue

        matched_artist = artist_match[0]
        candidates = reference[reference["norm_artist"] == matched_artist]

        # 2. Dentro de las canciones de ese artista, encontrar el título más parecido.
        track_match = process.extractOne(
            query_track, candidates["norm_track"].tolist(), scorer=fuzz.ratio,
            score_cutoff=TRACK_THRESHOLD,
        )
        if track_match is None:
            results.append(empty_features)
            continue

        matched_row = candidates[candidates["norm_track"] == track_match[0]].iloc[0]
        results.append(matched_row[FEATURE_COLUMNS])

    features_df = pd.DataFrame(results, index=my_tracks.index)
    return pd.concat([my_tracks, features_df], axis=1)


def main():
    print("📂 Cargando tus canciones y el dataset de referencia...")
    my_tracks = pd.read_csv("data/my_tracks.csv")
    reference = pd.read_csv("data/reference_tracks.csv")

    my_tracks = deduplicate_my_tracks(my_tracks)

    print("🔎 Cruzando por artista + canción (matching difuso, puede tardar un minuto)...")
    merged = match_tracks(my_tracks, reference)

    matched = merged.dropna(subset=FEATURE_COLUMNS)
    unmatched = merged[merged[FEATURE_COLUMNS].isna().any(axis=1)]

    print(f"✅ {len(matched)}/{len(merged)} canciones encontraron match "
          f"({len(matched) / len(merged) * 100:.1f}%).")
    if len(unmatched) > 0:
        print(f"⚠️  {len(unmatched)} canciones sin match (quedan afuera del análisis).")
        print("   Ejemplos:", unmatched["name"].head(5).tolist())

    output_path = "data/my_tracks_with_features.csv"
    matched.to_csv(output_path, index=False)
    print(f"💾 Dataset final guardado en {output_path}")
    print(matched[["name", "artist"] + FEATURE_COLUMNS].head())


if __name__ == "__main__":
    main()