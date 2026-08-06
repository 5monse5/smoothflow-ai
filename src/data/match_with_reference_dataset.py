"""
FASE 1B: Cruce con Dataset de Referencia - SmoothFlow AI
----------------------------------------------------------
Como Spotify ya no entrega audio features por API para apps nuevas,
usamos un dataset público (data/reference_tracks.csv, ~32.800 canciones)
que ya los tiene calculados, y le asignamos esos valores a TUS canciones
haciendo match por nombre de canción + artista.

Input:  data/my_tracks.csv         (de connect_spotify.py)
        data/reference_tracks.csv  (dataset de referencia)
Output: data/my_tracks_with_features.csv

Cómo correrlo:
    python src/data/match_with_reference_dataset.py
"""

import re
import pandas as pd

FEATURE_COLUMNS = ["danceability", "energy", "valence", "acousticness", "tempo"]


def normalize(text: str) -> str:
    """Normaliza texto para poder comparar: minúsculas, sin acentos ni símbolos."""
    text = text.lower().strip()
    text = re.sub(r"\(.*?\)|\[.*?\]", "", text)  # saca "(feat. X)", "(Remix)", etc.
    text = re.sub(r"[^a-z0-9 ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def match_tracks(my_tracks: pd.DataFrame, reference: pd.DataFrame) -> pd.DataFrame:
    reference = reference.dropna(subset=["track_name", "track_artist"])
    reference = reference.drop_duplicates(subset=["track_name", "track_artist"]).copy()
    reference["match_key"] = (
        reference["track_name"].apply(normalize) + "||" + reference["track_artist"].apply(normalize)
    )
    reference_lookup = reference.set_index("match_key")[FEATURE_COLUMNS]

    my_tracks = my_tracks.copy()
    my_tracks["match_key"] = (
        my_tracks["name"].apply(normalize) + "||" + my_tracks["artist"].apply(normalize)
    )

    merged = my_tracks.join(reference_lookup, on="match_key")
    return merged.drop(columns=["match_key"])


def main():
    print("📂 Cargando tus canciones y el dataset de referencia...")
    my_tracks = pd.read_csv("data/my_tracks.csv")
    reference = pd.read_csv("data/reference_tracks.csv")

    print("🔎 Cruzando por nombre + artista...")
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
