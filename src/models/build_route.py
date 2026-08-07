"""
FASE 3: Lógica de la Rampa de Transición ("El DJ") - SmoothFlow AI
---------------------------------------------------------------------
Arma la playlist final combinando:

1. BLOQUES DE PERMANENCIA: canciones del mood/género elegido para cada
   "estación" de tu ruta. Priorizan TUS canciones conocidas, pero no de
   forma absoluta - si una canción del catálogo general encaja MEJOR
   (está matemáticamente más cerca del centro del cluster), esa gana.
   Esto se logra con un "descuento de familiaridad": tus canciones
   conocidas compiten con una distancia artificialmente reducida, no
   con una ventaja infinita.

2. CANCIONES PUENTE: para pasar de una estación a la siguiente, se
   calculan M puntos intermedios (interpolación lineal en el espacio
   de audio features) entre el final de un bloque y el principio del
   siguiente, y se busca con KNN la canción real más cercana a cada
   punto - buscando en TODO el catálogo, para favorecer el descubrimiento
   de música nueva justo en las transiciones.

Input:  data/reference_tracks_clustered.csv (de train_clustering.py)
        data/models/scaler.pkl, kmeans.pkl, knn.pkl
        data/my_tracks_with_features.csv (de match_with_reference_dataset.py,
        opcional - si no existe, arma la ruta solo con el catálogo general)

Cómo correrlo (modo demo, con una ruta de ejemplo):
    python src/models/build_route.py
"""

import pickle
import re

import numpy as np
import pandas as pd

FEATURE_COLUMNS = ["danceability", "energy", "valence", "acousticness", "tempo", "loudness", "speechiness"]

# Cuánto "descuento" en distancia le damos a tus canciones conocidas
# frente a las del catálogo general. 0.0 = sin ventaja (compiten igual).
# 1.0 = ventaja total (siempre gana lo conocido, si hay opciones).
# 0.35 es un punto medio: ayuda a que aparezcan más tus canciones,
# pero una canción del catálogo mucho más cercana igual puede ganar.
FAMILIARITY_DISCOUNT = 0.35


def normalize_title(text: str) -> str:
    """Reduce distintas versiones del mismo tema (remaster, live, remix, etc.)
    a una misma clave, para no elegir dos veces 'la misma' canción.
    """
    text = str(text).lower().strip()
    text = re.sub(r"\(.*?\)|\[.*?\]", "", text)
    text = re.sub(r"-\s*((\d{4}\s*)?remaster(ed)?(\s*\d{4})?|"
                   r"single version|radio edit|album version|live|mono|stereo|"
                   r"version|edit|remix).*$", "", text)
    text = re.sub(r"[^a-z0-9 ]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_models():
    with open("data/models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("data/models/kmeans.pkl", "rb") as f:
        kmeans = pickle.load(f)
    with open("data/models/knn.pkl", "rb") as f:
        knn = pickle.load(f)
    return scaler, kmeans, knn


def load_data():
    catalog = pd.read_csv("data/reference_tracks_clustered.csv")
    try:
        my_tracks = pd.read_csv("data/my_tracks_with_features.csv")
        my_tracks = my_tracks.dropna(subset=FEATURE_COLUMNS).copy()
        my_tracks = my_tracks.rename(columns={"name": "track_name", "artist": "track_artist"})
    except FileNotFoundError:
        print("⚠️  No se encontró data/my_tracks_with_features.csv - "
              "la ruta se va a armar solo con el catálogo general.")
        my_tracks = pd.DataFrame(columns=["track_name", "track_artist"] + FEATURE_COLUMNS)
    return catalog, my_tracks


def select_block(
    cluster_id: int,
    n_songs: int,
    catalog: pd.DataFrame,
    my_tracks: pd.DataFrame,
    scaler,
    kmeans,
    exclude_ids: set,
    exclude_keys: set,
) -> pd.DataFrame:
    """Elige n_songs canciones para representar un cluster/mood, priorizando
    (pero sin garantizar) las canciones conocidas del usuario, según
    FAMILIARITY_DISCOUNT. Evita elegir dos versiones del mismo tema.
    """
    centroid = kmeans.cluster_centers_[cluster_id]

    catalog_candidates = catalog[catalog["cluster"] == cluster_id].copy()
    catalog_candidates = catalog_candidates[~catalog_candidates["track_id"].isin(exclude_ids)]
    catalog_candidates["is_known"] = False

    known_candidates = pd.DataFrame()
    if len(my_tracks) > 0:
        my_scaled = scaler.transform(my_tracks[FEATURE_COLUMNS])
        my_clusters = kmeans.predict(my_scaled)
        known_candidates = my_tracks[my_clusters == cluster_id].copy()
        known_candidates = known_candidates.rename(columns={"id": "track_id"})
        known_candidates = known_candidates[~known_candidates["track_id"].isin(exclude_ids)]
        known_candidates["is_known"] = True

    all_candidates = pd.concat([catalog_candidates, known_candidates], ignore_index=True)
    if len(all_candidates) == 0:
        return all_candidates

    all_candidates["_key"] = (
        all_candidates["track_artist"].apply(normalize_title) + "||"
        + all_candidates["track_name"].apply(normalize_title)
    )
    all_candidates = all_candidates[~all_candidates["_key"].isin(exclude_keys)]
    all_candidates = all_candidates.sort_values("is_known", ascending=False)
    all_candidates = all_candidates.drop_duplicates(subset="_key", keep="first")

    X = scaler.transform(all_candidates[FEATURE_COLUMNS])
    distances = np.linalg.norm(X - centroid, axis=1)

    discount = np.where(all_candidates["is_known"], FAMILIARITY_DISCOUNT, 0.0)
    adjusted_distances = distances * (1 - discount)

    all_candidates["_distance"] = adjusted_distances
    selected = all_candidates.sort_values("_distance").head(n_songs)
    return selected


def interpolate_points(start_point: np.ndarray, end_point: np.ndarray, n_points: int) -> np.ndarray:
    """Genera n_points puntos intermedios entre start_point y end_point
    (interpolación lineal), sin incluir los extremos.
    """
    steps = np.linspace(0, 1, n_points + 2)[1:-1]
    return np.array([start_point + step * (end_point - start_point) for step in steps])


def find_bridge_songs(
    points: np.ndarray, catalog: pd.DataFrame, knn, exclude_ids: set, exclude_keys: set
) -> pd.DataFrame:
    """Busca, para cada punto intermedio, la canción real más cercana en TODO
    el catálogo (para favorecer descubrir música nueva en las transiciones).
    Evita repetir versiones de temas ya usados en otra parte de la ruta.
    """
    bridge_rows = []
    used_ids = set(exclude_ids)
    used_keys = set(exclude_keys)

    for point in points:
        distances, indices = knn.kneighbors([point], n_neighbors=25)
        for idx in indices[0]:
            candidate = catalog.iloc[idx]
            key = normalize_title(candidate["track_artist"]) + "||" + normalize_title(candidate["track_name"])
            if candidate["track_id"] not in used_ids and key not in used_keys:
                bridge_rows.append(candidate)
                used_ids.add(candidate["track_id"])
                used_keys.add(key)
                break

    return pd.DataFrame(bridge_rows)


def build_route(
    stations: list[int],
    block_size: int,
    bridge_size: int,
    catalog: pd.DataFrame,
    my_tracks: pd.DataFrame,
    scaler,
    kmeans,
    knn,
) -> pd.DataFrame:
    """Arma la ruta completa: Bloque A -> Transición -> Bloque B -> ...

    stations: lista de IDs de cluster, en el orden en que el usuario los eligió.
    block_size: cuántas canciones por bloque de permanencia.
    bridge_size: cuántas canciones puente por transición.
    """
    route_parts = []
    used_ids = set()
    used_keys = set()

    def track_key(row):
        return normalize_title(row["track_artist"]) + "||" + normalize_title(row["track_name"])

    for i, cluster_id in enumerate(stations):
        block = select_block(cluster_id, block_size, catalog, my_tracks, scaler, kmeans, used_ids, used_keys)
        block["role"] = "bloque"
        block["station"] = i + 1
        if len(block) > 0:
            used_ids.update(block["track_id"].dropna().tolist())
            used_keys.update(block.apply(track_key, axis=1).tolist())
        route_parts.append(block)

        if i < len(stations) - 1:
            start_centroid = kmeans.cluster_centers_[cluster_id]
            end_centroid = kmeans.cluster_centers_[stations[i + 1]]
            points = interpolate_points(start_centroid, end_centroid, bridge_size)
            bridge = find_bridge_songs(points, catalog, knn, used_ids, used_keys)
            if len(bridge) > 0:
                bridge["role"] = "transición"
                bridge["station"] = f"{i + 1}→{i + 2}"
                used_ids.update(bridge["track_id"].tolist())
                used_keys.update(bridge.apply(track_key, axis=1).tolist())
                route_parts.append(bridge)

    return pd.concat(route_parts, ignore_index=True)


def main():
    print("📂 Cargando modelos entrenados y datasets...")
    scaler, kmeans, knn = load_models()
    catalog, my_tracks = load_data()

    n_clusters = kmeans.n_clusters
    print(f"   {n_clusters} moods disponibles (cluster 0 a {n_clusters - 1}).")

    demo_stations = [1, 0]
    demo_block_size = 3
    demo_bridge_size = 2

    print(f"\n🎧 Armando ruta de demo: estaciones {demo_stations}, "
          f"{demo_block_size} canciones por bloque, {demo_bridge_size} por transición...")

    route = build_route(
        demo_stations, demo_block_size, demo_bridge_size,
        catalog, my_tracks, scaler, kmeans, knn,
    )

    display_cols = ["role", "station", "track_name", "track_artist", "is_known"]
    display_cols = [c for c in display_cols if c in route.columns]
    print("\n🎵 Ruta generada:")
    print(route[display_cols].to_string(index=False))

    route.to_csv("data/demo_route.csv", index=False)
    print("\n💾 Ruta guardada en data/demo_route.csv")
    print("\n👉 Siguiente paso: Fase 4, la interfaz en Streamlit para elegir "
          "estaciones y tamaños de forma interactiva.")


if __name__ == "__main__":
    main()