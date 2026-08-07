"""
FASE 2: Clustering e Inteligencia Artificial - SmoothFlow AI
----------------------------------------------------------------
Entrena dos modelos sobre las 32.800 canciones del dataset de referencia
(no solo las tuyas, para tener más variedad y poder recomendar música
que quizás no conocías):

1. K-MEANS: agrupa las canciones en "micro-moods" según sonido similar.
   Se valida con silhouette score en una porción separada de los datos,
   para confirmar que los grupos son estables y no un artefacto del azar.

2. KNN (K-Nearest Neighbors): motor de búsqueda que, dado un punto
   cualquiera en el espacio de audio features, encuentra las canciones
   reales más cercanas a ese punto. Esto es lo que va a buscar las
   "canciones puente" en la Fase 3.

Input:  data/reference_tracks.csv
Output: data/models/scaler.pkl, data/models/kmeans.pkl, data/models/knn.pkl
        data/reference_tracks_clustered.csv (con la columna 'cluster' agregada)

Cómo correrlo:
    python src/models/train_clustering.py
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = ["danceability", "energy", "valence", "acousticness", "tempo", "loudness", "speechiness"]

# Rango de K (cantidad de clusters) a probar para elegir el mejor con el método del codo.
K_RANGE = range(4, 13)

MODELS_DIR = "data/models"


def load_data() -> pd.DataFrame:
    df = pd.read_csv("data/reference_tracks.csv")
    df = df.dropna(subset=FEATURE_COLUMNS)
    df = df.drop_duplicates(subset=["track_name", "track_artist"])
    return df.reset_index(drop=True)


def find_best_k(X_scaled: np.ndarray) -> int:
    """Prueba distintos valores de K y elige el que da mejor silhouette score
    en una muestra separada (no usada para entrenar), para evitar elegir un K
    que solo se ve bien sobre los mismos datos con los que se entrenó.
    """
    X_train, X_valid = train_test_split(X_scaled, test_size=0.2, random_state=42)

    print("🔬 Buscando el mejor número de clusters (K)...")
    scores = {}
    for k in K_RANGE:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(X_train)
        valid_labels = kmeans.predict(X_valid)
        score = silhouette_score(X_valid, valid_labels)
        scores[k] = score
        print(f"   K={k}: silhouette score = {score:.3f}")

    best_k = max(scores, key=scores.get)
    print(f"✅ Mejor K encontrado: {best_k} (silhouette score = {scores[best_k]:.3f})")
    return best_k


def main():
    print("📂 Cargando dataset de referencia completo...")
    df = load_data()
    print(f"   {len(df)} canciones disponibles para entrenar.")

    print("⚖️  Normalizando features con StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURE_COLUMNS])

    best_k = find_best_k(X_scaled)

    print(f"🎯 Entrenando K-Means final con K={best_k} sobre todo el dataset...")
    kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_scaled)

    print("\n📊 Tamaño de cada cluster:")
    print(df["cluster"].value_counts().sort_index())

    print("\n🔎 Entrenando motor de búsqueda KNN sobre todo el catálogo...")
    knn = NearestNeighbors(n_neighbors=10, metric="euclidean")
    knn.fit(X_scaled)

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(f"{MODELS_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    with open(f"{MODELS_DIR}/kmeans.pkl", "wb") as f:
        pickle.dump(kmeans, f)
    with open(f"{MODELS_DIR}/knn.pkl", "wb") as f:
        pickle.dump(knn, f)

    df.to_csv("data/reference_tracks_clustered.csv", index=False)

    print(f"\n💾 Modelos guardados en {MODELS_DIR}/")
    print("💾 Dataset con clusters guardado en data/reference_tracks_clustered.csv")
    print("\n👉 Siguiente paso: Fase 3, la lógica de la rampa de transición.")


if __name__ == "__main__":
    main()