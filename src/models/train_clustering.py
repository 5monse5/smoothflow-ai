"""
FASE 2: Clustering e Inteligencia Artificial - SmoothFlow AI
----------------------------------------------------------------
Entrena dos modelos sobre las 32.800 canciones del dataset de referencia
(no solo las tuyas, para tener más variedad y poder recomendar música
que quizás no conocías):

1. K-MEANS: agrupa las canciones en "micro-moods" según sonido similar.
   K=10 está fijado a mano (no se busca automático). Se probó de 4 a 12
   con silhouette score y K=4 daba el mejor número (0.181), pero con muy
   pocas opciones de mood para la interfaz. K=10 da mucha mejor variedad
   (10 moods distintos para elegir en Streamlit) sin perder demasiada
   calidad de cluster (silhouette 0.162 vs 0.181 de K=4). Decisión tomada
   y validada, no es un valor provisorio.

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

# K fijado a mano en 10 (más variedad de moods para la interfaz).
# Ya no se prueba un rango ni se elige automático por silhouette score.
N_CLUSTERS = 10

MODELS_DIR = "data/models"


def load_data() -> pd.DataFrame:
    df = pd.read_csv("data/reference_tracks.csv")
    df = df.dropna(subset=FEATURE_COLUMNS)
    df = df.drop_duplicates(subset=["track_name", "track_artist"])
    return df.reset_index(drop=True)


def report_silhouette(X_scaled: np.ndarray, k: int) -> None:
    """Informa el silhouette score de K en una porción separada de los datos
    (no usada para entrenar), solo a modo de referencia/registro - ya NO se
    usa para decidir K, que está fijado a mano en N_CLUSTERS.
    """
    X_train, X_valid = train_test_split(X_scaled, test_size=0.2, random_state=42)
    kmeans_check = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_check.fit(X_train)
    valid_labels = kmeans_check.predict(X_valid)
    score = silhouette_score(X_valid, valid_labels)
    print(f"ℹ️  K={k} fijado a mano (silhouette score de referencia = {score:.3f})")


def main():
    print("📂 Cargando dataset de referencia completo...")
    df = load_data()
    print(f"   {len(df)} canciones disponibles para entrenar.")

    print("⚖️  Normalizando features con StandardScaler...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[FEATURE_COLUMNS])

    report_silhouette(X_scaled, N_CLUSTERS)

    print(f"🎯 Entrenando K-Means final con K={N_CLUSTERS} sobre todo el dataset...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=10)
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
    print("\n👉 Siguiente paso: correr streamlit run src/app/app.py y confirmar")
    print("   que aparecen 10 moods en el multiselect.")


if __name__ == "__main__":
    main()