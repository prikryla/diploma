import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from pymilvus import connections, Collection, utility
from matplotlib.colors import ListedColormap, BoundaryNorm


def establish_milvus_connection():
    """
    Establishes connection to the Milvus server using environment variables.

    Returns:
        None
    """
    load_dotenv()
    token = os.getenv('MILVUS_TOKEN')
    endpoint = os.getenv('MILVUS_ENDPOINT')
    connections.connect(uri=endpoint, token=token)

def fetch_data_in_batches(ids, batch_size=100):
    """
    Fetches vectors and additional data from Milvus in batches.

    Args:
        ids (list): List of document IDs.
        batch_size (int): Size of each batch for fetching data.

    Returns:
        pandas.DataFrame: Dataframe containing fetched data.
    """
    collection_name = "clusteringenriched"
    collection = Collection(name=collection_name)
    full_data = []
    start_time = time.time()
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i + batch_size]
        expr = f"id in {batch_ids}"
        result = collection.query(expr, output_fields=["vector", "polarity", "subjectivity", "id"])
        full_data.extend([{
            'vector': item["vector"],
            'polarity': item["polarity"],
            'subjectivity': item["subjectivity"],
            'id': item["id"]
        } for item in result])
    end_time = time.time()
    print(f"Overall query execution time: {end_time - start_time} seconds")
    return pd.DataFrame(full_data)

def main():
    """
    Main function to perform clustering analysis on data fetched from Milvus.
    """
    # Connect to Milvus
    establish_milvus_connection()

    # Define the collection name
    collection_name = "clusteringenriched"

    # Check if the collection exists
    if not utility.has_collection(collection_name):
        print(f"Collection {collection_name} does not exist.")
        connections.disconnect()
        raise SystemExit

    # Get the collection
    collection = Collection(name=collection_name)

    # Assuming vectors are stored with integer IDs from 1 to 7600
    ids_list = [i for i in range(1, 7601)]

    # Fetch data
    data = fetch_data_in_batches(ids_list)

    # Normalize features
    scaler = StandardScaler()
    features = np.hstack((np.array(data['vector'].tolist()), 
                          np.expand_dims(data['polarity'], axis=1), 
                          np.expand_dims(data['subjectivity'], axis=1)))
    normalized_features = scaler.fit_transform(features)

    # Clustering with KMeans
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(normalized_features)

    # PCA for dimensionality reduction
    pca = PCA(n_components=2)
    features_reduced = pca.fit_transform(normalized_features)

    # Plotting the clusters
    plt.figure(figsize=(10, 8))
    cmap = ListedColormap(['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00'])  # Red, Blue, Green, Orange
    norm = BoundaryNorm(np.arange(0, 4+1), cmap.N)

    scatter = plt.scatter(features_reduced[:, 0], features_reduced[:, 1], c=kmeans.labels_, cmap=cmap, norm=norm, alpha=0.5)
    cbar = plt.colorbar(scatter, ticks=np.linspace(0.5, n_clusters-0.5, n_clusters), label='Cluster Labels')
    cbar.ax.set_yticklabels(np.arange(1, n_clusters+1))  # Update tick labels for clarity
    plt.title('Clusters created based on clustering with additional Milvus database information')
    plt.xlabel('X-Axis - First Component')
    plt.ylabel('Y-Axis - Second Component')
    plt.savefig('renders/milvus_cluster_visualization.png')
    plt.show()

    # Disconnect from Milvus
    connections.disconnect()

if __name__ == "__main__":
    main()