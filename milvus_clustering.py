import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from pymilvus import connections, Collection, utility
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

# Load environment variables
load_dotenv()
token = os.getenv('MILVUS_TOKEN')
endpoint = os.getenv('MILVUS_ENDPOINT')

# Connect to Milvus
connections.connect(uri=endpoint, token=token)

# Define the collection name
collection_name = "semanticsearch"

# Check if the collection exists
if not utility.has_collection(collection_name):
    print(f"Collection {collection_name} does not exist.")
    connections.disconnect()
    raise SystemExit

# Get the collection
collection = Collection(name=collection_name)

# Assuming vectors are stored with integer IDs from 1 to 7600
ids_list = [i for i in range(1, 7601)]

# Function to fetch vectors in batches
def fetch_vectors_in_batches(ids, batch_size=100):
    vectors = []
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i + batch_size]
        expr = f"id in {batch_ids}"
        # Note: Make sure the field name 'vector' is correct
        result = collection.query(expr, output_fields=["vector"])
        vectors.extend([item["vector"] for item in result])
    return np.array(vectors)

# Fetch vectors in batches; adjust batch size as needed
vectors = fetch_vectors_in_batches(ids_list, batch_size=100)

# Clustering with KMeans
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(vectors)
cluster_labels = kmeans.labels_

# Dimensionality Reduction for visualization
pca = PCA(n_components=2)
vectors_reduced = pca.fit_transform(vectors)

# Plotting the clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(vectors_reduced[:, 0], vectors_reduced[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5)
plt.colorbar(scatter)
plt.title('Cluster Visualization of Milvus Vectors')
plt.xlabel('First Principal Component - Direction of Most Variation')
plt.ylabel('Second Principal Component - Direction of Second Most Variation')
plt.savefig('milvus_cluster_visualization.png')
plt.show()

# Closest vector to each cluster's centroid
distances = cdist(vectors, kmeans.cluster_centers_, 'euclidean')
closest_indices = np.argmin(distances, axis=0)
for i, idx in enumerate(closest_indices):
    vector_id = ids_list[idx]
    print(f"Cluster {i+1}, Vector ID closest to centroid: {vector_id}")

# Disconnect from Milvus
connections.disconnect()
