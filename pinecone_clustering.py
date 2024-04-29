import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

load_dotenv()
api_key = os.getenv('API_KEY_PINECONE')

# Initialize Pinecone with your API key
pc = Pinecone(api_key=api_key)

# Load pre-trained Sentence Transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Define the index parameters
index_name = "semanticsearch"

# Retrieve the index
index = pc.Index(index_name)

ids_list = [str(i) for i in range(1, 7601)]

# Function to fetch vectors in batches
def fetch_vectors_in_batches(ids, batch_size=100):
    """ Fetch vectors from Pinecone in batches. """
    vectors = []
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i + batch_size]
        response = index.fetch(batch_ids)
        vectors.extend([item['values'] for item in response['vectors'].values()])
    return np.array(vectors)

# Fetch vectors in batches; adjust batch size as needed
vectors = fetch_vectors_in_batches(ids_list, batch_size=100)

# Set the number of clusters
n_clusters = 5

# Initialize KMeans
kmeans = KMeans(n_clusters=n_clusters, random_state=0)

# Fit KMeans with the fetched vectors
kmeans.fit(vectors)

# Get cluster labels for each vector
cluster_labels = kmeans.labels_

# Reduce dimensions to 2D for visualization
pca = PCA(n_components=2)
vectors_reduced = pca.fit_transform(vectors)

# Plotting the clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(vectors_reduced[:, 0], vectors_reduced[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5)
plt.colorbar(scatter)
plt.title('Cluster Visualization of Vector Embeddings')
plt.xlabel('First Principal Component - Variation Direction')
plt.ylabel('Second Principal Component - Variation Direction')
plt.savefig('pinecone_cluster_visualization.png')
plt.show()

# Calculate distances from each vector to each centroid
distances = cdist(vectors, kmeans.cluster_centers_, 'euclidean')

# Find the index of the closest vectors
closest_indices = np.argmin(distances, axis=0)

# Print closest vector details
for i, idx in enumerate(closest_indices):
    vector_id = ids_list[idx]
    print(f"Cluster {i+1}, Vector ID closest to centroid: {vector_id}")
