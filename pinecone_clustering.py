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

# Initialize Pinecone
pc = Pinecone(api_key=api_key)

# Define the index parameters
index_name = "clusteringenriched"  # Using the index where polarity and subjectivity are stored

# Retrieve the index
index = pc.Index(index_name)

# Get list of ids, here assuming you know the IDs range or have them listed
ids_list = [str(i) for i in range(1, 7601)]

# Fetch vectors and metadata in batches
def fetch_data_in_batches(ids, batch_size=100):
    """ Fetch vectors and metadata from Pinecone in batches. """
    full_data = []
    for i in range(0, len(ids), batch_size):
        batch_ids = ids[i:i + batch_size]
        response = index.fetch(batch_ids)
        for item in response['vectors'].values():
            full_data.append({
                'vector': item['values'],
                'polarity': item['metadata']['polarity'],
                'subjectivity': item['metadata']['subjectivity']
            })
    return pd.DataFrame(full_data)

# Fetch data
data = fetch_data_in_batches(ids_list, batch_size=100)

# Prepare features by combining vectors with polarity and subjectivity
vectors = np.array(data['vector'].tolist())
polarity = np.expand_dims(data['polarity'], axis=1)
subjectivity = np.expand_dims(data['subjectivity'], axis=1)

# Combine vectors with polarity and subjectivity
features = np.hstack((vectors, polarity, subjectivity))

# Clustering with only vectors
n_clusters = 5
kmeans_vectors = KMeans(n_clusters=n_clusters, random_state=0)
kmeans_vectors.fit(vectors)

# Clustering with combined features
kmeans_combined = KMeans(n_clusters=n_clusters, random_state=0)
kmeans_combined.fit(features)

# PCA and plotting for vector-only clustering
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# PCA for dimensionality reduction
pca = PCA(n_components=2)
vectors_reduced = pca.fit_transform(vectors)
combined_reduced = pca.transform(features)  # Ensure PCA is fitted only once and used for both

# Setting up the figure and axes
plt.figure(figsize=(20, 8))  # Wider figure to accommodate detailed labels

# Plot for clustering based only on vectors
plt.subplot(1, 2, 1)
scatter1 = plt.scatter(vectors_reduced[:, 0], vectors_reduced[:, 1], c=kmeans_vectors.labels_, cmap='viridis', alpha=0.5)
plt.colorbar(scatter1, label='Cluster Labels')
plt.title('Clusters Formed Using Only Textual Content')
plt.xlabel('Component 1: Capturing Main Variance in Text')
plt.ylabel('Component 2: Capturing Secondary Variance in Text')

# Plot for clustering based on vectors, polarity, and subjectivity
plt.subplot(1, 2, 2)
scatter2 = plt.scatter(combined_reduced[:, 0], combined_reduced[:, 1], c=kmeans_combined.labels_, cmap='viridis', alpha=0.5)
plt.colorbar(scatter2, label='Cluster Labels')
plt.title('Clusters Formed Using Textual Content with Sentiment Analysis')
plt.xlabel('Component 1: Integrating Text and Sentiment Variance')
plt.ylabel('Component 2: Secondary Variance from Combined Features')

plt.suptitle('Comparative Visualization of Document Clustering With and Without Sentiment Features', fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust layout to leave space for the suptitle
plt.show()

