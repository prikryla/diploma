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

# Clustering
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(features)

# Cluster labels
cluster_labels = kmeans.labels_

# Reduce dimensions to 2D for visualization (using only vectors here for PCA)
pca = PCA(n_components=2)
vectors_reduced = pca.fit_transform(vectors)

# Plotting
plt.figure(figsize=(10, 8))
scatter = plt.scatter(vectors_reduced[:, 0], vectors_reduced[:, 1], c=data['polarity'], cmap='coolwarm', alpha=0.5)
plt.colorbar(scatter, label='Polarity')
plt.title('Visualization by Polarity')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()

