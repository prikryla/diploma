import pandas as pd
import numpy as np
import os

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from sklearn.cluster import KMeans

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

# Example: Print cluster assignments
for label, vector_id in zip(cluster_labels, ids_list):
    print(f"Vector ID: {vector_id} is in Cluster: {label}")
