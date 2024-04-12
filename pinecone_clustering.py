import os
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
import pandas as pd
from sklearn.cluster import KMeans
import numpy as np

# Load pre-trained Sentence Transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Initialize Pinecone client
pc = Pinecone(api_key="eeb95b42-5012-4301-854b-7d75ae9fd293")  # Replace with your API key

# Define the index parameters
index_name = "clustering"  # Replace with your index name

# Check if the index exists
if index_name not in pc.list_indexes().names():
    print(f"Index '{index_name}' does not exist.")
    exit()

# Get the Pinecone index
index = pc.Index(index_name)

# Load the dataset (assuming you have a DataFrame df with 'id' and 'description' columns)
csv_file_path = 'test_bez_upravy.csv'
df = pd.read_csv(csv_file_path, delimiter=';')

# Encode descriptions into vectors
descriptions = df['description'].tolist()
description_vectors = model.encode(descriptions)

# Retrieve vectors from the index
all_vectors = []

batch_size = 1000
start = 0

while start < len(description_vectors):
    response = index.query(vector=description_vectors[start:start+batch_size].tolist(), top_k=50)
    vectors = response.data  # Change this line
        # Check if vectors is not None
    if vectors is not None:
        all_vectors.extend(vectors)
    
    start += batch_size

# Perform clustering using KMeans
n_clusters = 5  # Specify the number of clusters
kmeans = KMeans(n_clusters=n_clusters)
cluster_labels = kmeans.fit_predict(np.array(all_vectors).reshape(-1, 1))

# Add the cluster labels to the DataFrame
df['cluster_label'] = cluster_labels

# Output the cluster labels
print("Cluster labels:")
print(df['cluster_label'].value_counts())

# Perform similarity search within each cluster
for cluster_label in range(n_clusters):
    cluster_data = df[df['cluster_label'] == cluster_label]
    cluster_embeddings = [all_vectors[idx] for idx in range(len(all_vectors)) if cluster_labels[idx] == cluster_label]
    query_embedding = np.mean(cluster_embeddings, axis=0)
    
    # Print cluster information
    print(f"\nSearch results for cluster {cluster_label}:")
    print(f"Number of documents in cluster: {len(cluster_data)}")
    
    # Perform similarity search within the cluster
    for emb in cluster_embeddings:
        similarity = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
        print(f"Similarity: {similarity}\n")
