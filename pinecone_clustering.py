import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from pinecone import Pinecone
from dotenv import load_dotenv
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import time
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()
api_key = os.getenv('API_KEY_PINECONE')

# Initialize Pinecone
pc = Pinecone(api_key=api_key)

# Define the index parameters
index_name = "clusteringenriched"  # Using the index where polarity and subjectivity are stored

# Retrieve the index
index = pc.Index(index_name)

# Get list of ids, assuming you know the IDs range
ids_list = [str(i) for i in range(1, 7601)]

# Function to fetch vectors and metadata in batches
def fetch_data_in_batches(ids, batch_size=500):  # Adjusted batch size for optimal performance
    full_data = []
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(fetch_batch, ids[i:i + batch_size]) for i in range(0, len(ids), batch_size)]
        for future in futures:
            full_data.extend(future.result())
    return pd.DataFrame(full_data)

def fetch_batch(batch_ids):
    response = index.fetch(batch_ids)
    return [{
        'vector': item['values'],
        'polarity': item['metadata']['polarity'],
        'subjectivity': item['metadata']['subjectivity']
    } for item in response['vectors'].values()]

# Start measuring time for querying data
start_time_query = time.time()

# Fetch data
data = fetch_data_in_batches(ids_list)

# End measuring time for querying data
end_time_query = time.time()
print(f"Overall query execution time: {end_time_query - start_time_query} seconds")

# Function to perform clustering
def perform_clustering(data):
    vectors = np.array(data['vector'].tolist())
    polarity = np.expand_dims(data['polarity'], axis=1)
    subjectivity = np.expand_dims(data['subjectivity'], axis=1)
    features = np.hstack((vectors, polarity, subjectivity))  # Combine vectors with polarity and subjectivity

    n_clusters = 3  # Specify the number of clusters
    kmeans_combined = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans_combined.fit(features)  # Clustering with combined features

    # PCA for dimensionality reduction
    pca = PCA(n_components=2)
    combined_reduced = pca.fit_transform(features)  # Transform the feature set for visualization

    return combined_reduced, kmeans_combined.labels_

# Perform clustering
combined_reduced, kmeans_combined_labels = perform_clustering(data)

# Plotting the clusters
# Vykreslení clusterů
plt.figure(figsize=(10, 8))
scatter2 = plt.scatter(combined_reduced[:, 0], combined_reduced[:, 1], c=kmeans_combined_labels, cmap='viridis', alpha=0.5)
plt.colorbar(scatter2, label='Štítky clusterů (každá barva představuje jiný cluster)')
plt.title('Clustery vytvořené na základě textového obsahu s analýzou sentimentu')
plt.xlabel('Integrace rozptylu textu a sentimentu')
plt.ylabel('Sekundární rozptyl z kombinovaných prvků')
plt.savefig("Pinecone clustering")
plt.show()
