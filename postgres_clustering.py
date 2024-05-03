import psycopg2
import psycopg2.extras
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Connection parameters using environment variables
conn_params = {
    'dbname': os.getenv('PG_DBNAME'),
    'user': os.getenv('PG_USER'),
    'password': os.getenv('PG_PASSWORD'),
    'host': os.getenv('PG_HOST')
}

# Connect to the PostgreSQL database
conn = psycopg2.connect(**conn_params)
cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

# SQL to fetch vectors along with polarity and subjectivity; adjust based on actual table and column names
sql = """
SELECT a.id, a.embedding, s.polarity, s.subjectivity
FROM ag_dataset a
JOIN ag_dataset_sentiment s ON a.id = s.data_id
ORDER BY a.id LIMIT 7600;
"""

# Execute the query
cursor.execute(sql)
# Fetch all the results
results = cursor.fetchall()

# Close the database connection
cursor.close()
conn.close()

# Extract ids, vectors, polarity, and subjectivity, converting binary data to numpy arrays
ids_list = [result['id'] for result in results]
vectors = np.array([np.frombuffer(result['embedding'], dtype=np.float32) for result in results])
polarity = np.array([result['polarity'] for result in results]).reshape(-1, 1)
subjectivity = np.array([result['subjectivity'] for result in results]).reshape(-1, 1)

# Combine vectors with polarity and subjectivity for clustering
features = np.hstack((vectors, polarity, subjectivity))

# Clustering with KMeans
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(features)
cluster_labels = kmeans.labels_

# Dimensionality Reduction for visualization with PCA on the combined features
pca = PCA(n_components=2)
features_reduced = pca.fit_transform(features)

# Plotting the clusters with combined features
plt.figure(figsize=(10, 8))
scatter = plt.scatter(features_reduced[:, 0], features_reduced[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5)
plt.colorbar(scatter)
plt.title('Cluster Visualization with Polarity and Subjectivity')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# Calculate distances from each vector to each centroid
distances = cdist(features, kmeans.cluster_centers_, 'euclidean')
closest_indices = np.argmin(distances, axis=0)

# Print closest vector details
for i, idx in enumerate(closest_indices):
    vector_id = ids_list[idx]
    print(f"Cluster {i+1}, Vector ID closest to centroid: {vector_id}")
