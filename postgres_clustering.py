import psycopg2
import psycopg2.extras
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer

# Connection parameters
conn_params = {
    'dbname': 'diploma_2',
    'user': 'postgres',
    'password': 'postgres',
    'host': '127.0.0.1'
}

# Connect to the PostgreSQL database
conn = psycopg2.connect(**conn_params)
cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

# SQL to fetch vectors; adjust the SQL based on your actual table and column names
sql = "SELECT id, embedding FROM diploma_semantic_search ORDER BY id LIMIT 7600;"

# Execute the query
cursor.execute(sql)
# Fetch all the results
results = cursor.fetchall()

# Extract ids and vectors
ids_list = [result['id'] for result in results]
vectors = np.array([result['embedding'] for result in results])

# Close the database connection
cursor.close()
conn.close()

# Clustering with KMeans
n_clusters = 5
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(vectors)
cluster_labels = kmeans.labels_

# Dimensionality Reduction for visualization with PCA
pca = PCA(n_components=2)
vectors_reduced = pca.fit_transform(vectors)

# Plotting the clusters
plt.figure(figsize=(10, 8))
scatter = plt.scatter(vectors_reduced[:, 0], vectors_reduced[:, 1], c=cluster_labels, cmap='viridis', alpha=0.5)
plt.colorbar(scatter)
plt.title('Cluster Visualization')
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.show()

# Calculate distances from each vector to each centroid
distances = cdist(vectors, kmeans.cluster_centers_, 'euclidean')
closest_indices = np.argmin(distances, axis=0)

# Print closest vector details
for i, idx in enumerate(closest_indices):
    vector_id = ids_list[idx]
    print(f"Cluster {i+1}, Vector ID closest to centroid: {vector_id}")
