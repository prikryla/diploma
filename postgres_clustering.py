import psycopg2
import psycopg2.extras
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from dotenv import load_dotenv
import time

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

# SQL to fetch vectors along with polarity and subjectivity
sql = """
SELECT a.id, a.embedding, s.polarity, s.subjectivity
FROM ag_dataset a
JOIN ag_dataset_sentiment s ON a.id = s.data_id
ORDER BY a.id LIMIT 7600;
"""

# Execute the query
cursor.execute(sql)
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
n_clusters = 4
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
kmeans.fit(features)

# PCA for dimensionality reduction
pca = PCA(n_components=2)
features_reduced = pca.fit_transform(features)

# Plotting the clusters
# Plot setup
plt.figure(figsize=(10, 8))
cmap = ListedColormap(['#e41a1c', '#377eb8', '#4daf4a', '#ff7f00'])  # Red, Blue, Green, Orange
norm = BoundaryNorm(np.arange(0, n_clusters+1), cmap.N)

scatter = plt.scatter(features_reduced[:, 0], features_reduced[:, 1], c=kmeans.labels_, cmap=cmap, norm=norm, alpha=0.5)
cbar = plt.colorbar(scatter, label='Označení shluků')
plt.title('Shluky vytvořené na základě shlukování s dodatečnou informací s databází PostgreSQL')
plt.xlabel('Osa X - první komponenta')
plt.ylabel('Osa Y - druhá komponenta')

# Adjust the plot margins
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

plt.savefig('postgres_cluster_visualization.png')
plt.show()
