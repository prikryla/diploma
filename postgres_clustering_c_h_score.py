import psycopg2
import psycopg2.extras
import numpy as np
import os
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from sklearn.metrics import calinski_harabasz_score
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
start_time_query = time.time()  # Start measuring time for querying data
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
end_time_vector_conversion = time.time()  # End measuring time for converting data to vectors
print(f"Overall vector conversion time: {end_time_vector_conversion - start_time_query} seconds")

# Combine vectors with polarity and subjectivity for clustering
features = np.hstack((vectors, polarity, subjectivity))

# Define a range of cluster numbers to test
cluster_numbers = range(2, 7)  # From 2 to 6 clusters
calinski_scores = []

# Evaluate Calinski-Harabasz Index for each number of clusters
for n_clusters in cluster_numbers:
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(features)
    ch_score = calinski_harabasz_score(features, cluster_labels)
    calinski_scores.append(ch_score)
    print(f"Calinski-Harabasz Index for {n_clusters} clusters: {ch_score}")

# Plot Calinski-Harabasz Scores
plt.figure(figsize=(10, 5))
plt.plot(cluster_numbers, calinski_scores, marker='o')
plt.xlabel('Počet shluků')
plt.ylabel('Calinski-Harabaszovo skóre')
plt.title('Calinski-Harabaszovo skóre pro různé počty shluků')
plt.savefig('C-H graf')
plt.show()
