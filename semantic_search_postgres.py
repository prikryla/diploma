import psycopg2
import psycopg2.extras
import numpy as np
import os
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import time

# Load the Sentence Transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Compute the query vector
search_query = "WTA tennis tournament"
query_vector = model.encode(search_query)

# Function to compute cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

load_dotenv()

# Connection parameters using environment variables
conn_params = {
    'dbname': os.getenv('PG_DBNAME'),
    'user': os.getenv('PG_USER'),
    'password': os.getenv('PG_PASSWORD'),
    'host': os.getenv('PG_HOST')
}

# Dictionary to map class indexes to topics
class_to_topic = {
    1: 'Word',
    2: 'Sport',
    3: 'Business',
    4: 'Sci/Tech'
}

# Start tracking time for database query
start_time = time.time()

conn = psycopg2.connect(**conn_params)
cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

# SQL query to retrieve ID, embeddings, and other data
sql_query = """
SELECT id, class_index, title, description, embedding
FROM ag_dataset
"""
cur.execute(sql_query)

# List to hold results
results = []

# Iterate through each row in the result
for row in cur:
    # Ensure class_index is treated as integer
    class_index = int(row['class_index'])  # Convert class_index to int
    entry_id, title, description, stored_embedding_blob = row['id'], row['title'], row['description'], row['embedding']
    # Convert the stored BLOB back to numpy array
    stored_embedding = np.frombuffer(stored_embedding_blob, dtype=np.float32)
    # Calculate similarity
    similarity = cosine_similarity(query_vector, stored_embedding)
    # Determine topic from class index
    topic = class_to_topic.get(class_index, 'Unknown')  # Use converted class_index
    # Append results including similarity and topic
    results.append((entry_id, topic, title, description, similarity))

# Close the cursor and connection
cur.close()
conn.close()

# End tracking time for database query
end_time = time.time()
query_time = end_time - start_time
print(f"Querying time: {query_time} seconds")

# Sort results by similarity and get top 5
results.sort(key=lambda x: x[4], reverse=True)  # sort by similarity in descending order
top_5_results = results[:5]

# Display the top 5 results with better formatting
print("Top 5 Similar Entries:")
for i, (entry_id, topic, title, description, similarity) in enumerate(top_5_results, start=1):
    print(f"\nResult {i}:")
    print(f"  ID: {entry_id}")
    print(f"  Topic: {topic}")
    print(f"  Title: {title}")
    print(f"  Description: {description[:150]}...")  # Print first 150 characters of description
    print(f"  Similarity Score: {similarity:.4f}")  # Format similarity to 4 decimal places
