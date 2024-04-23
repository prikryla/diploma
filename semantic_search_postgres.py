import psycopg2
import psycopg2.extras
from sentence_transformers import SentenceTransformer
import numpy as np

# Connection parameters
conn_params = {
    'dbname': 'diploma_2',
    'user': 'postgres',
    'password': 'postgres',
    'host': '127.0.0.1'
}

# Load the Sentence Transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Compute the query vector
query = "WTA tennis tournament"
query_vector = model.encode(query)

# Function to compute cosine similarity
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Connect to the database
conn = psycopg2.connect(**conn_params)
cur = conn.cursor(cursor_factory=psycopg2.extras.DictCursor)

# Query to retrieve embeddings and other data
query = """
SELECT class_index, title, description, embedding
FROM diploma_semantic_search
"""
cur.execute(query)

# List to hold results
results = []

# Iterate through each row in the result
for row in cur:
    class_index, title, description, stored_embedding = row
    # Convert the stored embedding from list back to numpy array
    stored_embedding = np.array(stored_embedding)
    # Calculate similarity
    similarity = cosine_similarity(query_vector, stored_embedding)
    # Append results including similarity
    results.append((class_index, title, description, similarity))

# Close the cursor and connection
cur.close()
conn.close()

# Sort results by similarity and get top 5
results.sort(key=lambda x: x[3], reverse=True)  # sort by similarity in descending order
top_5_results = results[:5]

# Display the top 5 results
# Display the top 5 results with better formatting
print("Top 5 Similar Entries:")
for i, (class_index, title, description, similarity) in enumerate(top_5_results, start=1):
    print(f"\nResult {i}:")
    print(f"  Class Index: {class_index}")
    print(f"  Title: {title}")
    print(f"  Description: {description[:150]}...")  # Print first 150 characters of description
    print(f"  Similarity Score: {similarity:.4f}")  # Format similarity to 4 decimal places
