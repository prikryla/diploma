import psycopg2
from sentence_transformers import SentenceTransformer

# Connection parameters
conn_params = {
    'dbname': 'image_similarity',
    'user': 'postgres',
    'password': 'postgres',
    'host': '127.0.0.1'
}

# Load the Sentence Transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Connect to the PostgreSQL database
conn = psycopg2.connect(**conn_params)
cursor = conn.cursor()

def preprocess_text(text):
    # You might want to customize this function based on your specific requirements
    return text

def get_text_embedding(text):
    # Use the Sentence Transformer model to get the text embedding
    return model.encode(text).tolist()

def semantic_search_with_embeddings(search_query, cursor):
    # Preprocess the search query
    search_query = preprocess_text(search_query)

    # Get the embedding for the search query
    query_embedding = get_text_embedding(search_query)

    # Construct the SQL query for semantic search
    sql_query = f"""
        SELECT id, class_index, title, description,
               COSINE_SIMILARITY(
                   ARRAY[{', '.join(map(str, query_embedding))}]::double precision[],
                   ARRAY(
                       SELECT UNNEST(ARRAY[model.encode(description)::double precision]) FROM semantic_search
                   )
               ) AS similarity
        FROM semantic_search
        ORDER BY similarity DESC
        LIMIT 5;
    """

    # Execute the query
    cursor.execute(sql_query)

    # Fetch the results
    results = cursor.fetchall()

    return results

# Assuming search_query is defined
search_query = "tennis tournament"

# Perform semantic search with embeddings
search_results = semantic_search_with_embeddings(search_query, cursor)

# Display the results
for result in search_results:
    print(f"ID: {result[0]}, Class Index: {result[1]}, Title: {result[2]}, Description: {result[3]}, Similarity: {result[4]}\n")

# Close the cursor and connection
cursor.close()
conn.close()
