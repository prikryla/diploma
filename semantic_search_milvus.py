import json
import os
import time
import numpy as np
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
token = os.getenv('MILVUS_TOKEN')
endpoint = os.getenv('MILVUS_ENDPOINT')

# Milvus connection details
CLUSTER_ENDPOINT = endpoint
TOKEN = token

# Initialize Milvus client
client = MilvusClient(uri=CLUSTER_ENDPOINT, token=TOKEN)

# Load a pre-trained sentence transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Check if the collection exists, if not, create it with appropriate settings
collection_name = "semanticsearch"
if not client.has_collection(collection_name):
    client.create_collection(
        collection_name,
        fields=[
            {"name": "embedding", "type": "FLOAT_VECTOR", "params": {"dim": 384}},
            {"name": "title", "type": "VARCHAR"},
            {"name": "description", "type": "VARCHAR"},
            {"name": "topic", "type": "VARCHAR"}
        ],
        primary_field="id",
        auto_id=False
    )

# Your query sentence
query_sentence = "Innovative technologies reshape sports and international business practices"

# Convert the query sentence to an embedding and normalize it for cosine similarity
query_embedding = model.encode(query_sentence, convert_to_tensor=True).tolist()
normalized_query_embedding = query_embedding / np.linalg.norm(query_embedding)

# Convert the embedding into a list format that Milvus can understand
query_embedding_list = [normalized_query_embedding.tolist()]

start_time = time.time()
results = client.search(
    collection_name=collection_name,
    data=query_embedding_list,
    limit=10,
    search_params={"metric_type": "COSINE", "params": {}},
    output_fields=["id", "title", "description", "topic"]  # Specify fields to retrieve
)
end_time = time.time()

# Calculate the response time
final_time = end_time - start_time
print(f"Response time = {final_time} seconds")

# Process and format the search results
if results:
    top_5_results = [
        (
            result.entity.id,
            result.entity.get('topic', ''),
            result.entity.get('title', ''),
            result.entity.get('description', ''),
            result.score
        )
        for result in results[0]  # Assuming the first result set corresponds to the query
    ]

    for i, (entry_id, topic, title, description, similarity) in enumerate(top_5_results, start=1):
        print(f"\nResult {i}:")
        print(f"  ID: {entry_id}")
        print(f"  Topic: {topic}")
        print(f"  Title: {title}")
        print(f"  Description: {description[:150]}...")  # Print first 150 characters of description
        print(f"  Similarity Score: {similarity:.4f}")  # Format similarity to 4 decimal places
else:
    print("No results found.")
