import json
import pandas as pd
import os
import time

from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
token = os.getenv('MILVUS_TOKEN')
endpoint = os.getenv('MILVUS_ENDPOINT')

# Milvus connection details
CLUSTER_ENDPOINT = endpoint
TOKEN = token

client = MilvusClient(uri=CLUSTER_ENDPOINT, token=TOKEN)

# Load a pre-trained sentence transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Your query sentence
query_sentence = "Technology company released new software"

# Convert the query sentence to an embedding
query_embedding = model.encode(query_sentence)

# Convert the embedding into a list format that Milvus can understand
query_embedding_list = [query_embedding.tolist()]

start_time = time.time()
results = client.search(
    collection_name="semanticsearch",
    data=query_embedding_list,
    limit=10,
    search_params={"metric_type": "COSINE", "params": {}} 
)
end_time = time.time()

final_time = end_time - start_time
print(f"Response time = {final_time}")
result = json.dumps(results, indent=4)
print(result)
