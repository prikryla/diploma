import json
import pandas as pd
from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

# Milvus connection details
CLUSTER_ENDPOINT = "https://in03-223185a7ed9d4b1.api.gcp-us-west1.zillizcloud.com"
TOKEN = "c3fdf5db6f838bc340e7d1b5c3f6888ed54679c11eb725e9b17a73435969fcb94beeb5a9c602b7d5a2ebc194b046f57d9b5e1ce3"

client = MilvusClient(uri=CLUSTER_ENDPOINT, token=TOKEN)

# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Your query sentence
query_sentence = "WTA tennis tournament"

# Convert the query sentence to an embedding
query_embedding = model.encode(query_sentence)

# Convert the embedding into a list format that Milvus can understand
query_embedding_list = [query_embedding.tolist()]

results = client.search(
    collection_name="semanticsearch",
    data=query_embedding_list,
    limit=5,
    search_params={"metric_type": "IP", "params": {"nprobe": 10}} 
)

result = json.dumps(results, indent=4)
print(result)
