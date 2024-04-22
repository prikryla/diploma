from pymilvus import MilvusClient
import pandas as pd
from sentence_transformers import SentenceTransformer

# Milvus connection details
CLUSTER_ENDPOINT = "https://in03-223185a7ed9d4b1.api.gcp-us-west1.zillizcloud.com"
TOKEN = "c3fdf5db6f838bc340e7d1b5c3f6888ed54679c11eb725e9b17a73435969fcb94beeb5a9c602b7d5a2ebc194b046f57d9b5e1ce3"

client = MilvusClient(uri=CLUSTER_ENDPOINT, token=TOKEN)

# Check if the collection exists
collection_name = "semanticsearch"
if not client.has_collection(collection_name):
    # Create the collection if it does not exist
    client.create_collection(
        collection_name=collection_name,
        dimension=384,  # Set this to the dimensionality of your embeddings
        metric_type="IP"  # Inner Product (cosine similarity)
    )
    print(f"Collection '{collection_name}' created.")
else:
    print(f"Collection '{collection_name}' already exists.")

# Load data from CSV
data = pd.read_csv("test_bez_upravy.csv", delimiter=';')

# Initialize the Sentence Transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create embeddings for the descriptions
data['vector'] = data['description'].apply(lambda desc: model.encode(desc).tolist())

# Prepare the data in the required format for insertion
to_insert = [
    {"id": int(row['id']), "vector": row['vector']} for index, row in data.iterrows()
]

# Insert embeddings into the collection
res = client.insert(
    collection_name=collection_name,
    data=to_insert,
)

print("Insertion response:", res)
