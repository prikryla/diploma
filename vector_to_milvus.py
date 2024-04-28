import pandas as pd
import os

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

# Check if the collection exists
collection_name = "semanticsearch"
if not client.has_collection(collection_name):
    # Create the collection if it does not exist
    client.create_collection(
        collection_name=collection_name,
        dimension=384,
        metric_type="COSINE"
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
data['topic'] = data['class_index'].map({1: 'Word', 2: 'Sport', 3: 'Business', 4: 'Sci/Tech'}.get)

# Prepare the data in the required format for insertion
to_insert = [
    {
        "id": int(row['id']), 
        "vector": row['vector'],
        "topic": row['topic'],
        "title": row['title'],
        "description": row['description'][:1024]
    }
    for index, row in data.iterrows()
]

# Insert embeddings into the collection
res = client.insert(
    collection_name=collection_name,
    data=to_insert,
)

print("Insertion response:", res)
