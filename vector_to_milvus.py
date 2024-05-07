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

# Check if the collection exists
collection_name = "agdataset"
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

# Define batch size for insertion
batch_size = 1000

# Prepare the data in the required format for insertion
to_insert = [
    {
        "id": int(row['id']), 
        "vector": row['vector'],
        "topic": row['topic'],
        "title": row['title'],
        "description": row['description'][:1024]
        # "polarity": row['polarity'],
        # "subjectivity": row['subjectivity']
    }
    for index, row in data.iterrows()
]

# Insert embeddings into the collection in batches
start_time = time.time()
try:
    for i in range(0, len(to_insert), batch_size):
        batch = to_insert[i:i+batch_size]
        res = client.insert(
            collection_name=collection_name,
            data=batch,
        )
        print("Insertion response:", res)
except Exception as e:
    print("An error occurred:", e)
finally:
    end_time = time.time()
    insertion_time = end_time - start_time
    print("Insertion time:", insertion_time)
