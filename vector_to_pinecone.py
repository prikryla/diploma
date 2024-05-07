import pandas as pd
import os
import time

from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('API_KEY_PINECONE')

# Load CSV data
csv_file_path = 'test_bez_upravy.csv'
df = pd.read_csv(csv_file_path, delimiter=';')

# Load pre-trained Sentence Transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

pc = Pinecone(
    api_key=api_key
)

# Define the index parameters
index_name = "semanticsearch"
index_dimension = 384

# Check if the index already exists, if not, create it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=index_dimension,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

index = pc.Index(index_name)

# Function to create embeddings and add metadata
def create_embeddings():
    embeddings = []

    for idx, row in df.iterrows():
        description = row['description']
        class_index = row['class_index']
        title = row['title']

        # Generate embeddings for the description text
        description_embedding = model.encode(description, convert_to_tensor=True)
        embedding_list = description_embedding.numpy().tolist()

        # Define metadata based on class_index
        topic = {1: 'Word', 2: 'Sport', 3: 'Business', 4: 'Sci/Tech'}.get(class_index, 'Unknown')

        embeddings.append({
            'id': str(idx),
            'values': embedding_list,
            'metadata': {'topic': topic, 'title': title, 'description': description}
        })

    return embeddings

# Function to upsert embeddings in batches and measure time
def upsert_embeddings():
    embeddings = create_embeddings()
    batch_size = 1000
    total_batches = len(embeddings) // batch_size + (len(embeddings) % batch_size > 0)
    start_time = time.time()

    for i in range(0, len(embeddings), batch_size):
        batch = embeddings[i:i + batch_size]
        index.upsert(vectors=batch)
        print(f"Processed batch {i // batch_size + 1} of {total_batches}")

    end_time = time.time()
    print(f"Time taken to upsert vectors in batches: {end_time - start_time} seconds")

if __name__ == "__main__":
    upsert_embeddings()
