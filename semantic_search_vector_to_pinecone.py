from pinecone import Pinecone, ServerlessSpec
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load CSV data
csv_file_path = 'test_bez_upravy.csv'
df = pd.read_csv(csv_file_path, delimiter=';')

# Load pre-trained Sentence Transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

pc = Pinecone(
    api_key="eeb95b42-5012-4301-854b-7d75ae9fd293"  # Replace with your actual API key
)

# Define the index parameters
index_name = "semanticsearch"  # Replace with your actual index name
index_dimension = 384  # Assuming your Sentence Transformer model has an output dimension of 384

# Check if the index already exists, if not, create it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=index_dimension,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='gcp',
            region='us-central1'
        )
    )

index = pc.Index(index_name)

# Function to create embeddings, add metadata, and save to Pinecone
def create_and_save_embeddings():
    embeddings = []

    for idx, row in df.iterrows():
        description = row['description']
        class_index = row['class_index']

        # Generate embeddings for the description text
        description_embedding = model.encode(description, convert_to_tensor=True)
        embedding_list = description_embedding.numpy().tolist()

        # Define metadata based on class_index
        topic = {1: 'Word', 2: 'Sport', 3: 'Business', 4: 'Sci/Tech'}.get(class_index, 'Unknown')

        embeddings.append({
            'id': str(idx),
            'values': embedding_list,
            'metadata': {'topic': topic}
        })

    # Define batch size for upsert
    batch_size = 1000

    # Save vectors to Pinecone index in batches
    for i in range(0, len(embeddings), batch_size):
        batch = embeddings[i:i + batch_size]
        index.upsert(vectors=batch)

if __name__ == "__main__":
    create_and_save_embeddings()
