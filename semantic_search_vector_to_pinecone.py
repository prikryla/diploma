import pinecone
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load CSV data
csv_file_path = 'test_bez_upravy.csv'
df = pd.read_csv(csv_file_path, delimiter=';')

# Load pre-trained Sentence Transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Connect to the Pinecone index
pinecone.init(api_key="eeb95b42-5012-4301-854b-7d75ae9fd293", environment="gcp-starter")
index_name = "semantic"  # Replace with your desired index name

# Specify the correct dimension
index_dimension = 384
# pinecone.create_index(index_name, dimension=index_dimension)
index = pinecone.Index(index_name)

# Function to create embeddings and save to Pinecone
def create_and_save_embeddings():
    embeddings = []

    for idx, description in enumerate(df['description']):
        # Generate embeddings for the description text
        description_embedding = model.encode(description, convert_to_tensor=True)
        embedding_list = description_embedding.numpy().tolist()
        embeddings.append({'id': str(idx), 'values': embedding_list})

    # Define batch size
    batch_size = 1000

    # Save vectors to Pinecone index in batches
    for i in range(0, len(embeddings), batch_size):
        batch = embeddings[i:i + batch_size]
        index.upsert(vectors=batch)

if __name__ == "__main__":
    create_and_save_embeddings()
