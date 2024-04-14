import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
import pandas as pd

# Load pre-trained Sentence Transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Initialize Pinecone client
pc = Pinecone(
    api_key="eeb95b42-5012-4301-854b-7d75ae9fd293"  # Replace with your API key
)

# Define the index parameters
index_name = "clustering"  # Replace with your index name
index_dimension = 768  # Assuming your Sentence Transformer model has an output dimension of 768

# Check if the index already exists, if not, create it
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=index_dimension,
        metric='cosine',  # You might want to adjust the metric based on your use case
        spec=ServerlessSpec(
            cloud='gcp',
            region='us-central1'
        )
    )

index = pc.Index(index_name)

# Load the dataset (assuming you have a DataFrame df with 'id' and 'description' columns)
csv_file_path = 'test_bez_upravy.csv'
df = pd.read_csv(csv_file_path, delimiter=';')

# Encode the query text
query_text = "tennis tournament"
query_embedding = model.encode(query_text, convert_to_tensor=True).tolist()

# Perform semantic search
results = index.query(
    vector=[query_embedding],
    top_k=5,
    include_values=True
)

# Retrieve the relevant information from the DataFrame based on the search results
search_results = []
for result in results['matches']:
    idx = int(result.id)
    description = df.loc[df['id'] == idx, 'description'].values[0]
    similarity = result.score
    search_results.append({'id': idx, 'description': description, 'similarity': similarity})

# Print the search results
print("Search results for query:", query_text)
for result in search_results:
    print(f"ID: {result['id']}\nDescription: {result['description']}\nSimilarity: {result['similarity']}\n")
