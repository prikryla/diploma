import os
import time
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('API_KEY_PINECONE')

# Load pre-trained Sentence Transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Initialize Pinecone client
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

# Encode the query text
query_text = "Innovative technologies reshape sports and international business practices"
query_embedding = model.encode(query_text, convert_to_tensor=True).tolist()

start_time = time.time()

# Perform semantic search
results = index.query(
    vector=[query_embedding],
    top_k=10,
    include_metadata=True
)

end_time = time.time()

# Retrieve the relevant information from the metadata directly
search_results = []
for result in results['matches']:
    idx = int(result['id'])
    description = result['metadata'].get('description', 'No description available')
    similarity = result['score']
    title = result['metadata'].get('title', 'No title available')  # Corrected retrieval of title
    topic = result['metadata'].get('topic', 'Unknown')
    search_results.append({
        'id': idx, 'title': title, 'description': description, 
        'similarity': similarity, 'topic': topic
    })

print(f"Query time - {end_time - start_time} seconds")

# Print the search results
print("Search results for query:", query_text)
for result in search_results:
    print(f"ID: {result['id']}\nTitle: {result['title']}\nDescription: {result['description']}\nTopic: {result['topic']}\nSimilarity: {result['similarity']}\n")
