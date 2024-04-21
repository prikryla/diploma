import os
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

# Load pre-trained Sentence Transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Initialize Pinecone client
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
        metric='cosine',  # You might want to adjust the metric based on your use case
        spec=ServerlessSpec(
            cloud='aws',
            region='us-east-1'
        )
    )

index = pc.Index(index_name)

# Encode the query text
query_text = "tennis tournament"
query_embedding = model.encode(query_text, convert_to_tensor=True).tolist()

# Perform semantic search
results = index.query(
    vector=[query_embedding],
    top_k=5,
    include_metadata=True  # Make sure to include this to retrieve metadata
)

# Retrieve the relevant information from the metadata directly
search_results = []
for result in results['matches']:
    idx = int(result['id'])
    description = result['metadata'].get('description', 'No description available')
    similarity = result['score']
    topic = result['metadata'].get('topic', 'Unknown')  # Retrieve topic from metadata
    search_results.append({'id': idx, 'description': description, 'similarity': similarity, 'topic': topic})

# Print the search results
print("Search results for query:", query_text)
for result in search_results:
    print(f"ID: {result['id']}\nDescription: {result['description']}\nTopic: {result['topic']}\nSimilarity: {result['similarity']}\n")
