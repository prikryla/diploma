from pymilvus import Collection, connections, list_collections, DataType, MilvusClient, FieldSchema, CollectionSchema, IndexType
from sentence_transformers import SentenceTransformer
import pandas as pd
import numpy as np

# Connect to Milvus
connections.connect()

# Define the collection parameters
collection_name = "news_articles"  # Replace with your collection name

# Load data from the .csv file with semicolon delimiter
csv_file_path = 'test_bez_upravy.csv'  # Replace with the path to your .csv file
df = pd.read_csv(csv_file_path, delimiter=';')

# Extract the 'description' column
descriptions = df['description'].tolist()

# Convert text descriptions to embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
embeddings = model.encode(descriptions)

# Adjust the dimensionality of the embeddings to match the collection's field dimension
expected_dimension = 384
if embeddings.shape[1] != expected_dimension:
    embeddings = embeddings[:, :expected_dimension]  # Truncate or pad the embeddings to match the expected dimension

# Check if the collection already exists, if not, create it with the updated schema
client = MilvusClient()
if collection_name not in list_collections():
    field = FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=expected_dimension)
    primary_field = FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True)
    schema = CollectionSchema(fields=[field, primary_field])
    client.create_collection(collection_name=collection_name, schema=schema)

# Insert vectors into the Milvus collection
collection = Collection(name=collection_name)

# Prepare data in the correct format for insertion
data = [{'embedding': embed.tolist()} for embed in embeddings]

collection.insert(data=data)

index_params = {
  "metric_type":"L2",
  "index_type":"IVF_FLAT",
  "params":{"nlist":1024}
}


# Create index
client.create_index(collection_name, field_name="embedding", index_params=index_params)

# Load collection
collection.load()

# Encode the query text
query_text = "email scam"

# Convert the query text to embeddings
query_embedding = model.encode([query_text])[0]

# Perform semantic search
results = collection.search(data=[query_embedding.tolist()], anns_field="embedding", param={"metric_type": "COSINE"}, limit=5)

# Retrieve the relevant information from the DataFrame based on the search results
search_results = []
for result in results:
    # Convert the result back to numpy array
    result_embedding = np.array(result.embedding)
    idx = int(result.id)
    description = df.loc[df['id'] == idx, 'description'].values[0]
    similarity = result.distance
    search_results.append({'id': idx, 'description': description, 'similarity': similarity})
# Print the search results
print("Search results for query:", query_text)
for result in search_results:
    print(f"ID: {result['id']}\nDescription: {result['description']}\nSimilarity: {result['similarity']}\n")
