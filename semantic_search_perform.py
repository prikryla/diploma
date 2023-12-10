import pinecone
from sentence_transformers import SentenceTransformer
import pandas as pd

# Load pre-trained Sentence Transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Connect to the Pinecone index
pinecone.init(api_key="eeb95b42-5012-4301-854b-7d75ae9fd293", environment="gcp-starter")
index_name = "semantic"  # Replace with your index name
index = pinecone.Index(index_name)

# Load the dataset (assuming you have a DataFrame df with 'id' and 'description' columns)
# If you have a separate CSV file, load it using pd.read_csv()
# For example, df = pd.read_csv('your_dataset.csv')
# Make sure it has the same format as the one you used to create the index
csv_file_path = 'test_bez_upravy.csv'
df = pd.read_csv(csv_file_path, delimiter=';')

# Encode the query text
query_text = "Fears for T N pension after talks"
query_embedding = model.encode(query_text, convert_to_tensor=True).numpy().tolist()

# Perform semantic search
results = index.query(vector=[query_embedding], top_k=5)

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
