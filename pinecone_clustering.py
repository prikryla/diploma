import pandas as pd
import numpy as np
import os

from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('API_KEY_PINECONE')

# Initialize Pinecone with your API key
pc = Pinecone(api_key=api_key)

# Load pre-trained Sentence Transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Define the index parameters
index_name = "clustering"  # Replace with your index name

# Retrieve the index
index = pc.Index(index_name)

# Load your data from the .csv file
data = pd.read_csv("test_bez_upravy.csv", sep=";")

# Function to find similar documents given a query text
def find_similar_documents(query_text, top_k=3):
    # Encode the query text
    query_embedding = model.encode(query_text, convert_to_tensor=True).tolist()

    # Query the vector database to retrieve similar documents
    query_result = index.query(vector=[query_embedding], top_k=top_k, include_values=True)

    # Extract document IDs and scores from the query result
    similar_document_ids = [item.id for item in query_result.items]
    similar_document_scores = [item.score for item in query_result.items]

    # Get the text of similar documents
    similar_documents = data[data['id'].isin(similar_document_ids)]

    # Add scores to the DataFrame
    similar_documents['score'] = similar_document_scores

    return similar_documents

# Example usage
query_text = "What is the capital of Spain?"
similar_documents = find_similar_documents(query_text)
print("Similar documents:")
for _, doc in similar_documents.iterrows():
    print(doc["title"], "-", doc["description"])
