import pinecone
import json

# Initialize Pinecone with your API key
pinecone.init(api_key="eeb95b42-5012-4301-854b-7d75ae9fd293", environment="gcp-starter")

# Specify the index name
index_name = "cifar"

# Replace "your_vector_id" with the actual ID you want to retrieve
vector_id_to_retrieve = "image_0060"

# Fetch the vector based on its ID
fetch_result = pinecone.Index(index_name).fetch([vector_id_to_retrieve])

# Extract the relevant information
result_dict = {
    "namespace": fetch_result.namespace,
    "vectors": {
        vector_id_to_retrieve: {
            "id": vector_id_to_retrieve,
            "values": fetch_result.vectors[vector_id_to_retrieve]['values']
        }
    }
}

main_vector = fetch_result.vectors[vector_id_to_retrieve]['values']

response = pinecone.Index(index_name).query(vector=[main_vector], top_k=5)

print(response)

# Extract 'id' and 'score' from the 'matches' list
matches = response.get('matches', [])
id_and_score_list = [(match.get('id'), match.get('score')) for match in matches]

# Print or use the extracted information
for id, score in id_and_score_list:
    print(f"ID: {id}, Score: {score}")