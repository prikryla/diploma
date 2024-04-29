import pandas as pd
import psycopg2
import numpy as np
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

# Load environment variables from .env file
load_dotenv()

# Connection parameters using environment variables
conn_params = {
    'dbname': os.getenv('PG_DBNAME'),
    'user': os.getenv('PG_USER'),
    'password': os.getenv('PG_PASSWORD'),
    'host': os.getenv('PG_HOST')
}

# Load the Sentence Transformer model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Read CSV file
df = pd.read_csv('test_bez_upravy.csv', delimiter=';')

# Using context manager for handling connection and cursor
with psycopg2.connect(**conn_params) as conn:
    with conn.cursor() as cursor:
        for index, row in df.iterrows():
            description = row['description']
            # Generate embedding using Sentence Transformer
            embedding = model.encode(description, convert_to_tensor=False)  # Get the embedding as a numpy array
            binary_embedding = embedding.astype(np.float32).tobytes()  # Convert numpy array to bytes for BLOB storage
            
            insert_query = """
                INSERT INTO diploma_semantic (class_index, title, description, embedding)
                VALUES (%s, %s, %s, %s)
            """
            data = (row['class_index'], row['title'], row['description'], binary_embedding)
            cursor.execute(insert_query, data)
        conn.commit()
