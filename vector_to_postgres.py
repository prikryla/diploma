import pandas as pd
import psycopg2
import numpy as np
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import time

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

# Read CSV file for text data
df = pd.read_csv('train_fixed.csv', delimiter=';')

# Connect to the PostgreSQL server using context managers
with psycopg2.connect(**conn_params) as conn:
    with conn.cursor() as cursor:
        # Start tracking time for ag_dataset
        start_time_ag_dataset = time.time()

        for index, row in df.iterrows():
            description = row['description']

            # Generate embedding using Sentence Transformer
            embedding = model.encode(description, convert_to_tensor=False)
            binary_embedding = embedding.astype(np.float32).tobytes()

            # Insert text data and embedding into ag_dataset
            insert_query = """
                INSERT INTO ag_dataset_full (class_index, title, description, embedding)
                VALUES (%s, %s, %s, %s) RETURNING id
            """
            data = (row['class_index'], row['title'], row['description'], binary_embedding)
            cursor.execute(insert_query, data)
            cursor.fetchone()  # Fetch the ID, but don't use it for ag_dataset_sentiment insertion

        # Calculate execution time for ag_dataset
        end_time_ag_dataset = time.time()
        execution_time_ag_dataset = end_time_ag_dataset - start_time_ag_dataset
        print(f"Data upload completed for ag_dataset_full in {execution_time_ag_dataset} seconds.")

        conn.commit()
