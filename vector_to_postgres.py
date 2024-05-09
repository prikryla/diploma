import pandas as pd
import psycopg2
import psycopg2.extras
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

# Precompute embeddings for all descriptions
embeddings = model.encode(df['description'].tolist(), convert_to_tensor=False)
df['embedding'] = [e.astype(np.float32).tobytes() for e in embeddings]

# Define the batch size
batch_size = 1000  # Adjust batch size based on memory and database handling capacity

# SQL insert statement
insert_query = """
    INSERT INTO ag_dataset_full (class_index, title, description, embedding) VALUES %s;
"""

# Connect to the PostgreSQL server using context managers
with psycopg2.connect(**conn_params) as conn:
    with conn.cursor() as cursor:
        start_time = time.time()

        # Batch insert data
        for i in range(0, len(df), batch_size):
            batch = [
                (row['class_index'], row['title'], row['description'], row['embedding'])
                for index, row in df.iloc[i:i+batch_size].iterrows()
            ]
            psycopg2.extras.execute_values(cursor, insert_query, batch, template=None, page_size=100)

        # Commit the transaction
        conn.commit()

        # Calculate execution time
        end_time = time.time()
        print(f"Data upload completed in {end_time - start_time} seconds.")
