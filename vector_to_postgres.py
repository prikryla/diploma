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

# Read CSV file for text data
df = pd.read_csv('test_bez_upravy.csv', delimiter=';')

# Read CSV file for enriched data containing subjectivity and polarity
df_enriched = pd.read_csv('enriched.csv', delimiter=',')

# Connect to the PostgreSQL server using context managers
with psycopg2.connect(**conn_params) as conn:
    with conn.cursor() as cursor:
        for index, row in df.iterrows():
            description = row['description']

            # Generate embedding using Sentence Transformer
            embedding = model.encode(description, convert_to_tensor=False)
            binary_embedding = embedding.astype(np.float32).tobytes()

            # Insert text data and embedding into ag_dataset
            insert_query = """
                INSERT INTO ag_dataset (class_index, title, description, embedding)
                VALUES (%s, %s, %s, %s) RETURNING id
            """
            data = (row['class_index'], row['title'], row['description'], binary_embedding)
            cursor.execute(insert_query, data)
            data_id = cursor.fetchone()[0]

            insert_sentiment_query = """
                INSERT INTO ag_dataset_sentiment (data_id, subjectivity, polarity)
                VALUES (%s, %s, %s)
            """

            enriched_row = df_enriched.iloc[index]
            sentiment_data = (data_id, enriched_row['subjectivity'], enriched_row['polarity'])
            cursor.execute(insert_sentiment_query, sentiment_data)

        conn.commit()
