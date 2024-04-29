import pandas as pd
import psycopg2
import spacy
import numpy as np
import os

from dotenv import load_dotenv

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Load environment variables from .env file
load_dotenv()

# Connection parameters using environment variables
conn_params = {
    'dbname': os.getenv('PG_DBNAME'),
    'user': os.getenv('PG_USER'),
    'password': os.getenv('PG_PASSWORD'),
    'host': os.getenv('PG_HOST')
}

# Connect to the PostgreSQL server
conn = psycopg2.connect(**conn_params)
cursor = conn.cursor()

# Read CSV file
df = pd.read_csv('test_bez_upravy.csv', delimiter=';')

# Iterate over rows
for index, row in df.iterrows():
    # Extract description from CSV row
    description = row['description']
    
    # Tokenize and get the mean vector of the tokens
    doc = nlp(description)
    embedding = np.array(doc.vector, dtype=np.float32)  # Ensure it's a numpy array of type float32
    binary_embedding = embedding.tobytes()  # Convert numpy array to bytes, suitable for BLOB storage
    
    # Insert data into PostgreSQL table
    insert_query = """
        INSERT INTO diploma_semantic_search (class_index, title, description, embedding)
        VALUES (%s, %s, %s, %s)
    """
    data = (row['class_index'], row['title'], row['description'], binary_embedding)
    cursor.execute(insert_query, data)

# Commit the changes
conn.commit()

# Close the cursor and connection
cursor.close()
conn.close()
