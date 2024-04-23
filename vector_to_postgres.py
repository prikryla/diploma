import pandas as pd
import psycopg2
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Connect to the PostgreSQL server
conn_params = {
    'dbname': 'diploma_2',
    'user': 'postgres',
    'password': 'postgres',
    'host': '127.0.0.1'
}
conn = psycopg2.connect(**conn_params)
cursor = conn.cursor()

# Read CSV file
df = pd.read_csv('test_bez_upravy.csv', delimiter=';')

# Iterate over rows
for index, row in df.iterrows():
    # Extract description from CSV row
    description = row['description']
    
    # Preprocess description (optional)
    # For example, remove special characters, punctuation, etc.
    # description = preprocess(description)
    
    # Tokenize and get the mean vector of the tokens
    doc = nlp(description)
    embedding = doc.vector.tolist()  # Convert embedding array to list
    
    # Insert data into PostgreSQL table
    insert_query = """
        INSERT INTO diploma_semantic_search (class_index, title, description, embedding)
        VALUES (%s, %s, %s, %s)
    """
    data = (row['class_index'], row['title'], row['description'], embedding)
    cursor.execute(insert_query, data)

# Commit the changes
conn.commit()

# Close the cursor and connection
cursor.close()
conn.close()
