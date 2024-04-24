import psycopg2
from psycopg2 import sql

# Connection parameters
conn_params = {
    'dbname': 'diploma_2',
    'user': 'postgres',
    'password': 'postgres',
    'host': '127.0.0.1'
}

# Connect to the PostgreSQL server
conn = psycopg2.connect(**conn_params)
cursor = conn.cursor()

create_table_query = """
    CREATE TABLE diploma_semantic_search (
        id SERIAL PRIMARY KEY,
        class_index VARCHAR,
        title VARCHAR,
        description TEXT,
        embedding FLOAT[]
    );
"""

cursor.execute(create_table_query)

# Commit the changes
conn.commit()

# Close the cursor and connection
cursor.close()
conn.close()