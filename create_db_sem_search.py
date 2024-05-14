import psycopg2
import os

from psycopg2 import sql
from dotenv import load_dotenv

def establish_database_connection():
    """
    Establishes connection to the PostgreSQL server using environment variables.

    Returns:
        psycopg2.connection: Connection object to the PostgreSQL server.
    """
    # Load environment variables from .env file
    load_dotenv()

    # Connection parameters
    conn_params = {
        'dbname': os.getenv('PG_DBNAME'),
        'user': os.getenv('PG_USER'),
        'password': os.getenv('PG_PASSWORD'),
        'host': os.getenv('PG_HOST')
    }

    # Connect to the PostgreSQL server
    conn = psycopg2.connect(**conn_params)
    return conn

def create_tables(cursor):
    """
    Creates necessary tables in the PostgreSQL database if they don't exist.

    Args:
        cursor (psycopg2.cursor): Cursor object for executing SQL commands.
    """
    create_table_query = """
        CREATE TABLE IF NOT EXISTS ag_dataset (
            id SERIAL PRIMARY KEY,
            class_index VARCHAR,
            title VARCHAR,
            description TEXT,
            embedding BYTEA
        );
    """

    cursor.execute(create_table_query)

    create_sentiment_table_query = """
    CREATE TABLE IF NOT EXISTS ag_dataset_sentiment (
        data_id INTEGER PRIMARY KEY,
        subjectivity FLOAT,
        polarity FLOAT,
        FOREIGN KEY (data_id) REFERENCES ag_dataset(id) ON DELETE CASCADE
    );
    """
    cursor.execute(create_sentiment_table_query)

def main():
    """
    Main function to establish database connection, create tables, commit changes, and close connection.
    """
    conn = establish_database_connection()
    cursor = conn.cursor()

    create_tables(cursor)

    # Commit the changes
    conn.commit()

    # Close the cursor and connection
    cursor.close()
    conn.close()

if __name__ == "__main__":
    main()
