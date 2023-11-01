import psycopg2

# Connection parameters
conn_params = {
    'dbname': 'image_similarity',
    'user': 'postgres',
    'password': 'postgres',
    'host': '127.0.0.1'
}

# Connect to the default 'postgres' database (or another existing database)
# to create a new database
conn = psycopg2.connect(
    dbname='postgres',
    user=conn_params['user'],
    password=conn_params['password'],
    host=conn_params['host']
)
conn.autocommit = True

# Connect to the newly created database
conn = psycopg2.connect(**conn_params)
cur = conn.cursor()

# Create the table for 24x24 images
create_table_query = '''
CREATE TABLE IF NOT EXISTS cifar_images (
    image_id SERIAL PRIMARY KEY,
    image_name VARCHAR(255),
    image_data BYTEA,
    image_category VARCHAR(30)
    image_type VARCHAR(10)
    created_at TIMESTAMP
);
'''

#update_table_query = '''ALTER TABLE cifar_images ADD COLUMN image_type VARCHAR(30)'''
update_value_in_table_query = '''UPDATE cifar_images SET image_type = 'test' '''

#cur.execute(update_table_query)
cur.execute(update_value_in_table_query)
#cur.execute(create_table_query)
conn.commit()

# Close the cursor and connection
cur.close()
conn.close()
