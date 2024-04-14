import preprocessing_images
import psycopg2
import pickle

img = preprocessing_images.preprocess_and_resize_png('0001.png')

print(img)

conn_params = {
    'dbname': 'image_similarity',
    'user': 'postgres',
    'password': 'postgres',
    'host': '127.0.0.1'
}

# Connect to the PostgreSQL database
conn = psycopg2.connect(**conn_params)
cur = conn.cursor()

