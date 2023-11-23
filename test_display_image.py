import psycopg2
from PIL import Image
from io import BytesIO
import numpy as np

# Connect to the PostgreSQL database
conn_params = {
    'dbname': 'image_similarity',
    'user': 'postgres',
    'password': 'postgres',
    'host': '127.0.0.1'
}

conn = psycopg2.connect(**conn_params)
cur = conn.cursor()

# Retrieve the image from the database
cur.execute("SELECT image_data FROM cifar_images LIMIT 1")
record = cur.fetchone()

# Convert bytea to NumPy array
image_np_array = np.frombuffer(record[0], dtype=np.uint8)

# Convert NumPy array to PIL Image
image = Image.fromarray(image_np_array)

# Display or process the image as needed
image.show()

cur.close()
conn.close()