import numpy as np
import psycopg2

from PIL import Image
from pathlib import Path

def preprocess_and_resize_png(image_path, target_size=(24, 24)):
    # Open and load the PNG image
    img = Image.open(image_path)

    # Resize the image to the target size (24x24 pixels)
    img = img.resize(target_size)

    # Convert the PIL Image to a NumPy array
    img_array = np.array(img)

    # Normalize pixel values to the range [0, 1]
    img_array = img_array / 255.0

    # Ensure the image has the correct dimensions (24x24 pixels)
    if img_array.shape != (24, 24, 3):
        raise ValueError("Resized image should have dimensions 24x24 pixels.")

    # Expand dimensions to create a batch of one image
    img_array = np.expand_dims(img_array, axis=0)

    return img_array

conn_params = {
    'dbname': 'image_similarity',
    'user': 'postgres',
    'password': 'postgres',
    'host': '127.0.0.1'
}

# Connect to the PostgreSQL database
conn = psycopg2.connect(**conn_params)
cur = conn.cursor()

directory = Path('cifar10/test/airplane')
image_files = list(directory.glob('*.png'))

# Get the number of image files in the directory
number_of_images = len(image_files)


cur.execute("SELECT current_timestamp")
timestamp = cur.fetchone()[0]
date_time = timestamp.strftime('%Y-%m-%d %H:%M:%S') 


# Loop through each file in the directory
for file_path in image_files:
    cur.execute("SELECT current_timestamp")
    timestamp = cur.fetchone()[0]
    date_time = timestamp.strftime('%Y-%m-%d %H:%M:%S') 

    preprocessed_image = preprocess_and_resize_png(str(file_path))

    # Convert NumPy array to BYTEA suitable for PostgreSQL
    serialized_array = psycopg2.Binary(preprocessed_image.tobytes())

    try:
        cur.execute(f"INSERT INTO cifar_images (image_name, image_data, created_at) VALUES (%s, %s, %s)",
                    (file_path.name, serialized_array, date_time))
        conn.commit()
        print("Record inserted successfully.")
    except Exception as e:
        print(f"Error inserting data: {e}")
        conn.rollback()

cur.close()
conn.close()

