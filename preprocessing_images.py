import numpy as np
import psycopg2
from PIL import Image
from pathlib import Path

def preprocess_png(image_path):
    # Open and load the PNG image
    img = Image.open(image_path)

    # Convert the PIL Image to a NumPy array
    img_array = np.array(img)

    # Normalize pixel values to the range [0, 1]
    img_array = img_array / 255.0

    # Ensure the image has the correct dimensions (32x32 pixels)
    if img_array.shape != (32, 32, 3):
        raise ValueError("Image should have dimensions 32x32 pixels.")

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

directory = Path('cifar10/test/truck')
image_files = list(directory.glob('*.png'))

# Get the number of image files in the directory
number_of_images = len(image_files)

# Loop through each file in the directory
for file_path in image_files:
    # Read and preprocess the original image
    preprocessed_image = preprocess_png(str(file_path))

    # Convert NumPy array to BYTEA suitable for PostgreSQL
    serialized_array = psycopg2.Binary(preprocessed_image.tobytes())

    try:
        cur.execute(
            f"INSERT INTO cifar_images (image_name, image_data, image_category, image_type) VALUES (%s, %s, %s, %s)",
            (file_path.name, serialized_array, 'truck', 'test'))
        conn.commit()
        print("Record inserted successfully.")
    except Exception as e:
        print(f"Error inserting data: {e}")
        conn.rollback()

cur.close()
conn.close()
