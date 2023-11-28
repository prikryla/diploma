import numpy as np
import psycopg2
from PIL import Image
from pathlib import Path
from datetime import datetime

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

def numpy_array_to_bytea(image_np_array):
    # Ensure the pixel values are in the correct range [0, 255]
    image_np_array_scaled = np.clip(image_np_array * 255, 0, 255).astype(np.uint8)

    # Convert the NumPy array to bytes and then to BYTEA
    bytea_data = psycopg2.Binary(image_np_array_scaled.tobytes())

    return bytea_data

def check_inserted_data(conn, cur):
    try:
        # Retrieve the saved data from the database
        cur.execute("SELECT image_data FROM cifar_images_dimension_test WHERE image_category = 'automobile' AND image_type = 'test';")
        serialized_array = cur.fetchone()[0]
        
        # Deserialize the data to a NumPy array
        preprocessed_image = np.frombuffer(bytes(serialized_array), dtype=np.uint8)
        preprocessed_image = preprocessed_image.reshape((32, 32, 3))

        # Check if the dimensions match
        if preprocessed_image.shape != (32, 32, 3):
            print("Error: Saved image data has unexpected dimensions.")
        else:
            print("Saved image data has the expected dimensions.")
        
    except Exception as e:
        print(f"Error checking data: {e}")

# Connect to the PostgreSQL database
conn_params = {
    'dbname': 'image_similarity',
    'user': 'postgres',
    'password': 'postgres',
    'host': '127.0.0.1'
}
conn = psycopg2.connect(**conn_params)
cur = conn.cursor()

directory = Path('cifar10/test/automobile')
image_files = list(directory.glob('*.png'))

current_timestamp = datetime.now()

# Loop through each file in the directory
for file_path in image_files:
    # Read and preprocess the original image
    preprocessed_image = preprocess_png(str(file_path))

    # Convert NumPy array to BYTEA suitable for PostgreSQL
    serialized_array = numpy_array_to_bytea(preprocessed_image)

    try:
        cur.execute(
            f"INSERT INTO cifar_images_dimension_test (image_name, image_data, created_at, image_category, image_type) VALUES (%s, %s, %s, %s, %s)",
            (file_path.name, serialized_array, current_timestamp, 'automobile', 'test'))
        conn.commit()
        print("Record inserted successfully.")

        # Check the saved data after each insert
        check_inserted_data(conn, cur)

    except Exception as e:
        print(f"Error inserting data: {e}")
        conn.rollback()

cur.close()
conn.close()