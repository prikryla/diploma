import psycopg2
import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import numpy as np
from PIL import Image
import PIL
import io

from sklearn.metrics.pairwise import cosine_similarity

# Connect to the PostgreSQL database
conn_params = {
    'dbname': 'image_similarity',
    'user': 'postgres',
    'password': 'postgres',
    'host': '127.0.0.1'
}

conn = psycopg2.connect(**conn_params)
cur = conn.cursor()

# Function to convert bytea to NumPy array
def bytea_to_np_array(bytea):
    return np.frombuffer(bytea, dtype=np.uint8)

# Function to preprocess and get embedding from an image
def get_embedding_from_image(image_array):
    # Use a pre-trained model (VGG16 in this case) for feature extraction
    model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))

    # Resize the image to the target size (32x32 pixels)
    img_array = image.img_to_array(image_array)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Get the embeddings
    embeddings = model.predict(img_array)

    # Flatten the embeddings to a 1D array
    embedding = embeddings.flatten()

    return embedding

# Database part
cur.execute("SELECT image_data FROM cifar_images_dimension_test WHERE image_id=2655 LIMIT 1;")
image_array_memory_view = cur.fetchone()[0]
image_array_bytes = bytes(image_array_memory_view)
image_np_array = np.frombuffer(image_array_bytes, dtype=np.uint8)

# Check the raw bytea data
#print("Raw Bytea Data:", image_array_bytes)

# Reverse byte order if necessary
image_np_array = np.clip(image_np_array, 0, 255)
image_np_array = image_np_array.byteswap().newbyteorder()

# Adjust dimensions based on the actual size of the image
height, width, channels = 32, 32, 3

# Ensure that the size matches the expected size
expected_size = height * width * channels
if image_np_array.size != expected_size:
    raise ValueError(f"Expected array size {expected_size}, but got {image_np_array.size}.")

# Reshape the NumPy array to the correct dimensions
image_np_array = image_np_array.reshape((height, width, channels))

# Convert NumPy array to PIL Image
img_pil = Image.fromarray(image_np_array)

# Save the image to a file (for debugging purposes)
#img_pil.save("debug_image.png")

# Display the image using an external viewer (e.g., open with default image viewer)
#img_pil.show()

# Replace 'path_to_cifar_image.png' with the path to your CIFAR-10 image
cifar_image_ship = 'cifar10/test/ship/ship_0001.png'
cifar_image_deer = 'cifar10/test/ship/ship_0002.png'

# Load and preprocess the image
img_ship = image.load_img(cifar_image_ship, target_size=(32, 32))
img_array_ship = image.img_to_array(img_ship)

img_deer = image.load_img(cifar_image_deer, target_size=(32, 32))
img_array_deer = image.img_to_array(img_deer)

img_array_automobile = image.img_to_array(img_pil)

# Get the embedding for the image
embedding_ship = get_embedding_from_image(img_array_ship)
embedding_deer = get_embedding_from_image(img_array_deer)
embedding_automobile = get_embedding_from_image(img_array_automobile)

embedding1_2d = embedding_ship.reshape(1, -1)
embedding2_2d = embedding_deer.reshape(1, -1)
embedding3_2d = embedding_automobile.reshape(1, -1)

similarity_score = cosine_similarity(embedding2_2d, embedding3_2d)[0, 0]

print(f"Cosine Similarity: {similarity_score}")
cur.close()
conn.close()
