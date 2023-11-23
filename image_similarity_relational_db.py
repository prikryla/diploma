import psycopg2
import numpy as np
from scipy.spatial.distance import cosine
from PIL import Image
from io import BytesIO

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
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
import numpy as np
from PIL import Image

def get_embedding_from_image(image_array):
    # Use a pre-trained model (VGG16 in this case) for feature extraction
    model = VGG16(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
    
    # Convert the NumPy array to a PIL Image
    img = Image.fromarray(image_array)

    # Resize the image to the target size (224x224 pixels)
    img = img.resize((32, 32))

    # Convert the PIL Image to a NumPy array
    img_array = image.img_to_array(img)

    # Ensure the image has three color channels
    if img_array.shape[-1] == 1:
        # Convert grayscale to RGB
        img_array = np.stack((img_array,) * 3, axis=-1)

    # Normalize pixel values to the range [0, 1]
    img_array = img_array / 255.0

    # Ensure the image has the correct dimensions (224x224 pixels)
    if img_array.shape[:2] != (32, 32):
        raise ValueError("Resized image should have dimensions 224x224 pixels.")

    # Remove singleton dimensions
    img_array = np.squeeze(img_array)

    # Expand dimensions to create a batch of one image
    x = np.expand_dims(img_array, axis=0)

    # Preprocess the input
    x = preprocess_input(x)
    
    # Get the embeddings
    embeddings = model.predict(x)
    
    # Flatten the embeddings to a 1D array
    embedding = embeddings.flatten()
    
    return embedding


# Specify the path to the query image
query_image_path = 'cifar10/test/airplane/airplane_0001.png'

# Read the query image and preprocess it
query_image = Image.open(query_image_path)
query_image_array = np.array(query_image)
query_embedding = get_embedding_from_image(query_image_array)

print("Query Image Similarity:")
print(f"Image: {query_image_path}")
print(f"Similarity Score: 1.0 (Query Image)")

# Retrieve embeddings from the database for the 'dog' category (limit to 20 for testing)
cur.execute("SELECT image_data FROM cifar_images WHERE image_category IN ('dog', 'automobile') LIMIT 2")
records = cur.fetchall()

# Assuming `result` is a tuple and contains the binary data at index 0
binary_data = records[0]

# Extract binary data if it's nested in a tuple or another structure
while isinstance(binary_data, tuple):
    binary_data = binary_data[0]  # Assuming binary data is at index 0


# Convert the memoryview to bytes (if needed)
if isinstance(binary_data, memoryview):
    binary_data = bytes(binary_data)

print(binary_data)

image_data = BytesIO(binary_data)

print(image_data)



# Calculate cosine similarity and store results
results = []

for record in records:
    # Convert bytea to NumPy array
    image_np_array = bytea_to_np_array(record[0])

    print(f"Converted bytea to np: {image_np_array}")

    # Get embedding for the image from the database
    database_embedding = get_embedding_from_image(image_np_array)

    print(f"Database embedding: {database_embedding}")

    # Calculate cosine similarity
    similarity_score = 1 - cosine(query_embedding, database_embedding)

    # Store results (image_data, similarity_score, database_embedding)
    results.append((record[0], similarity_score, database_embedding))

    print(results)

# Sort results by similarity scores
results.sort(key=lambda x: x[1], reverse=True)

# Print results for debugging
for image_data, cosine_similarity, _ in results:
    print("\nSimilar Image:")
    print(f"Similarity Score: {cosine_similarity}")
    # If you want to display the image, you can use the following code:
    # img = Image.open(BytesIO(image_data))
    # img.show()

# Close the database connection
cur.close()
conn.close()
