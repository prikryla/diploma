import numpy as np
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from sklearn.metrics.pairwise import cosine_similarity


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

# Replace 'path_to_cifar_image.png' with the path to your CIFAR-10 image
cifar_image_ship = 'cifar10/test/ship/ship_0001.png'
cifar_image_deer = 'cifar10/test/ship/ship_0002.png'
# Load and preprocess the image
img_ship = image.load_img(cifar_image_ship, target_size=(32, 32))
img_deer = image.load_img(cifar_image_deer, target_size=(32, 32))
img_array_ship = image.img_to_array(img_ship)
img_array_deer = image.img_to_array(img_deer)

# Get the embedding for the image
embedding_ship = get_embedding_from_image(img_array_ship)
embedding_deer = get_embedding_from_image(img_array_deer)

# Now, 'embedding' contains the feature representation of the image
#print(embedding_ship)
#print(embedding_deer)

embedding1_2d = embedding_ship.reshape(1, -1)
embedding2_2d = embedding_deer.reshape(1, -1)

similarity_score = cosine_similarity(embedding2_2d, embedding1_2d)[0, 0]

print(f"Cosine Similarity: {similarity_score}")