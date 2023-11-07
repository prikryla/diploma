from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.models import Model
import numpy as np
import pinecone

api_key = 'eeb95b42-5012-4301-854b-7d75ae9fd293'
pinecone.init(api_key=api_key, environment="gcp-starter")

# Load the VGG16 model pre-trained on ImageNet data
base_model = VGG16(weights='imagenet')
model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc2').output)

# Create a Pinecone index
index_name = 'first-test-cifar10-image'
dimension = 4096  # Replace with the correct dimension based on your VGG16 model
#pinecone.create_index(index_name, dimension=dimension)


embeddings_list = []
# Loop over 1000 images
for i in range(1, 101):
    img_path = f'cifar10/test/airplane/airplane_{i:04}.png'  # Adjust the path to match your image naming convention

    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Get the features (embedding) for the image
    embedding = model.predict(x)

    # Define additional metadata for the image
    metadata = {"category": "aeroplane", "type": "test"}

    # Append image name, embedding, and metadata to the list
    embeddings_list.append((f'aeroplane_{i:04}.png', embedding[0].tolist(), metadata))

index = pinecone.Index("first-test-cifar10-image")
index.upsert(embeddings_list)