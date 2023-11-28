from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input, Flatten, Dense
import numpy as np
import pinecone

# Initialize Pinecone with your API key
api_key = 'eeb95b42-5012-4301-854b-7d75ae9fd293'
pinecone.init(api_key=api_key, environment="gcp-starter")

# Load the VGG16 model pre-trained on ImageNet data
input_tensor = Input(shape=(32, 32, 3))  # Adjust input shape
base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)

# Add new layers to accommodate the desired input shape
x = Flatten()(base_model.output)
x = Dense(3072, activation='relu', name='fc1')(x)
x = Dense(3072, activation='relu', name='fc2')(x)
output = Dense(3072, activation='softmax')(x)  # Adjust output size based on your needs

# Create a new model
model = Model(inputs=base_model.input, outputs=output)

# Create a Pinecone index
index_name = 'cifar'
dimension = 3072  # Adjust the dimension based on your VGG16 model
#pinecone.create_index(index_name, dimension=dimension)

embeddings_list = []

# Loop over 1000 images 
for i in range(1, 101):
    img_path = f'cifar10/test/airplane/airplane_{i:04}.png'

    img = image.load_img(img_path, target_size=(32, 32))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    # Get the features (embedding) for the image
    embedding = model.predict(x)

    # Define additional metadata for the image
    metadata = {"category": "aeroplane", "type": "test", "image_name": f'aeroplane_{i:04}.png'}

    # Specify the 'id' field explicitly, ensuring its length is less than or equal to 512
    id_value = f'image_{i:04}'
    
    # Append id, embedding, and metadata to the list
    embeddings_list.append((id_value, embedding[0].tolist(), metadata))

# Create the Pinecone index
index = pinecone.Index(index_name)

# Upsert embeddings and metadata into the index
index.upsert(embeddings_list)
