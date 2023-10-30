from PIL import Image
import numpy as np

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