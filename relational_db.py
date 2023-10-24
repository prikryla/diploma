import base64

# Open the PNG image in binary mode and read its content
with open('0001.png', 'rb') as image_file:
    image_binary_data = image_file.read()

# Encode the binary data as base64
base64_encoded = base64.b64encode(image_binary_data)

# Convert the base64-encoded bytes to a base64 string
base64_string = base64_encoded.decode('utf-8')

# Print the base64 string
print(base64_string)