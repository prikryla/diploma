import preprocessing_images
import psycopg2
import pickle

img = preprocessing_images.preprocess_and_resize_png('0001.png')

print(img)
