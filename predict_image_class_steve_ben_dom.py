import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import tensorflow as tf
from PIL import Image
import glob
import random
import pandas as pd

from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Input, AveragePooling2D
from tensorflow.keras.models import Sequential, load_model #
from tensorflow.keras import layers, preprocessing #
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split, GridSearchCV

import streamlit as st #
import os #
import tensorflow_hub as hub #

import os
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input
import tensorflow_hub as hub

st.header("Image class predictor")

def main():
    file_uploaded = st.file_uploader("Choose file", type = ['.jpg'])
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        figure = plt.figure()
        plt.imshow(image)
        plt.axis('off')
        result = predict_class(image)
        st.write(result)
        st.pyplot(figure)
                 
def predict_class(image):
    classifier_model = tf.keras.models.load_model(r'../code_ben/model_ben_kaggle.h5')
    shape = ((128, 128, 3))
    # model = tf.keras.Sequential([hub.KerasLayer(classifier_model, input_shape=shape)])
    test_image = image.resize((128, 128))
    test_image = preprocessing.image.img_to_array(test_image)  # Convert to array
    test_image = test_image / 255.0  # Normalize
    test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension
    class_names = ['a hot dog', 'not a hot dog']
    predictions = classifier_model.predict(test_image)
    # scores = tf.nn.sigmoid(predictions)  # predictions tf.nn.sigmoid(predictions) 
    scores = predictions #scores.numpy()
    # image_class = [class_names[i] for i in np.argmax(scores, axis=1)] # we try this 
    binary_predictions = (scores >= 0.5).astype(int)
    inverted_predictions_flat = 1 - binary_predictions.ravel().astype(int)

    # Determine the class based on the inverted binary prediction
    image_class = "hotdog" if inverted_predictions_flat == 0 else "not a hotdog"

    result = "The image uploaded is {}".format(image_class)
    return result

if __name__ == "__main__":
    main()

# def main():
#     st.header("Image Class Predictor")

#     # File uploader
#     file_uploaded = st.file_uploader("Choose file", type=['.jpg'])

#     if file_uploaded is not None:
#         # Load and display the image
#         image = Image.open(file_uploaded)
#         st.image(image, caption='Uploaded Image', use_column_width=True)

#         # Make predictions
#         result = predict_class(image)

#         # Display the result
#         st.write(result)

# def predict_classes_in_folder(image_folder_path):
#     # Load the saved model
#     classifier_model = tf.keras.models.load_model('../code_ben/my_model.h5')

#     # List all files in the folder
#     image_files = [f for f in os.listdir(image_folder_path) if os.path.isfile(os.path.join(image_folder_path, f))]

#     # Make predictions for each image
#     predictions_results = []
#     class_names = ['a hot dog', 'not a hot dog']

#     for image_file in image_files:
#         # Load and preprocess the image
#         image_path = os.path.join(image_folder_path, image_file)
#         test_image = Image.open(image_path).resize((128, 128))
#         test_image_array = image.img_to_array(test_image)
#         test_image_array = preprocess_input(test_image_array)  # Use the appropriate preprocess_input function
#         test_image_array = np.expand_dims(test_image_array, axis=0)  # Add batch dimension

#         # Make predictions
#         predictions = classifier_model.predict(test_image_array)

#         # Use sigmoid activation function on predictions
#         scores = tf.nn.sigmoid(predictions)
#         scores = scores.numpy()

#         # Choose the class with the highest probability
#         image_class = class_names[np.argmax(scores)]
#         predictions_results.append({'image_file': image_file, 'predicted_class': image_class})

#     return predictions_results

# if __name__ == "__main__":
#     # Replace 'path/to/your/image_folder' with the actual path to your image folder
#     image_folder_path = '../hotdog-nothotdog'
#     predictions_results = predict_classes_in_folder(image_folder_path)

#     # Print the results
#     for result in predictions_results:
#         print("Image {}: {}".format(result['image_file'], result['predicted_class']))
                 