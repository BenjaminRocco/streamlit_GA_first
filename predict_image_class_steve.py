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
from scikeras.wrappers import KerasClassifier

import streamlit as st #
import os #
import h5py #
import tensorflow_hub as hub #

st.header("Image class predictor")

def main():
    st.file_uploader("Choose file" type = ['jpg', 'png', 'jpeg'])
    if file_uploaded is not None:
        image = Image.open(file_uploaded)
        figure = plt.figure()
        plt.imshow(image)
        plt.axis('off')
        result = predict_class(image)
        st.write(result)
        st.pyplt(figure)
                 
def predict_class(image):
    classifier_model = tf.keras.models.load_model(r'/content/drive/MyDrive/archive/my_model.hdf5')
    shape = ((128, 128, 3))
    model = tf.keras.Sequential([hub.KerasLayer(classifier_model, input_shape=shape)])
    test_image = image.resize((128, 128))
    preprocessing.image.img_to_array(test_image)
    test_image= test_image/255.0
    test_image = np.expand_dims(test_image, axis=0)
    class_names = ['a hot dog', 'not a hot dog']
    #The below might be different
    predictions = model.predict(test_image)
    tf.nn.sigmoid(predictions[0])
    scores = scores.numpy()
    image_class = class_names[np.argmax(scores)]
    result = "The image uploaded is: {}".format(image_class)
    return result

if __name__ == "__main__":
    main()

                 