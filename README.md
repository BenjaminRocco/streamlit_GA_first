# Hackathon 2/Project 4: Hotdog or Not Hotdog

## The Team: Group 2

Ben Moss: Initial Model Setup, Repo Management, Transfer Learning

Steve Goulden: GridSearch, Streamlit

Dom Martorano: README, Colab Management

## Running Requirements

### Libraries

*   numpy
*   pandas
*   matplotlib.pyplot
*   glob
*   random
*   PIL (Python Imaging Library)
*   tensorflow.keras
*   sklearn.model_selection
*   scikeras.wrappers
*   streamlit
*   tensorflow_hub
*   os

## Modeling Summary

Image data was transferred from the Hotdog-Not-Hotdog dataset on Kaggle. Image data was then "globbed" into training and testing data sets. A sequential convolutional neural network model was instantiated using both Average and Max Pooling, as well as Dropout layers between each set of pooling layer and convolutional layer, as well as preceding and succeeding the dense layer. Each convolutional layer in the model doubled the number of nodes from the previous layer, beginning with 16 nodes and ending with 128 nodes for a total of four sets of layers. After creating the CNN model and fitting it to the image data, the model was saved as a .keras file in order to be transferred into a web-deployed Streamlit app.

## Streamlit App

Streamlit app run from code within `.py` script. Link here: 

`https://hawtdawgornawtdawg.streamlit.app`
