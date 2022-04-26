import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tf_explain.callbacks.grad_cam import GradCAM
import os

OR_PATH = os.getcwd()
TEST_DIR = os.getcwd() + os.path.sep + 'test_images' + os.path.sep

photos = []

for file in os.listdir(TEST_DIR):
    filepath = os.path.join(TEST_DIR, file)
    photos.append(file)


st.title('Trading Card Authenticator')
st.header('DATS 6203 Final Project Demo')
st.write('by: Ashwin Dixit, Jose Garcia -- Group 5')
st.write('--------------')
st.write('The goal of this project is to identify genuine Pokémon trading cards.')

st.write('In this demo, users should upload an JPG image or select one from menu:')

#IMG_SIZE = 256, cv2 read the image
image_upload = st.file_uploader("Choos a image file", type='jpg')

option = st.multiselect("...or select an image:", photos)

if image_upload is not None:
    array_img = np.asarray(bytearray(image_upload.read()))
    cv2_img = cv2.imdecode(array_img,1)
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    #rezised = cv2.resize(cv2_img, (256,256))
    pil_img = tf.keras.preprocessing.image.array_to_img(cv2_img)
    tensor_img = tf.image.resize(pil_img, size=(256,256))

    st.image(cv2_img, channels="RGB")

if len(option) > 0:
    tensor_img = tf.keras.utils.load_img(TEST_DIR+'/'+option[0], target_size = (256,256))
    st.image(tensor_img, channels="RGB")

    

#Load the model and ready to use
labels = {0:'Fake', 1:'Real', 2:'Not valid input'}
model = tf.keras.models.load_model(OR_PATH+'/Code/model_Pokemon.h5')
generate_pred = st.button("Generate Prediction")
if generate_pred:
    prediction = model.predict(tf.expand_dims(tensor_img,0)).argmax()
    st.header("The model predicted this card is {}".format(labels[prediction]))

    st.subheader("Grad-Cam visual explanation")

    explainer = GradCAM()
    img = tf.keras.preprocessing.image.img_to_array(tensor_img)
    data = ([img], None)
    grid = explainer.explain(data, model, class_index=prediction)
    st.image(grid)

    




