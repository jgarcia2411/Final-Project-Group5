import streamlit as st
import tensorflow as tf
import numpy as np
import cv2

st.title('ML 2 Final Project')

#IMG_SIZE = 256, cv2 read the image
image_upload = st.file_uploader("Choos a image file", type='jpg')

if image_upload is not None:
    array_img = np.asarray(bytearray(image_upload.read()))
    cv2_img = cv2.imdecode(array_img,1)
    cv2_img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    #rezised = cv2.resize(cv2_img, (256,256))
    pil_img = tf.keras.preprocessing.image.array_to_img(cv2_img)
    tensor_img = tf.image.resize(pil_img, size=(256,256))

    st.image(cv2_img, channels="RGB")

    

#Load the model and ready to use
labels = {0:'Fake', 1:'Real', 2:'Not valid input'}
model = tf.keras.models.load_model('model_Pokemon.h5')
generate_pred = st.button("Generate Prediction")
if generate_pred:
    prediction = model.predict(tf.expand_dims(tensor_img,0)).argmax()
    st.title("Predicted Label for the image is {}".format(labels[prediction]))






