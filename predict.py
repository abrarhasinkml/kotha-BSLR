# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 10:58:57 2021

@author: ahasi
"""

import numpy as np
import cv2
from tensorflow import keras
import streamlit as st
from PIL import Image 
mapping_dict={
    1:'অ',
    2:'আ',
    3:'ই',
    4:'উ',
    5:'ঋ',
    6:'এ',
    7:'ঐ',
    8:'ও',
    9:'ঔ',
    }
model=keras.models.load_model('model70')
st.title('Bengali Sign Language Recognition-kotha')
st.write('BSLR-kotha was one of my very first deep learning projects during my bachelors. The goal of this project was to create a system to tackle the communication gap between the general mass and the deaf and mute.')
st.write('The Bengali script consists of a total of eleven sorobornos, where two vowel dipthongs and seven major vowel sounds are represented by them. The Consonant letters are referred to as the baenjonborno the name of which are usually the consonant sounds and the inherent vowels')
st.write('The image below has been collected from the Bengali Sign Language dictionary and represents the eleven vowels in the Bengali scriptwhich are represented by nine signs')

reference_image=Image.open('BanglaManualAlphabet.png')
st.image(reference_image, caption='Sign Languages of the Vowels of the Bengali Language acquired from the Bengali Sign Language dictionary')

test_image=st.file_uploader("Please upload an image showing a sign language: ", type=['png','jpeg', 'jpg'])

@st.cache
def load_image(image):
    img=Image.open(image)
    return img

def process_image(image):
    img = np.asarray(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (200, 200))
    #img = np.asarray(img)
    img = img.reshape(1,200,200,1)
    img = img.astype('float32')
    predict = np.argmax(model.predict(img))
    st.header('The sign your image shows is:')
    st.write(mapping_dict[predict])
if test_image is not None:
    image=load_image(test_image)
    process_image(image)
else:
    st.write('Please reupload the file')


st.write('The CNN Network used in this project is shown in the image below')
model_image=Image.open('model.png')
st.image(model_image, caption='The Convolutional Neural Network layer for Kotha-BSLR')

st.subheader('The image below shows a sample output of the model')
sample_output=Image.open('SampleImage.jpg')
st.image(sample_output, caption='A sample image from the dataset and its output')