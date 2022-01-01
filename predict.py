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

model=keras.models.load_model('model70')

test_image=st.file_uploader("Please upload an image showing a sign language: ", type=['png','jpeg', 'jpg'])


def process_image(image):
    img=cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    img=cv2.resize(img, (200, 200))
    img=np.asarray(img)
    img=img.reshape(1,200,200,1)
    img=img.astype('float32')
    predict=np.argmax(model.predict(img))
    st.write(predict)
#test_image='./Dataset/3_10p3_20170319_135524 32.jpg'
if test_image is not None:
    process_image(test_image.name)
else:
    st.write('Please reupload the file')
