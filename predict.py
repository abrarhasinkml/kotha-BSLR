# -*- coding: utf-8 -*-
"""
Created on Fri Dec 31 10:58:57 2021

@author: ahasi
"""

import numpy as np
import cv2
from tensorflow import keras
import streamlit as st

model=keras.models.load_model('model70')

test_image=st.file_uploader("Please upload an image showing a sign language: ")
st.write(type(test_image))
#test_image='./Dataset/3_10p3_20170319_135524 32.jpg'
if test_image is not None:
    img=cv2.imread(test_image, cv2.IMREAD_GRAYSCALE)
    img=cv2.resize(img, (200, 200))
    img=np.asarray(img)
    img=img.reshape(1,200,200,1)
    img=img.astype('float32')
    predict=np.argmax(model.predict(img))
    st.write(predict)
else:
    st.write('Please reupload the file')