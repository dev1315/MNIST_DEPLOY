# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 17:30:56 2020

@author: DevyansHu
"""
# https://docs.streamlit.io/en/stable/api.html#display-text DOCUMENTATION
# https://www.w3schools.com/colors/colors_names.asp FOR COLORS

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import cv2
from PIL import Image



st.title("Digit Recognizer")
st.markdown('<p style="color:#708090;font-size:100%" >This is the first project that sparked the curiosity of understanding Machine Learning in me </p>',
            unsafe_allow_html=True)

# @st.cache(persist=True)
def model_loading():
    return tf.keras.models.load_model("mnist_model.h5")

def draww():
    drawing=False
    ex=-1
    ey=-1

    # Function
    # x,y, flags, param are feed from OpenCV automaticaly
    def draw_rect(event,x,y,flags,param):

        global ex,ey,drawing

        if event==cv2.EVENT_LBUTTONDOWN: #drawing 
            start=False
            drawing=True
            ex,ey=x,y
    #         cv2.circle(img,
    #                    center=(ex,ey),
    #                    radius=5,
    #                    color=(0,255,255),
    #                    thickness=-1)

        elif event==cv2.EVENT_RBUTTONDOWN: #eraser
            drawing=False
            cv2.circle(img,
                       center=(x,y),
                       radius=50,
                       color=(0,0,0),
                       thickness=-1)

        elif event==cv2.EVENT_MOUSEMOVE:
            if drawing==True:
                cv2.circle(img,
                          center=(x,y),
                           radius=10,
                           color=(255,0,0),
                          thickness=-1)

        elif event==cv2.EVENT_LBUTTONUP:
            drawing=False
    #         cv2.rectangle(img,
    #                       (ex,ey),
    #                      (x,y),
    #                      (255,0,255),
    #                      -1)


    # Connect the Function with the Callback
    cv2.namedWindow(winname="my drawing")


    # Callback
    cv2.setMouseCallback("my drawing",draw_rect)

    # Using OpenCV to show the Image
    img=np.zeros([400,400,1],np.uint8) #black bg
    while True:
        cv2.imshow("my drawing",img)
        if cv2.waitKey(5) & 0xFF==27: #if you press excape then close
            plt.imshow(img)
            break
    cv2.destroyAllWindows()
    return img

model=model_loading()

radio = st.sidebar.radio("Do You want to draw your digit or upload a image file(.jpeg)",("Draw","Upload"))
if radio=="Draw":
    if st.button("Drawing Panel"):
        image=draww()
        st.write("<p style='color:red;font-size:150%' >You drew this!</p>",
                 unsafe_allow_html=True)
        st.image(image, width =300)
        
        image=cv2.resize(image,(28,28)).reshape(1, 28, 28, 1)
        st.write(f"<p style='color:red;font-size:150%' > And i think its a {model.predict(image).argmax()}</p>",
                 unsafe_allow_html=True)
if radio=="Upload":
    # st.set_option('deprecation.showfileUploaderEncoding', False)
    image=st.file_uploader("Browse your image",type=['png','jpg'])
    image = np.asarray(Image.open(image))
    st.image(image, width =300)
    shape=int(image.shape[0])
    image = np.average(image,axis=2).reshape(shape,-1,1)
    # st.write(image.shape)
    imageu=cv2.resize(image,(28,28)).reshape(1, 28, 28, 1)
    st.write(f"<p style='color:red;font-size:150%' > And i think its a {model.predict(imageu).argmax()}</p>",
              unsafe_allow_html=True)
    
    # st.write(type(image))


# streamlit run mnist_deploy.py