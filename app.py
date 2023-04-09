import streamlit as st
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from utils import preprocess_data,prediction


model = tf.keras.models.load_model("keras_model.h5")
labels = ['Healthy', 'Bleached']

df = pd.DataFrame(['Healthy','Bleached'],columns=['Coral Health Status'])

st.set_page_config(page_icon="🌊")
st.image('Banner.jpg')

def main():
    st.sidebar.markdown("<h2>About the app</h2>", unsafe_allow_html=True)
    st.sidebar.write("""
            Corals play a vital role in maintaining a healthy aquatic ecosystem, 
            but their bleaching has become a major concern as it disrupts the balance
            of the ecosystem. Early detection of bleached corals is crucial in 
            preventing potential disasters to aquatic life.

            The main objective of this web application is to classify either a coral image is
            bleached or unbleached through the use of machine learning algorithms.
            This classification can help identify bleached corals at an early stage and allow for 
            timely intervention, ultimately contributing to the preservation and protection 
            of the aquatic ecosystem. 
            """)
    st.sidebar.write(df) 
    
    file = st.file_uploader('Upload your image',type=["jpg", "png"])
    st.set_option('deprecation.showfileUploaderEncoding', False)
    
    if file is None:
        st.text("Please upload image")
    else:
        image = Image.open(file)
        st.image(image, use_column_width=True)
        image = image.save("img.jpg")
        
        img = preprocess_data("img.jpg")
        image_class, score = prediction(model,img,labels)
    
        st.write("The current health staus of the reef is: ",image_class)
        st.write("The similarity score is approximately",round(score,2))

        if image_class == 'Bleached':
            st.write(":red[URGENT ALERT: The coral reef is undergoing bleaching. Please contact your local marine institute immediately for assistance.]")
        #print("The image is classified as ",image_class, "with a similarity score of",score)
           

if __name__ == '__main__':
    main()