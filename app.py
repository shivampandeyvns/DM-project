import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import keras 

def names(number):
    if number==1:
        return 'a Tumor'
    else:
        return 'not a tumor'

def classifier():
    uploaded_file = st.file_uploader("Choose an image...", type="jpg")
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("")
        st.write("Classifying...")
        model=keras.models.load_model('CNN_Brain_tumor1.h5')
        
        x = np.array(image.resize((128,128)))
        x = x.reshape(1,128,128,3)
        res = model.predict_on_batch(x)
        classification = np.where(res == np.amax(res))[1][0]
        st.subheader(str(res[0][classification]*100) + '% Confidence This Is ' + names(classification))


st.title('Brain Tumor Classifier')

image=Image.open('download.jfif')
st.image(image)

menu=['Home','Classifier','Info']
choice=st.sidebar.selectbox('Menu',menu)

check=False

if(choice=='Home'):
    st.markdown("""
    
    """)

elif(choice=='Classifier'):
    classifier()


elif(choice=='Info'):
    st.markdown("""

    """)
