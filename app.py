import streamlit as st
from fastai.vision.all import *

import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt


# load the model
learn = load_learner("export.pkl")
categories = learn.dls.vocab

def load_image(image_file):
    img = PILImage.create(image_file)
    img.thumbnail((192,192))
    return img


# clsasify an image and return all probabilities
def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs))) # streamlit might not be able to recognise tensors

# load an image beforehand
img = load_image("download.jpg")


# creating the streamlit app
st.header("Welcome to this Shrek character classifier")
st.write(f"It can classify the following Shrek characters: \nShrek\nLord Farquaad\nDonkey\nPuss in Boots\nDragon")
st.image(img)
st.write("This image has the following predictions:")
st.write(classify_image(img))
st.write("Now try it yourself:")

image_file = st.file_uploader("Upload Images", type=["png","jpg","jpeg"])

if image_file is not None:

    # To See details
    file_details = {"filename":image_file.name, "filetype":image_file.type,
                    "filesize":image_file.size}
    uploaded_image = load_image(image_file)
    st.image(uploaded_image)    
    
    preds = classify_image(uploaded_image)  
    cats = list(preds.keys())
    values = list(preds.values())
    
    st.write(preds)
    st.bar_chart(pd.DataFrame.from_dict(preds, orient="index"))

