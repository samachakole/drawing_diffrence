from PIL import Image, ImageChops
import matplotlib.pyplot as plt
import cv2
from skimage.color import rgb2gray, rgba2rgb
from skimage.metrics import structural_similarity as sim_comp
import imutils
import sys
from skimage.transform import resize
import streamlit as st
import numpy as np
import numpy as asarray

def image_reader(img1,img2):
    img1 = img1
    img2 = img2
    if img1.shape != (800, 999, 3):
        img1 = resize(img1, (800, 999, 3))
    if img2.shape != (800, 999, 3):
        img2 = resize(img2, (800, 999, 3))
    img1 = rgb2gray(img1)
    img2 = rgb2gray(img2)
    diff_val, diff = sim_comp(img1,img2, full=True)
    diff = (diff*225).astype("uint8")
    limit = cv2.threshold(diff,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    No_of_time = cv2.findContours(limit.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    No_of_time = imutils.grab_contours(No_of_time)
    for Num in No_of_time:
        (x,y,w,h) = cv2.boundingRect(Num)
        img1=cv2.rectangle(img1,(x,y),(x+w,y+h),(0,0,255),2)
        img2=cv2.rectangle(img2,(x,y),(x+w,y+h),(0,0,255),2)
    return img1,img2

def convert_uploaded_cv2(file):
    file_image1 = np.asarray(bytearray(file.read()), dtype=np.uint8)
    converted_image = cv2.imdecode(file_image1, 1)
    return converted_image

image1 = st.file_uploader('Please upload old image',type=['png','jpg','pdf'])
image2 = st.file_uploader('Please upload new image',type=['png','jpg','pdf'])

if image2 is not None and st.button("Get Comparision"):
    if image1 is not None:
        image1 = convert_uploaded_cv2(image1)
    if image2 is not None:
        image2 = convert_uploaded_cv2(image2)

    img1,img2 = image_reader(image1,image2)
    st.image(img1, caption="Actual1")
    st.image(img2, caption="Actual2")
    cv2.waitKey(0)
