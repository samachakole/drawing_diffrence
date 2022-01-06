from PIL import Image, ImageChops
import matplotlib.pyplot as plt
import cv2
from skimage.color import rgb2gray, rgba2rgb
from skimage.metrics import structural_similarity as sim_comp
import imutils
import sys
import argparse
from skimage.transform import resize
import streamlit as st
import numpy as np
import numpy as asarray
from pdf2image import convert_from_path, convert_from_bytes

def image_reader(img1,img2):
    img1 = img1
    img2 = img2
    if img1.shape != (800, 999, 3):
        img1 = resize(img1, (800, 999, 3))
    if img2.shape != (800, 999, 3):
        img2 = resize(img2, (800, 999, 3))
    img1 = rgb2gray(img1)
    img2 = rgb2gray(img2)
    rate, diffrences = sim_comp(img1,img2, full=True)
    diffrences = (diffrences*225).astype("uint8")
    thresh = cv2.threshold(diffrences,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    counts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    counts = imutils.grab_contours(counts)
    for count in counts:
        (x,y,w,h) = cv2.boundingRect(count)
        img1=cv2.rectangle(img1,(x,y),(x+w,y+h),(0,0,255),2)
        img2=cv2.rectangle(img2,(x,y),(x+w,y+h),(0,0,255),2)
    return img1,img2

def convert_uploaded_cv2(file):
    file_image1 = np.asarray(bytearray(file.read()), dtype=np.uint8)
    converted_image = cv2.imdecode(file_image1, 1)
    return converted_image

image1 = st.file_uploader('Please upload old image',type=['png','jpg','pdf'])
image2 = st.file_uploader('Please upload new image',type=['png','jpg','pdf'])

#image1 = convert_from_path(image1)
#image2 = convert_from_path(image2)

if image2 is not None and st.button("Get Comparision"):
    if image1 is not None:
        image1 = convert_uploaded_cv2(image1)
    if image2 is not None:
        image2 = convert_uploaded_cv2(image2)
        
    img1,img2 = image_reader(image1,image2)
    cv2.imshow("Actual1",img1)
    cv2.imshow("Actual2",img2)
    cv2.waitKey(0)
