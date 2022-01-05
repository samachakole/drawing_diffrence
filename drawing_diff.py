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


def image_reader(img1,img2):

    img1 = cv2.imread(img1)
    img2 = cv2.imread(img2)

    if img1.shape != (900, 999, 3):
        img1 = resize(img1, (900, 999, 3))
    if img2.shape != (900, 999, 3):
        img2 = resize(img2, (900, 999, 3))

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
    cv2.imshow("Actual1",img1)
    cv2.imshow("Actual2",img2)
    cv2.waitKey(0)

image1 = st.file_uploader('Please upload old image')

image2 = st.file_uploader('Please upload new image')

if image2 is not None and st.button("Get Comparision"):
        input_image1 = Image.open(image1)
        input_image2 = Image.open(image2)
        image_reader(input_image1,input_image2)
