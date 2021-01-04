import gradio as gr
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from transfer import load_model
from preprocessing import preprocess_image
# required library
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from local_utils import detect_lp
from os.path import splitext,basename
from keras.models import model_from_json
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder
import glob
from segementing_Letter_Using_CV2 import DiffImage,sort_contours,get_Crop_Letter
from get_plate import get_plate


def demo(img):
	
	wpod_net_path = "wpod-net.json"
	wpod_net = load_model(wpod_net_path)
	vehicle, LpImg,cor = get_plate(wpod_net,img)
	plate_image,gray,blur,binary,thre_mor=DiffImage(LpImg)
	fig = plt.figure(figsize=(12,7))
	plt.rcParams.update({"font.size":18})
	grid = gridspec.GridSpec(ncols=2,nrows=3,figure = fig)
	plot_image = [plate_image, gray, blur, binary,thre_mor]
	plot_name = ["plate_image","gray","blur","binary","dilation"]
	return binary


iface = gr.Interface(demo, gr.inputs.Image(), "image")
iface.launch()


