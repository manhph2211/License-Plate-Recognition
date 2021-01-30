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
from local_utils import detect_lp, getPath
from os.path import splitext,basename
from keras.models import model_from_json
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder
import glob
from segementing_Letter_Using_CV2 import DiffImage,sort_contours,get_Crop_Letter
from get_plate import get_plate
import json



def getResults(path_result='./results_0_15.json'):
	with open(path_result,'r') as f:
		results=json.load(f)
		dic={}
		for k,v in results.items():
			for x in v:
				true_val=x[0][0]
				path=x[0][1]
				predict_value=x[1]
				dic[path]=predict_value
		return dic

# local host
def demo(img):
	global dic_

	return "Result: " + dic_[img]



dic_=getResults()
values=list(getPath().values())
paths=[el[1] for el in values[0]]

iface = gr.Interface(demo, 
   [ gr.inputs.Radio(paths)],
   ['text','image']
)

iface.launch(share=False)




# test_folder= './TDCN_IMG'
# test_image_paths= [os.path.join(test_folder,x) for x in os.listdir(test_folder)]
