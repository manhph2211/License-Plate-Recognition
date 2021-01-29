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



def predict_from_model(image,model,labels):
  image = cv2.resize(image,(80,80))
  image = np.stack((image,)*3, axis=-1)
  prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
  return prediction


def getResults(path_result='./results_0_15.json'):
	with open(path,'r') as f:
		results=json.load(f)
		dic={}
		for k,v in results.items():
			true_val=v[0][0]
			path=v[0][1]
			predict_value=v[1]
			dic[path]=predict_value
		return dic

dic=getResults()
keys=list(getPath().keys())
paths=[]
for el in keys[0]:
	for x in el:
		paths.append(x[1])



def demo(img):
	return "Result: " + dic[img]



test_folder= './TDCN_IMG'
test_image_paths= [os.path.join(test_folder,x) for x in os.listdir(test_folder)]

iface = gr.Interface(demo, 
   [ gr.inputs.gradio(paths)],
   ['text']
)

iface.launch(share=False)



# test_image_path = "./TDCN_IMG/0-15/E5A40A0C-6AC1-4517-AFD1-FE5DFA6C2288.jpeg"
# img=cv2.imread(test_image_path)
# plt.imshow(img)
# plt.show()
# print(demo(test_image_path))
