from local_utils import getPath
import numpy as np 
import sys
import os
import json



def finalOutput(img_):
  img=cv2.imread(img_)
  #print(img.shape)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img = img/255
  wpod_net_path = "wpod-net.json"
  wpod_net = load_model(wpod_net_path)
  test_roi,crop_characters=get_Crop_Letter(wpod_net,img)
  #load pretrained
  #Load model architecture, weight and labels
  json_file = open('MobileNets_character_recognition.json', 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  model = model_from_json(loaded_model_json)
  model.load_weights("License_character_recognition_weight.h5")
  #print("[INFO] Model loaded successfully...")
  labels = LabelEncoder()
  labels.classes_ = np.load('license_character_classes.npy')
  #print("[INFO] Labels loaded successfully...")

  # pre-processing input images and pedict with model

  final_string = ''
  for character in crop_characters:
    title = np.array2string(predict_from_model(character,model,labels))
    final_string+=title.strip("'[]")
  #print(final_string)
  return final_string


  
def predict(path_dic):
	results={}
	for k,v in path_dic.items():
		results[k]=[]
		print("Angle ",k)
		for i,path in enumerate(v):
			re=finalOutput(path[1])
			print("IMAGE {0} - {1} ---------> {2}".format(i+1,path[0],re))
			results[k].append([path,re])

	return results

# results=predict(getPath())

# with open('results_0_15.json','w') as f:
# 	json.dump(results,f,indent=4)

