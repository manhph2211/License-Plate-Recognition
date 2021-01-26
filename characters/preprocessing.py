import torch
import numpy as np
from sklearn.preprocessing import scale,StandardScaler
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt 




IMAGE_SHAPE = 28

# 1. Dataset
DATA_PATH = './data'
character_image_paths = {}
for _folder in os.listdir(DATA_PATH): 
    current_folder = os.path.join(DATA_PATH, _folder) 
    for image_name in os.listdir(current_folder):
        if _folder not in character_image_paths:
            character_image_paths[_folder] = [os.path.join(current_folder, image_name)]
        else:
            character_image_paths[_folder].append(os.path.join(current_folder, image_name))



X = []
y = []
for i, (k, v) in enumerate(human_image_paths.items()):
 
    for image_path in v:
        try:
            image = cv2.imread(image_path)
            image = cv2.resize(image, (IMAGE_SHAPE, IMAGE_SHAPE))
            X.append(image)
            y.append(i)
        except:
            print('Ignore image:', image_path)

X = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y, dtype=torch.long)

X = torch.reshape(X, [X.shape[0], -1])

# scaler = StandardScaler()
# X= scaler.fit_transform(X.numpy())
# X=torch.from_numpy(X)
#print(X.shape)
torch.save(X, 'X.pt')
torch.save(y, 'y.pt')

