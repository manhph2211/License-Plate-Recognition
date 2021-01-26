import numpy as np 
import cv2
import os
import os.path as osp
import random
import shutil



ROOT_DATA='./data'
_name = os.listdir(ROOT_DATA)


_paths = {}
for sc in _name:
    _paths[sc] = []

for sc in _name:
    folder_path = osp.join(ROOT_DATA, sc)
    image_names = os.listdir(folder_path)
    _paths[sc] = [osp.join(folder_path, _name) for _name in image_names]


DEST_FOL = "./data_after_splitting"
TRAIN_FOL = "./data_after_splitting/train"
VAL_FOL = "./data_after_splitting/val"

for sc, sc_paths in _paths.items():
    sc_train_fol = osp.join(TRAIN_FOL, sc)
    sc_val_fol = osp.join(VAL_FOL, sc)
    
    #print(sc)
    if not osp.exists(sc_train_fol):
        os.makedirs(sc_train_fol)
    if not osp.exists(sc_val_fol):
        os.makedirs(sc_val_fol)
    
    random.shuffle(sc_paths)
    train_idx = int(0.8*len(sc_paths))
    
    for path in sc_paths[:train_idx]:
        shutil.copy2(path, sc_train_fol)
    
    for path in sc_paths[train_idx:]:
        shutil.copy2(path, sc_val_fol)







