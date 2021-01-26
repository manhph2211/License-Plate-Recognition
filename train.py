import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, utils, datasets,models
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from tqdm import tqdm
#from model import AlexNet, VGG, ResNet,BasicBlock


IMAGE_SHAPE = 28
# 1. Dataset
DATA_PATH = './data_after_splitting/'

# 2. Training & Evaluation
# 2.1 Define models


image_transforms = {
    "train": transforms.Compose([
        transforms.Resize((IMAGE_SHAPE, IMAGE_SHAPE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ]),
    "val": transforms.Compose([
        transforms.Resize((IMAGE_SHAPE, IMAGE_SHAPE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])
}

train_dataset = datasets.ImageFolder(root = DATA_PATH + "train",
                                   transform = image_transforms["train"]
                                  )

val_dataset = datasets.ImageFolder(root = DATA_PATH + "val",
                                   transform = image_transforms["val"]
                                  )

#print(train_dataset.class_to_idx)
#idx2class = {v: k for k, v in train_dataset.class_to_idx.items()}

train_dataloader = DataLoader(train_dataset, batch_size=20, num_workers=2, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=20, num_workers=2, shuffle=True)

# print(len(train_dataloader))

# single_batch = next(iter(train_dataloader))
# esingle_batch_grid = utils.make_grid(single_batch[0], nrow=4)
# plt.figure(figsize = (10,10))
# plt.imshow(single_batch_grid.permute(1, 2, 0))
# plt.show()

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)        
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)    
    acc = torch.round(acc) * 100    
    return acc


device = 'cuda' if torch.cuda.is_available() else 'cpu'
#model_ = ResNet(BasicBlock, [3, 4, 6, 3],num_classes=36)
model = models.resnet18(True)
model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# for train, val in train_dataloader:
#     print(train, val)

print("Begin training.")
train_loss=[]
val_loss=[]
for epoch in tqdm(range(1, 10)):    
    # TRAINING   

    running_loss = 0.0
    val_running_loss = 0.0
    model.train()
    for i, data in enumerate(train_dataloader):
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
       
        # zero the parameter gradients
        optimizer.zero_grad()
       
        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        train_loss.append(loss)
        #torch.save(model.state_dict(),'./model.pt')
    #val
    print('Epoch [%d] loss: %.3f' %(epoch + 1, running_loss))
    # model.eval()

    # for i, val_data in enumerate(val_dataloader):
    #     val_inputs, val_labels = val_data
    #     val_inputs = val_inputs.to(device)
    #     val_labels = val_labels.to(device)

    #     # zero the parameter gradients
    #     optimizer.zero_grad()

    #     # forward + backward + optimize
    #     val_outputs = model(val_inputs)
    #     loss = criterion(val_outputs, val_labels)
    #     loss.backward()
    #     optimizer.step()

    #     val_running_loss += loss.item()
    #     val_loss.append(loss)

#print(train_loss)
plt.plot(range(len(train_loss)),train_loss)
plt.plot(range(len(val_loss)),val_loss)
plt.show()
