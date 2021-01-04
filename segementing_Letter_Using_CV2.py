import cv2
import glob
import matplotlib.pyplot as plt
from os.path import splitext,basename
from local_utils import detect_lp
from transfer import load_model
from preprocessing import preprocess_image
from get_plate import get_plate


# 1. see what it looks like in different types: plate_image, gray, blur, binary,thre_mor
def DiffImage(LpImg):
	if (len(LpImg)): #check if there is at least one license image
    # Scales, calculates absolute values, and converts the result to 8-bit.
	    plate_image = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
	    
	    # convert to grayscale and blur the image
	    gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
	    blur = cv2.GaussianBlur(gray,(7,7),0)
	    
	    # Applied inversed thresh_binary 
	    binary = cv2.threshold(blur, 180, 255,
	                         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
	    
	    kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
	    thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
	
	return plate_image,gray,blur,binary,thre_mor
	
## test DiffImage


### image test
test_image_path = "Plate_examples/germany_car_plate.jpg"
vehicle, LpImg,cor = get_plate(test_image_path)

### load model
wpod_net_path = "wpod-net.json"
wpod_net = load_model(wpod_net_path)

plate_image,gray,blur,binary,dilation=DiffImage(LpImg)

fig = plt.figure(figsize=(12,7))
plt.rcParams.update({"font.size":18})
grid = gridspec.GridSpec(ncols=2,nrows=3,figure = fig)
plot_image = [plate_image, gray, blur, binary,thre_mor]
plot_name = ["plate_image","gray","blur","binary","dilation"]

for i in range(len(plot_image)):
    fig.add_subplot(grid[i])
    plt.axis(False)
    plt.title(plot_name[i])
    if i ==0:
        plt.imshow(plot_image[i])
    else:
        plt.imshow(plot_image[i],cmap="gray")
plt.show()