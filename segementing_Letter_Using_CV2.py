import cv2
import glob
import matplotlib.pyplot as plt
from os.path import splitext,basename
from local_utils import detect_lp
from transfer import load_model
from preprocessing import preprocess_image
from get_plate import get_plate
import matplotlib.gridspec as gridspec


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


### load model
'''
wpod_net_path = "wpod-net.json"
wpod_net = load_model(wpod_net_path)


### image test
test_image_path = "Plate_examples/germany_car_plate.jpg"
vehicle, LpImg,cor = get_plate(wpod_net,test_image_path)

plate_image,gray,blur,binary,thre_mor=DiffImage(LpImg)

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
'''


# Create sort_contours() function to grab the contour of each digit from left to right
def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts

def get_Crop_Letter(wpod_net,image):

	vehicle, LpImg,cor = get_plate(wpod_net,image)
	plate_image,gray,blur,binary,thre_mor=DiffImage(LpImg)
	cont, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	# creat a copy version "test_roi" of plat_image to draw bounding box
	test_roi = plate_image.copy()

	# Initialize a list which will be used to append charater image
	crop_characters = []

	# define standard width and height of character
	digit_w, digit_h = 30, 60

	for c in sort_contours(cont):
	    (x, y, w, h) = cv2.boundingRect(c)
	    ratio = h/w
	    if 1<=ratio<=3.5: # Only select contour with defined ratio
	        if h/plate_image.shape[0]>=0.5: # Select contour which has the height larger than 50% of the plate
	            # Draw bounding box arroung digit number
	            cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 2)

	            # Sperate number and gibe prediction
	            curr_num = thre_mor[y:y+h,x:x+w]
	            curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
	            _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
	            crop_characters.append(curr_num)

	return test_roi,crop_characters
	
## test getCropLetters
'''
### load model
wpod_net_path = "wpod-net.json"
wpod_net = load_model(wpod_net_path)


### image test
test_image_path = "Plate_examples/germany_car_plate.jpg"
image=preprocess_image(test_image_path)
test_roi,crop_characters=get_Crop_Letter(wpod_net,image)

print("Detect {} letters...".format(len(crop_characters)))
# fig = plt.figure(figsize=(10,6))
# plt.axis(False)
# plt.imshow(test_roi)
# plt.show()


fig = plt.figure(figsize=(14,4))
grid = gridspec.GridSpec(ncols=len(crop_characters),nrows=1,figure=fig)
print(grid)
for i in range(len(crop_characters)):
    fig.add_subplot(grid[i])
    plt.axis(False)
    plt.imshow(crop_characters[i],cmap="gray")
plt.show()
#plt.savefig("segmented_leter.png",dpi=300)    

'''
