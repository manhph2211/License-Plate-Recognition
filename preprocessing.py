import cv2
import glob
import matplotlib.pyplot as plt
from os.path import splitext,basename
from local_utils import detect_lp
from transfer import load_model

# 1. 
def preprocess_image(image_path,resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img


# test some images in folder Plate_examples

'''
# Create a list of image paths using glob
image_paths = glob.glob("Plate_examples/*.jpg")
print("Found %i images..."%(len(image_paths)))

fig = plt.figure(figsize=(12,8))

cols = 5
rows = 4
fig_list = []
for i in range(cols*rows):

	if i+1==len(image_paths):
		break
	else:
	    fig_list.append(fig.add_subplot(rows,cols,i+1))
	    title = splitext(basename(image_paths[i]))[0]
	    fig_list[-1].set_title(title)
	    img = preprocess_image(image_paths[i],True)
	    plt.axis(False)
	    plt.imshow(img)

plt.tight_layout(True)
plt.show()
'''


# 2. 

image_paths = glob.glob("Plate_examples/*.jpg")
wpod_net_path = "wpod-net.json"
wpod_net = load_model(wpod_net_path)


def get_plate(image_path, Dmax=608, Dmin=256):
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return LpImg, cor

# Obtain plate image and its coordinates from an image
test_image = image_paths[2]
LpImg,cor = get_plate(test_image)
print("Detect %i plate(s) in"%len(LpImg),splitext(basename(test_image))[0])
print("Coordinate of plate(s) in image: \n", cor)

# Visualize our result
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.axis(False)
plt.imshow(preprocess_image(test_image))
plt.subplot(1,2,2)
plt.axis(False)
plt.imshow(LpImg[0]) 
plt.show()