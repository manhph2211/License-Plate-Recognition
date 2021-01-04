import cv2
import glob
import matplotlib.pyplot as plt
from os.path import splitext,basename
from local_utils import detect_lp
from transfer import load_model
from preprocessing import preprocess_image



def get_plate(wpod_net,image, Dmax=608, Dmin=256):
    vehicle = image
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return vehicle,LpImg, cor


# test
'''
## Obtain plate image and its coordinates from an image
image_paths = glob.glob("Plate_examples/*.jpg")
test_image = image_paths[0]
wpod_net_path = "wpod-net.json"
wpod_net = load_model(wpod_net_path)
image=preprocess_image(test_image)
vehicle,LpImg,cor = get_plate(wpod_net,image)
print("Detect %i plate(s) in"%len(LpImg),splitext(basename(test_image))[0])
print("Coordinate of plate(s) in image: \n", cor)

## Visualize our result
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.axis(False)
plt.imshow(preprocess_image(test_image))
plt.subplot(1,2,2)
plt.axis(False)
plt.imshow(LpImg[0]) 
plt.show()
'''