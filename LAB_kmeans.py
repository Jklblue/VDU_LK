import cv2
import numpy as np
from sklearn.cluster import KMeans
image = cv2.imread('mask_guide_images.png')
resized_image = cv2.resize(image, (1024, 768))
lab_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2LAB)
ab_channels = lab_image[:, :, 1:3]
ab_channels_reshaped = ab_channels.reshape(-1, 2)
kmeans = KMeans(n_clusters=3, random_state=0)
labels = kmeans.fit_predict(ab_channels_reshaped)
clustered_image = labels.reshape(ab_channels.shape[0], ab_channels.shape[1])
mask = np.zeros((clustered_image.shape[0], clustered_image.shape[1], 3), dtype=np.uint8)
mask[clustered_image == 0] = [30, 144, 255]
mask[clustered_image == 1] = [255, 255, 204]
mask[clustered_image == 1] = [255, 0, 0]
cv2.imshow('Mask', mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Only one example is provided here. In practical applications, the pixels guided by the VD - UNet
# mask do not participate in the clustering. For automatic segmentation, the clustering centers of
# the target objects obtained from the training images are used. According to the Euclidean distance,
# finding the closest one can achieve the automatic segmentation of the corresponding target.
