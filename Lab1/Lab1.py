#task 2
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

img = cv2.imread('images/cameraman.tif')
plt.imshow(img)
plt.show()

img2 = Image.open('images/lena_gray_256.tif')
plt.imshow(img2, cmap=cm.Greys_r)
plt.show()

#task 3
cv2.imwrite('new_image.jpg', img)
img2.save('new_image2.jpg')

#task 4
import numpy as np
print(img.shape)
print(img)
img_array = np.array(img2)
print(img_array.shape)
print(img_array)
