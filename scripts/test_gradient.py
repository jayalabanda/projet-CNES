import numpy as np
import cv2
import matplotlib.pyplot as plt

fire = cv2.imread('../output/fire_var.png')

grad = cv2.Laplacian(fire, cv2.CV_64F)
grad = np.uint8(np.absolute(grad))
sobelx = cv2.Sobel(fire, 0, dx=1, dy=0)
sobelx = np.uint8(np.absolute(sobelx))
sobely = cv2.Sobel(fire, 0, dx=0, dy=1)
sobely = np.uint8(np.absolute(sobely))

results = [grad, sobelx, sobely, sobelx + sobely]
images = ["Gradient Image", "Gradient In X direction",
          "Gradient In Y direction", "Gradient In X + Y direction"]

plt.figure(figsize=(20, 10))
for i in range(4):
    plt.subplot(1, 4, i+1)
    plt.title(images[i])
    plt.imshow(results[i], "plasma")
    plt.xticks([])
    plt.yticks([])
plt.show()
