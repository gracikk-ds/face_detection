from model import FaceDetector
from PIL import Image
import numpy as np
import cv2


img = np.array(Image.open("./images.jpeg"))

print(img.shape)
model = FaceDetector()
result_img = model(img)

result_img = np.moveaxis(result_img, 0, -1).astype(np.uint8)

print(result_img.shape)

img = Image.fromarray(result_img)

print(img)


cv2.imwrite("./result.jpg", result_img.astype(int))
print(result_img.astype(int))
