import numpy as np
import cv2
import glob

img = cv2.imread('/home/sumitsaha/Desktop/test-7.jpg',0) 
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
cl1 = clahe.apply(img) 
cv2.imwrite('histo-result-test-7.jpg',cl1)
	
