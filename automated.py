import numpy as np
import cv2
import glob

def get_files():
	files = []
	for file in glob.glob('*.jpg'):
		files.append(file)
	return files	
	
def clahe_histo(file_name):
	file_name = file_name.strip('.jpg')
	img = cv2.imread('{}.jpg'.format(file_name),0) 
	clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
	cl1 = clahe.apply(img)
	cv2.imwrite('histo-result-{}.png'.format(file_name),cl1)

def haar_cascade(file_name):
	file_name = file_name.strip('.jpg')
	print(file_name)
	face_cascade = cv2.CascadeClassifier('/home/sumitsaha/opencv-3.3.1/data/haarcascades/haarcascade_frontalface_default.xml')
	eye_cascade = cv2.CascadeClassifier('/home/sumitsaha/opencv-3.3.1/data/haarcascades/haarcascade_eye.xml')
	img = cv2.imread('{}.jpg'.format(file_name))
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
	    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
	    roi_gray = gray[y:y+h, x:x+w]
	    roi_color = img[y:y+h, x:x+w]
	    eyes = eye_cascade.detectMultiScale(roi_gray)
	    for (ex,ey,ew,eh) in eyes:
	        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
	cv2.imwrite('haar-cascade-{}.png'.format(file_name),img)
	#cv2.waitKey(0)
	#cv2.destroyAllWindows()

def canny_edge(file_name):
	file_name = file_name.strip('.jpg')
	img = cv2.imread('{}.jpg'.format(file_name),cv2.IMREAD_GRAYSCALE)
 
	filter = cv2.Canny(img,100,200)
 
	#cv2.imshow('Original - Canny Edge',img)
	cv2.imwrite('canny-edge-{}.png'.format(file_name),filter)
 
if __name__=='__main__':
	files = get_files()
	print(files)
	for file in files:
		clahe_histo(file)
	for file in files:
		canny_edge(file)
	for file in files:
		haar_cascade(file)
