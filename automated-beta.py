import numpy as np
import cv2
import glob
import face_recognition
import os

class face_recog:
	
	faces = []
	dataset = []
	opencv_version = str(cv2.__version__)
	cascade_classifier_path = os.system('locate opencv-{}/data/haarcascades | head -n 1'.format(opencv_version))
	
	def __init__(self):
		self.get_files()
		for file in self.faces:
			print(file)

	def get_files(self):
		for file in glob.glob('*.jpg'):
			self.faces.append(file)	
	
	def clahe_histo(self,file_name):
		file_name = file_name.strip('.jpg')
		img = cv2.imread('{}.jpg'.format(file_name),0) 
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
		cl1 = clahe.apply(img)
		cv2.imwrite('histo-result-{}.png'.format(file_name),cl1)
	
	# need to have a automated placeholder module to scout haarcascade files
	def haar_cascade(self,file_name):
		file_name = file_name.strip('.jpg')
		print(file_name)
		face_cascade = cv2.CascadeClassifier('{}/haarcascade_frontalface_default.xml'.format(cascade_classifier_path))
		eye_cascade = cv2.CascadeClassifier('{}/haarcascade_eye.xml'.format(cascade_classifier_path))
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
	
	def canny_edge(self,file_name):
		file_name = file_name.strip('.jpg')
		img = cv2.imread('{}.jpg'.format(file_name),cv2.IMREAD_GRAYSCALE)
	 
		filter = cv2.Canny(img,100,200)
	 
		#cv2.imshow('Original - Canny Edge',img)
		cv2.imwrite('canny-edge-{}.png'.format(file_name),filter)
		
	def recog_image(self,file1,file2):
		known_image = face_recognition.load_image_file("{}".format(file1))
		unknown_image = face_recognition.load_image_file("{}".format(file2))      
		       
		biden_encoding = face_recognition.face_encodings(known_image)[0]	
		unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
		
		results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
		return(results)
	
	def enchance_images(self):
		for file in self.faces:
			self.clahe_histo(file)
		for file in self.faces:
			self.canny_edge(file)
		for file in self.faces:
			self.haar_cascade(file) 
		
	def compare_faces(self):	
		for file2 in self.faces:
			ctr=1
			flag=False
			diffctr=0
			for file1 in self.dataset:
				if(self.recog_image(file1,file2)==[True]):
					fname=file1.strip(".jpg")+"_"+str(ctr)					
					fname=fname[8:]
					ctr+=1
					print("{} and {} are same".format(file1,file2))
					loc=self.faces.index(file2)
					self.faces.remove(file2)
					self.faces.insert(loc,fname+".jpg")
					os.rename(file2,fname+".jpg")
					flag=True
					break
				else:
					print("{} and {} are different".format(file1,file2))
					diffctr+=1
					if diffctr==len(self.dataset):
						inp=input("\n\n{} is unidentified.\nWho are you?? ".format(file2))
						print("\n{}.jpg was added to the dataset.".format(inp))
						os.rename(file2,"dataset/"+inp+".jpg")
			if(flag):
				continue
		
	 
if __name__=='__main__':
	
	recognition = face_recog()
	
	#recognition.enchance_images()
	recognition.compare_faces()
