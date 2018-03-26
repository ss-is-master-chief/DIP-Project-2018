'''
The following code has been created by the following:
> Ishan Bhattacharya
> Sumit Saha
> Prerna Agarwal
> Ayush Rawal
for CSE4019 : Digital Image Processing
during Winter-Semester : 2017-2018
under the supervision of Dr. Malathi G.
'''

import numpy as np
import cv2
import glob
import face_recognition
import os

class face_recog:
	
	# faces contains all the test images
	faces = [] 
	# dataset contains all the to-be-queried-against images
	dataset=[]

	opencv_version = str(cv2.__version__)
	cascade_classifier_path = os.system('locate opencv-{}/data/haarcascades | head -n 1'.format(opencv_version))
	
	#
	# call the 'get_files()' function at the creation of class object
	#	
	def __init__(self):
		self.get_files()
		for file in self.faces:
			print(file)

	# 
	# to get all the image files ending with '.jpg'
	#
	def get_files(self):
		for file in glob.glob('*.jpg'):
			self.faces.append(file)
		for file in glob.glob('dataset/*.jpg'):
			self.dataset.append(file)
				
	#
	# performs clahe histogram for image enhancement
	#
	def clahe_histo(self,file_name):
		file_name = file_name.strip('.jpg')
		img = cv2.imread('{}.jpg'.format(file_name),0) 
		clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
		cl1 = clahe.apply(img)
		cv2.imwrite('histo-result-{}.png'.format(file_name),cl1)
	
	#
	# detects edges using Canny Edge Detection method
	#
	def canny_edge(self,file_name):
		file_name = file_name.strip('.jpg')
		img = cv2.imread('{}.jpg'.format(file_name),cv2.IMREAD_GRAYSCALE)
	 
		filter = cv2.Canny(img,100,200)
	 
		#cv2.imshow('Original - Canny Edge',img)
		cv2.imwrite('canny-edge-{}.png'.format(file_name),filter)
	
	#
	# function to compare features of two images - file1 and file2
	#	
	def recog_image(self,file1,file2):
		known_image = face_recognition.load_image_file("{}".format(file1))
		unknown_image = face_recognition.load_image_file("{}".format(file2))      
		       
		biden_encoding = face_recognition.face_encodings(known_image)[0]	
		unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
		
		results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
		return(results)
	
	#
	# call all functions to enchance images
	#
	def enchance_images(self):
		for file in self.faces:
			self.clahe_histo(file)
		for file in self.faces:
			self.canny_edge(file)
	
	#
	# deals with images and calls 'recog_image()' against those images
	#
	def compare_faces(self):
		for file2 in self.faces:
			ctr=1
			flag=False #flag to check skipping of test cases
			diffctr=0
			for file1 in self.dataset:
				if(self.recog_image(file1,file2)==[True]):

					# retrieve some part of image file name from dataset
					# use stripped name to rename test images 
					
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
						# if there is a file in 'faces' which has no corresponding image in 'dataset'
						# ask the user to enter the name of the person
						inp=input("\n\n{} is unidentified.\nWho are you?? ".format(file2))
						print("\n{}.jpg was added to the dataset.".format(inp))
						os.rename(file2,"dataset/"+inp+".jpg")
			if(flag):
				continue
			

#
# the main function
#
if __name__=='__main__':
	
	recognition = face_recog()
	
	#recognition.enchance_images()
	recognition.compare_faces()
