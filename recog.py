import face_recognition, glob

class face_recog:

        faces = []
        
        def __init__(self):
                pass
        
        def get_images(self):
        
        	for file in glob.glob("*.jpg"):
		        self.faces.append(file)
		print(self.faces)	
	
	def recog_image(self,file1,file2):
        	
        	known_image = face_recognition.load_image_file("{}".format(file1))
        	unknown_image = face_recognition.load_image_file("{}".format(file2))
             
        	biden_encoding = face_recognition.face_encodings(known_image)[0]	
        	unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
        	results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
        	
        	return(results)
        	
if __name__ == '__main__':
        fr = face_recog()
	fr.get_images()
	for file1 in fr.faces:
		for file2 in fr.faces:
			if(fr.recog_image(file1,file2)==True):
				print("{} and {} are same".format(file1,file2))
			else:
				print("{} and {} are different".format(file1,file2))
