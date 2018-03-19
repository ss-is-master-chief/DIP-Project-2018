import face_recognition, glob

class face_recog:

        faces = []
        
        def __init__(self):
                pass
        
        def get_images(self):
        	for file in glob.glob("*.jpg"):
		        self.faces.append(file)
		print(self.faces)	
	
        def recog_image(self):
        	known_image = face_recognition.load_image_file("test-9.jpeg")
        	unknown_image = face_recognition.load_image_file("test-8.jpeg")
             
        	biden_encoding = face_recognition.face_encodings(known_image)[0]	
        	unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
        	results = face_recognition.compare_faces([biden_encoding], unknown_encoding)
        	print(results)
        	
if __name__ == '__main__':
        fr = face_recog()
        fr.get_images()
