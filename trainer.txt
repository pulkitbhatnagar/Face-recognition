import cv2
import os
import numpy as np
from PIL import Image

recognizer = cv2.createLBPHFaceRecognizer()
path='datset'

def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    #create empth face list
    faces=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        faceImg=Image.open(imagePath).convert('L')
        faceNp=np.array(faceImg,'uint8')
        Id=int(os.path.split(imagePath)[-1].split('.')[1])
        faces.append(faceNp)
        print(Id)
        Ids.append(Id)
        cv2.imshow("training",faceNp)
        cv2.waitKey(10)
    return np.array(Ids),faces
Ids,faces = getImagesAndLabels('dataSet')
recognizer.train(faces,Ids)
recognizer.save('trainner/trainner.yml')
cv2.destroyAllWindows()
