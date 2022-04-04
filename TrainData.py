import cv2
import numpy as np
import os
from PIL import Image

recognizer = cv2.face.LBPHFaceRecognizer_create()

path = 'dataSet'

def getImageWithId(path):
    ### get url cua anh
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    print(imagePaths)
    faces = []
    IDs = []

    for imagePath in imagePaths:

        faceImage = Image.open(imagePath).convert('L')
        # am tran du lieu anh
        faceNP = np.array(faceImage, 'uint8')

        Id = int(imagePath.split('\\')[1].split('.')[1])

        faces.append(faceNP)

        IDs.append(Id)

        cv2.imshow('tranings',faceNP)
        cv2.waitKey(10)

    return faces, IDs

faces, Ids = getImageWithId(path)

recognizer.train(faces , np.array(Ids))

if not os.path.exists('recognizer'):
    os.makedirs('recognizer')

recognizer.save('recognizer/trainingData.yml')

cv2.destroyAllWindows()
