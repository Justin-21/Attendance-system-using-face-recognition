#Importing the Necessary Modules
import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


# Creating the Path of Resources Folder
path = 'Resources'
# Creating A Image Array
images = []
#Creating The Name Array Where it stored all the names of the faces
Names = []

#get the list of all the files in this directory
myList = os.listdir(path)

"""
    Creating the function for the face Encoding
    soo basically face_recognition works on dlib and lib can splitting our face into 128 
    with 128 features it can recognize anyone's face
    Basically it use HOG algorithms for encoding the images
"""


for cu_img in myList:
    current_Img = cv2.imread(f'{path}/{cu_img}')
    images.append(current_Img)
    Names.append(os.path.splitext(cu_img)[0])


def faceEncodings(images):

    # Creating the encode list where we store the encoded list

    encodeList = []
    for img in images:
        """Converting the img into bgr to rgb """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


def attendance(name):
    with open('Attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'{dStr}')
            f.writelines(f'\n{name},{tStr}')



encodeListKnown = faceEncodings(images)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    # Now we have to find the face location and face encoding from the webcam

    facesCurrentFrame = face_recognition.face_locations(faces)
    encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)

    # Now further we have to find the face distance and matching the face

    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        # print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = Names[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            attendance(name)


    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()