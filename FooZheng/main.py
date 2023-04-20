# Code Merged with Victor's Emotion Tracker with Firebase

# Without changing the firebase stuff, you can first test this with ELon Musk's face,
# use another device with elon's picture and point it to the camera

import os
import pickle
import numpy as np
import cv2
import face_recognition
import cvzone
from datetime import datetime

import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
from firebase_admin import db

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# firebase Credentials linked to my Google Account
# Need to create your own if you want to see realtime database updating
cred = credentials.Certificate("airost-face-attendance.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://airost-face-attendance-default-rtdb.asia-southeast1.firebasedatabase.app/",
    'storageBucket': "airost-face-attendance.appspot.com"
})

bucket = storage.bucket()

# Load the pre-trained emotion recognition model
emotion_model = load_model('emotion_model.hdf5', compile=False)

# Define emotions list
emotions = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

# Load the face detection classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

imgBackground = cv2.imread('Resources/backgroundtest.png')

# Importing the mode images into a list
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))

# Load the encoding file
print("Loading Encode File ...")
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, faceIds = encodeListKnownWithIds
print("Encode File Loaded")

modeType = 0
counter = 0
id = -1
imgFace = []

while True:
    success, img = cap.read()

    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    faceCurFrame = face_recognition.face_locations(imgS)
    encodeCurFrame = face_recognition.face_encodings(imgS, faceCurFrame)

    imgBackground[162:162 + 480, 55:55 + 640] = img
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

    if faceCurFrame:
        # Convert the frame to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                              flags=cv2.CASCADE_SCALE_IMAGE)

        # For each detected face
        for (x, y, w, h) in faces:
            # Extract the face ROI
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray.astype("float") / 255.0
            roi_gray = img_to_array(roi_gray)
            roi_gray = np.expand_dims(roi_gray, axis=0)

            # Predict the emotion
            preds = emotion_model.predict(roi_gray)[0]
            emotion = emotions[np.argmax(preds)]

            for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
                matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
                faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)

                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    y1, x2, y2, x1 = faceLoc
                    y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4

                    # x1 += 55
                    # y1 += 162

                    bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1

                    # Rectangle and Text
                    imgBackground = cvzone.cornerRect(imgBackground, bbox, rt=0)
                    # cv2.rectangle(imgBackground, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    # cvzone.putTextRect(imgBackground, "Loading", (x1, y1))
                    cv2.putText(imgBackground, f"{emotion} | {faceIds[matchIndex]}", (x1 + 55, y1 + 162), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                    id = faceIds[matchIndex]
                    if counter == 0:
                        # cvzone.putTextRect(imgBackground, "Loading", (275, 400))
                        cv2.imshow("Face Attendance", imgBackground)
                        cv2.waitKey(1)
                        counter = 1
                        modeType = 1

            if counter != 0:

                if counter == 1:
                    # Get the Data
                    faceInfo = db.reference(f'Faces/{id}').get()
                    print(faceInfo)
                    # Get the Image from the storage
                    blob = bucket.get_blob(f'Images/{id}.png')
                    array = np.frombuffer(blob.download_as_string(), np.uint8)
                    imgFace = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)
                    # Update data of attendance
                    datetimeObject = datetime.strptime(faceInfo['last_attendance_time'],
                                                       "%Y-%m-%d %H:%M:%S")
                    secondsElapsed = (datetime.now() - datetimeObject).total_seconds()
                    print(secondsElapsed)
                    # Set Time Elapsed
                    if secondsElapsed > 10:
                        ref = db.reference(f'Faces/{id}')
                        faceInfo['total_attendance'] += 1
                        ref.child('total_attendance').set(faceInfo['total_attendance'])
                        ref.child('last_attendance_time').set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    else:
                        modeType = 3
                        counter = 0
                        imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

                if modeType != 3:

                    if 10 < counter < 30:
                        modeType = 2

                    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]

                    if counter <= 10:
                        cv2.putText(imgBackground, str(faceInfo['total_attendance']), (1000, 125),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
                        cv2.putText(imgBackground, str(faceInfo['major']), (1000, 550),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
                        cv2.putText(imgBackground, str(id), (1000, 500),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
                        cv2.putText(imgBackground, str(faceInfo['year']), (1000, 600),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 1)
                        cv2.putText(imgBackground, str(faceInfo['starting_year']), (1000, 650),
                                    cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 0), 1)

                        (w, h), _ = cv2.getTextSize(faceInfo['name'], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                        offset = (414 - w) // 2
                        cv2.putText(imgBackground, str(faceInfo['name']), (808 + offset, 445),
                                    cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 50), 1)

                        imgBackground[175:175 + 216, 909:909 + 216] = imgFace

                    counter += 1

                    if counter >= 30:
                        counter = 0
                        modeType = 0
                        faceInfo = []
                        imgFace = []
                        imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]
    else:
        modeType = 0
        counter = 0

    cv2.imshow("Face Attendance", imgBackground)
    cv2.waitKey(1)
