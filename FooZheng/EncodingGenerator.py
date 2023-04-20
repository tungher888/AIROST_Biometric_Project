# Run this to generate a new Face Encoding File "EncodeFile.p" which will be used in "main.py"

# Firebase "Storage" credentials
# Those current credentials are linked to my Google account

import cv2
import face_recognition
import pickle
import os

import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage

cred = credentials.Certificate("airost-face-attendance.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://airost-face-attendance-default-rtdb.asia-southeast1.firebasedatabase.app/",
    'storageBucket': "airost-face-attendance.appspot.com"
})

# Importing face images
folderPath = 'Images'
pathList = os.listdir(folderPath)
print(pathList)
imgList = []
faceIds = []
for path in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath, path)))
    faceIds.append(os.path.splitext(path)[0])

    # Uploading encodings to Storage (firebase)
    fileName = f'{folderPath}/{path}'
    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    blob.upload_from_filename(fileName)

print(faceIds)


def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList


print("Encoding Started ...")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown, faceIds]
print("Encoding Complete")

file = open("EncodeFile.p", 'wb')
pickle.dump(encodeListKnownWithIds, file)
file.close()
print("File Saved")