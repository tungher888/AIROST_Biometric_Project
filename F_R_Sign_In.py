import face_recognition
import numpy as np
import cv2 as cv
import glob
import serial
import time

video_capture = cv.VideoCapture(0)
Serial = serial.Serial('COM3',9600)
pathway = r"F:\CodeBuilders\SharedProjectCode\SecurityImages\*.*"

for file in glob.glob(pathway):
    image_face = face_recognition.load_image_file(file)
    image_face_encoding = face_recognition.face_encodings(image_face)[0]

    known_face_encodings = [
    image_face_encoding
    ]

face_locations = []
face_encodings = []
match_status = []
check = False
condition = "Door Locked"
chkc=False
Serial.write(b'0')

while (not check): 
    Serial.write(b'0')
    ret, frame = video_capture.read()

    small_frame = cv.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = small_frame[:, :, ::-1]
                        
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
                
    while not chkc:
        print(condition)
        chkc = True
        break
                
    for face_encoding in face_encodings:
        status = "NoMatch"
        match_status.append(status)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            
        if True in matches:
            first_match_index = matches.index(True)
            status = "MatchFound"
            match_status.append(status)
            check = True
            if check:
                print("Door Unlocked")     
                break

    for (top, right, bottom, left), name in zip(face_locations, match_status):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        cv.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv.FILLED)
        font = cv.FONT_ITALIC
        cv.putText(frame, status, (left + 6, bottom - 6), font, 0.8, (255, 255, 255), 1)

    cv.imshow('Video Feed', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    if cv.getWindowProperty('Video Feed',cv.WND_PROP_VISIBLE) < 1:        
        break  

video_capture.release()
if True in matches :
    Serial.write(b'1')
time.sleep(5)
cv.destroyAllWindows()