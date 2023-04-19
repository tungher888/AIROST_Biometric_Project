import face_recognition
import numpy as np
import cv2 as cv
import glob
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

# Load the pre-trained emotion recognition model
emotion_model = load_model('F:\CodeBuilders\SharedProjectCode\AIROST_Biometric_Project\AIROST_Project_MSI\emotion_model.hdf5', compile=False)

# Define emotions list
emotions = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]

# Load the face detection classifier
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load images for facial recognition
known_face_encodings = []
pathway = "F:\CodeBuilders\SharedProjectCode\SecurityImages\*.*"
for file in glob.glob(pathway):
    image_face = face_recognition.load_image_file(file)
    image_face_encoding = face_recognition.face_encodings(image_face)[0]
    known_face_encodings.append(image_face_encoding)

# Start the video capture
video_capture = cv.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = video_capture.read()
    
    # Convert the frame to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    
    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv.CASCADE_SCALE_IMAGE)
    
    # For each detected face
    for (x, y, w, h) in faces:
        # Extract the face ROI
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype("float") / 255.0
        roi_gray = img_to_array(roi_gray)
        roi_gray = np.expand_dims(roi_gray, axis=0)

        # Predict the emotion
        preds = emotion_model.predict(roi_gray)[0]
        emotion = emotions[np.argmax(preds)]

        # Perform facial recognition
        face_encodings = face_recognition.face_encodings(frame, [(y, x+w, y+h, x)])
        match_status = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            status = "Unknown"
            print("Door Locked")

            if True in matches:
                first_match_index = matches.index(True)
                status = "MatchFound"
                print("Door Unlocked")
            match_status.append(status)

        # Draw a rectangle around the face and display the predicted emotion and match status
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(frame, f"{emotion} | {match_status[0]}", (x, y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Display the resulting frame
    cv.imshow('frame', frame)
    
    # Exit the loop if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the window
video_capture.release()
cv.destroyAllWindows()
