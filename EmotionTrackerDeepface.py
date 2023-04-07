import cv2
from deepface import DeepFace

# initialize camera
cap = cv2.VideoCapture(0)

while True:
    # capture a frame
    ret, frame = cap.read()

    # perform emotion analysis
    result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

    # extract the first (and only) element of the list
    result = result[0]

    # extract emotions from the result
    emotions = result['emotion']

    # display the result on the frame
    i = 1
    for emotion in emotions:
        cv2.putText(frame, f"{emotion}: {emotions[emotion]}", (10, 40 * i), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        i += 1

    # display the frame
    cv2.imshow('Emotion Tracker', frame)

    # check for user input to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release camera and close window
cap.release()
cv2.destroyAllWindows()
