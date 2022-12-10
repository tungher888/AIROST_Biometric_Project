## Arduino Board Setting for LED indicator
# include <servo.h>
# Servo LockServo;
# int data;
# int ServoLock=13;

# void setup() { 
#   Serial.begin(9600);
#   pinMode(ServoLock, OUTPUT);
#   digitalWrite (ServoLock, LOW);
#   myservo.attach(9);
  
#   Serial.println("So far so good, I hope...");
# }
 
# void loop() {
# while (Serial.available())

# { 
# data = Serial.read();
# }

# if (data == '1')

# digitalWrite (ServoLock, HIGH);
# LockServo.write(180);
# delay(10000);
# LockServo.write(180);

# else if (data == '0')

# digitalWrite (ServoLock, LOW);
# }

import face_recognition
import numpy as np
import cv2 as cv
import glob
import PySimpleGUI as sg
import time


video_capture = cv.VideoCapture(0)
pathway = r"F:\CodeBuilders\SharedProjectCode\SecurityImages\*.*"
layout = [[sg.Text("Welcome to Facial Recognition Security Lock! \n Please press 'Create New Profile' to register new user or 'Sign In' to begin or 'Quit' to exit.")], [sg.Button("Create New Profile")], [sg.Button("Sign In")], [sg.Button("Quit")]]
window = sg.Window("Facial Recognition System", layout)

while True:
    event, values = window.read()
    if event == "Create New Profile":
        window.close()
        layout = [[sg.Text("Steps:\n1) Click 'OK button to start.\n2) Press 'Space' to take a picture.\n3)Name your picture.(Include .jpg at the very end or press'0' to cancel.")], [sg.Button("OK")]]
        window = sg.Window("Create new profile", layout)

        while True:
            event, values = window.read()
            if event == "OK" :
                window.close()
                import F_R_Create_New_Profile_2
            if event ==  sg.WIN_CLOSED :
                break    
        window.close()
        
    if event == "Sign In": 
        window.close()
        import F_R_Sign_In
        break
    if event == sg.WIN_CLOSED or event == "Quit" :
        break
    
window.close()
