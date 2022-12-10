import numpy as np
import cv2 as cv
import PySimpleGUI as sg
import os
import shutil

video_capture = cv.VideoCapture(0)
while(True) :
    ret, frame = video_capture.read()
    cv.imshow ('New Profile', frame)

    if cv.waitKey(1) & 0xFF == ord(' '):
        print(os.getcwd()) 
        layout = [[sg.Text("Enter your name (please include extension name '.png')")], [sg.Text("Name", size = (15,1)), sg.InputText()], [sg.Button("Submit")], [sg.Button("Cancel")]]
        window = sg.Window("Register New Profile", layout)

        while True:
            event, values = window.read()
            if event == "Submit":
                window.close()
                name = values[0]
            if event == "Cancel" or event == sg.WIN_CLOSED :
                window.close()
                break
        cv.imwrite(name, frame)
        break

    if cv.getWindowProperty('New Profile', cv.WND_PROP_VISIBLE) < 1 :
        break  
    
video_capture.release()
cv.destroyAllWindows()

Security_Profiles = [f for f in os.listdir() if '.jpg' in f.lower()]

for Security_Profile in Security_Profiles :
    New_Profile_Path = 'F:/CodeBuilders/SharedProjectCode/SecurityImages/' + Security_Profile
    shutil.move(Security_Profile, New_Profile_Path)