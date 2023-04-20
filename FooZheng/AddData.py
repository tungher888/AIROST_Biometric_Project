# Run this to add data to Firebase

# If you want to view the changes on a realtime database pls create ur own firebase and replace all the credentials
# Firebase "Realtime Database" credentials
# Those current credentials are linked to my Google account

import firebase_admin
from firebase_admin import credentials
from firebase_admin import db

cred = credentials.Certificate("airost-face-attendance.json")
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://airost-face-attendance-default-rtdb.asia-southeast1.firebasedatabase.app/"
})

ref = db.reference('Faces')

data = {
    "FZ":
        {
            "name": "Lai Foo Zheng",
            "major": "Software",
            "starting_year": 2017,
            "total_attendance": 7,
            "year": 4,
            "last_attendance_time": "2022-12-11 00:54:34"
        },
    "Emily":
        {
            "name": "Emily Blunt",
            "major": "Economics",
            "starting_year": 2021,
            "total_attendance": 12,
            "year": 1,
            "last_attendance_time": "2022-12-11 00:54:34"
        },
    "Elon":
        {
            "name": "Elon Musk",
            "major": "Physics",
            "starting_year": 2020,
            "total_attendance": 7,
            "year": 2,
            "last_attendance_time": "2022-12-11 00:54:34"
        }
}

for key, value in data.items():
    ref.child(key).set(value)
