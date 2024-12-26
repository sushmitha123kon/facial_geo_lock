from flask import Flask, render_template, Response, request, send_file, jsonify
import cv2
import numpy as np
import face_recognition
import os
import base64
from datetime import datetime
import geocoder 
import math
import socket

app = Flask(__name__)

# classNames=[]
# encodeListKnown=[]
# Directory to store user images
image_directory = "static/images"
office_latitude = 38.83278
office_longitude = -77.30655
radius = 29000
1
@app.route('/')
def index():
    return render_template('index.html')

# Global variables for class names and known encodings
classNames = []
encodeListKnown = []

@app.route('/capture_image', methods=['POST'])
def capture_image():
    # Get the image data from the request
    data = request.json
    name = data['name']
    image_data = data['image_data']
    
    # Decode base64-encoded image data
    image_bytes = base64.b64decode(image_data.split(',')[1])
    
    # Convert image bytes to OpenCV format
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Perform face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # If no faces detected, return an error message
    if len(faces) == 0:
        return jsonify({"error": "No faces detected"})
    
    # Crop the first detected face
    x, y, w, h = faces[0]
    cropped_face = img[y:y+h, x:x+w]

    # Convert the cropped face back to base64-encoded image data
    _, buffer = cv2.imencode('.jpg', cropped_face)
    cropped_image_data = base64.b64encode(buffer).decode('utf-8')
    
    # Generate a unique filename (you can use a timestamp or UUID)
    filename = f"{name}-{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
    
    # Save the cropped image to the specified directory
    with open(os.path.join(image_directory, filename), 'wb') as f:
        f.write(base64.b64decode(cropped_image_data))
    
    # Call functions to load images and find encodings
    load_images()
    print('Images loaded')
    return jsonify({"filename": filename})
# def capture_image():
#     # Get the image data from the request
#     data = request.json
#     name = data['name']
#     image_data = data['image_data']
    
#     # Decode base64-encoded image data
#     image_bytes = base64.b64decode(image_data.split(',')[1])
    
#     # Generate a unique filename (you can use a timestamp or UUID)
#     filename = f"{name}-{datetime.now().strftime('%Y%m%d%H%M%S')}.jpg"
    
#     # Save the image to the specified directory
#     with open(os.path.join(image_directory, filename), 'wb') as f:
#         f.write(image_bytes)
    
#     # Call functions to load images and find encodings
#     load_images()
#     print('Images loaded')
#     return jsonify({"filename": filename})

def load_images():
    global encodeListKnown, classNames
    images = []
    myList = os.listdir(image_directory)
    print(myList)
    for cl in myList:
        curImg = cv2.imread(os.path.join(image_directory, cl))
        images.append(curImg)
        classNames.append(cl)  # Use the full filename
    encodeListKnown = findEncodings(images)
    print('Encoding Complete')
    print(encodeListKnown)


def findEncodings(images):
    encodeList = []
    for img in images:
        # Debugging: Print out the shape of the image
        print("Image shape:", img.shape if img is not None else "None")
        
        # Check if the image is empty or None
        if img is None:
            print("Error: Empty image encountered")
            continue  # Skip this image and proceed to the next one
        
        # Convert the color space if the image is not empty
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Encode the face
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

def calculate_distance(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, sqrt, atan2

    # Convert latitude and longitude from degrees to radians
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    # Calculate the differences in latitude and longitude
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Calculate the Haversine formula parameters
    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    
    # Ensure that the value of 'a' is within the valid range [-1, 1]
    if a > 1:
        a = 1
    elif a < -1:
        a = -1

    # Calculate the distance using the Haversine formula
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    distance = 6371 * c * 1000  # Radius of the Earth in meters
    
    return distance




class AttendanceSystem:
    def mark_attendance(self, name):
        with open(r"C:\Users\Administrator\Desktop\New Text Document (2).txt", 'a+') as f:
            myDataList = f.readlines()
            nameList = [entry.split(',')[0] for entry in myDataList]
            if name not in nameList:
                time_now = datetime.now()
                tString = time_now.strftime('%H:%M:%S')
                dString = time_now.strftime('%d/%m/%Y')
                f.write(f'\n{name},{tString},{dString}')

    def recognize_faces(self, frame_data, threshold=0.6):
        global classNames, encodeListKnown

        # Decode the image data from the request
        image_bytes = base64.b64decode(frame_data)

        # Convert the image bytes to a NumPy array
        nparr = np.frombuffer(image_bytes, np.uint8)

        # Decode the image using OpenCV's imdecode function
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Check if the input image is valid
        if img is None:
            print("Error: Invalid input image")
            return None

        # Resize the image
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        # Perform face detection and recognition
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        # Check if any face encodings were found
        if not encodesCurFrame:
            print("No faces detected")
            # Encode the processed image
            _, buffer = cv2.imencode('.jpg', img)
            return buffer.tobytes()

        # Process the detected faces
        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            face_recognized = False
            matches = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(matches)
            if matches[matchIndex] <= threshold:
                face_recognized = True
                name = classNames[matchIndex].upper()
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2-35), (x2, y2), (0, 250, 0), cv2.FILLED)
                cv2.putText(img, name, (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                self.mark_attendance(name)
                print("Attendance marked for:", name)

        # If no face is recognized, print a message
        if not face_recognized:
            print("Face not recognized")

        # Encode the processed image
        _, buffer = cv2.imencode('.jpg', img)
        return buffer.tobytes()




attendance_system = AttendanceSystem()
# @app.route('/login', methods=['POST'])
# def login():
#     # Retrieve the user's geographic coordinates based on their IP address
#     user_location = geocoder.ip('me').latlng
    
#     # Calculate the distance between the user's location and the office premises
#     distance = calculate_distance(user_location[0], user_location[1], office_latitude, office_longitude)
    
#     # Check if the user's location is within the geofence boundary
#     if distance <= radius:
#         return jsonify({'status': 'success', 'message': 'Location verified. Access granted.'})
#     else:
#         return jsonify({'status': 'error', 'message': 'User is outside the designated area. Access denied.'}), 403
    
@app.route('/video_feed', methods=['POST'])
def video_feed():
# Get the local IP address
    local_ip = socket.gethostbyname(socket.gethostname())
    print("Local IP address:", local_ip)
    # Retrieve the user's geographic coordinates based on their IP address or GPS location
    user_location = geocoder.ip('me').latlng # Example using IP address, replace with actual method to get user's location
    print(user_location)
    # Calculate the distance between the user's location and the specified location
    distance = calculate_distance(user_location[0], user_location[1], office_latitude, office_longitude)
    print(distance)
    
     # Check if the user's location is within the specified boundaries
    if distance <= radius:
         # User is within the specified location boundaries, allow attendance to be taken
         print("here")
         frame_data = request.data  # Get the raw image data from the request
         processed_image_data = attendance_system.recognize_faces(frame_data)
         return Response(processed_image_data, mimetype='image/jpeg')
    else:
        # User is outside the specified location boundaries, deny access to take attendance
        print("outside area")
        return jsonify({'status': 'error', 'message': 'User is outside the designated area. Access denied.'}), 403
                                                                                                                                                                                                                                                            

if __name__ == '__main__':
    load_images()
    app.run(debug=True)