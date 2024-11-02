import cv2
import pickle
import face_recognition
import numpy as np
import cvzone

cap = cv2.VideoCapture(0)
cap.set(3, 900)
cap.set(4, 700)

image_background = cv2.imread('resources/background-img.png')

# Get the encoded files
print("Loading encoded files ...")
file = open('EncodedFile.p', 'rb')
encode_known_images_with_ids = pickle.load(file)
file.close()
encode_known_image, user_list_ids = encode_known_images_with_ids
print(user_list_ids)

while True: 
    success, img = cap.read()
    
    small_img = cv2.resize(img, (0,0), None, 0.25, 0.25)
    small_img = cv2.cvtColor(small_img, cv2.COLOR_BGR2RGB)
    
    current_face = face_recognition.face_locations(small_img) # old 
    encode_current_face = face_recognition.face_encodings(small_img, current_face) # new images
    
    source_image = cv2.resize(img, (400, 400))  # Resize to match target shape
    image_background[0:0+800, 0:0+400] = source_image
    
    for encode_face, face_location in zip(encode_current_face, current_face):
        matches = face_recognition.compare_faces(encode_known_image, encode_face)
        face_distance = face_recognition.face_distance(encode_known_image, encode_face)
        print("Match: ", matches)
        print("Face Distance: ", face_distance)
        
        match_index = np.argmin(face_distance)
        # Check match
        if matches[match_index]:
            print(f"Match Detected! User: {user_list_ids[match_index]}")
            y1, x2, y2, x1 = face_location
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            bbox = -150 + x1, -60 + y1, x2 - x1, y2 - y1
            image_background = cvzone.cornerRect(image_background, bbox, rt=0)
            
    #cv2.imshow("Camera", img)
    cv2.imshow("Face Recognition Attendance Monitoring", image_background)
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break