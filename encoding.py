import cv2
import face_recognition
import pickle
import os

# Import the images
image_path = 'C:/xampp/htdocs/capstone/uploads'
image_path_list = os.listdir(image_path)
image_list = []
user_list_ids = []

for path in image_path_list:
    image_list.append(cv2.imread(os.path.join(image_path, path)))
    user_list_ids.append(os.path.splitext(path)[0])
    
# Test values
print(f"Total Image Found: {len(image_list)}")
print(f"Total Number of Registered User: {len(user_list_ids)}")
print(image_list)
print(user_list_ids)

def image_encode(image_list):
    encode_list = []
    for img in image_list:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    return encode_list
        
print("Encoding images please wait . . .")
encode_known_image = image_encode(image_list)
encode_known_images_with_ids = [encode_known_image, user_list_ids] # need to store 
# Test
print(encode_known_image)

# Save the data
file = open("EncodedFile.p", 'wb')
pickle.dump(encode_known_images_with_ids, file)
file.close()
print("Encoded file saved.")