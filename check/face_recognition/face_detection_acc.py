import cv2
import numpy as np
import dlib 
import os

detector = dlib.get_frontal_face_detector()

def save(save_path, image):    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("[+]Save image to file : ", save_path)
    cv2.imwrite(save_path, image)

def check_face_detection(data_dir):

    undetected = 0
    detected = 0

    directories = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]

    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f) 
                      for f in os.listdir(label_dir) if f.endswith(".jpg")]

        for f in file_names:
            img = cv2.imread(f)
            dets = detector(img, 1) 
            if(len(dets) == 0):
                undetected += 1
                save_path = f.replace("original", "undetected")
                save(save_path, img)
            else:
                detected += 1
                save_path = f.replace("original", "detected")
                save(save_path, img)

    total = detected + undetected
    accuracy = 0
    if(total != 0):
        accuracy = detected/total
    
    return detected, undetected, accuracy
            


ROOT_PATH = "images"
data_dir = os.path.join(ROOT_PATH, "original")
detected, undetected, accuracy = check_face_detection(data_dir)
print("Detected: ", detected)
print("Undetected: ", undetected)
print("Total images: ", detected + undetected)
print("Accuracy: ", accuracy)

