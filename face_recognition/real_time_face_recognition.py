import numpy as np
import sys
import os
from PIL import ImageFile
import cv2
import dlib
from sklearn import svm
from sklearn.externals import joblib
from sklearn import metrics
# import openface

# def pause():
#     programPause = input("Press the <ENTER> key to continue...")

predictor_model = "./weight/shape_predictor_68_face_landmarks.dat"
face_recognition_model = './weight/dlib_face_recognition_resnet_model_v1.dat'

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_model)
facerec = dlib.face_recognition_model_v1(face_recognition_model)
# face_aligner = openface.AlignDlib(predictor_model)  

def dlib_to_opencv(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
 
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def face_distance(face_encodings, labels, face_to_compare, tolerance):
    if len(face_encodings) == 0:
        return np.empty((0))

    preds = np.linalg.norm(face_encodings - face_to_compare, axis=1)
    sort_prob = np.argsort(preds)
    if preds[sort_prob[0]] > tolerance:
        return -1
    return labels[sort_prob[0]]

def compare_faces(known_face_encodings, labels, face_encoding_to_check, tolerance=0.45):
    label = face_distance(known_face_encodings, labels, face_encoding_to_check, tolerance)
    if label == -1:
        return 'unknown'
    else:
        return label

def load_features(src):
    print("[+] Load data....")
    data = []
    label = []
    with open(src, "r") as file:
        for i,line in enumerate(file):
            img_path = line[:-1]
            #print("[+] Read image  : ", img_path," id : ", i)
            if os.path.isfile(img_path) and img_path.find(".jpg") != -1:            
                save_path = img_path.replace("images", "features").replace(".jpg", ".npy")        
                if os.path.isfile(save_path):
                    lb = save_path.split("/")[1]
                    # lb1 = lb.split(".")[1]
                    # print (lb)
                    # print(save_path)
                    data.append(np.load(save_path))
                    label.append(lb)
    print("[+] Load data finished")
    return data, label

def face_recognition(features, labels):
    stream = "rtsp://foscam:foscam1@172.20.10.14:554/videoSub"
    cap = cv2.VideoCapture(stream)

    while(True):
        ret, frame = cap.read()
        if ret == True:

            # dets = detector(frame, 1)  #Xác định vị trí khuôn mặt trong bức ảnh
            # # win.add_overlay(dets)   #Vẽ khung hình bao quanh khuôn mặt

            # # pause()
            # # dlib.hit_enter_to_continue()

            # for k, d in enumerate(dets):
            #     (x, y, w, h) = dlib_to_opencv(d)

            #     # Xác định facial landmark trên khuôn mặt
            #     shape = sp(frame, d)

            #     # Vẽ facial landmark lên bức ảnh
            #     # win.add_overlay(shape)

            #     # Affine transformations using openface
            #     # alignedFace = face_aligner.align(534, img, d, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

            #     # Affine transformations using dlib
            #     face_chip = dlib.get_face_chip(frame, shape)

            #     # win1 = dlib.image_window()
            #     # win1.clear_overlay()
            #     # win1.set_image(face_chip)

            #     # dlib.hit_enter_to_continue()

            #     # Encoding faces
            #     face_descriptor = facerec.compute_face_descriptor(face_chip)
                
            #     feature = np.reshape(face_descriptor, (1, -1))
            #     result = compare_faces(features, labels, feature)

            #     # result = clf.predict(feature)[0]
                
            #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 1)
            #     startX = x
            #     startY = y - 15 if y - 15 > 15 else y + 15
            #     cv2.putText(frame, result, (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                
            #     print("The face is: " + result)
                # print(face_descriptor)
            cv2.imshow("Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__=="__main__":

    address = sys.argv[1]  #input parameter của script
    features_path = sys.argv[2]
    features, labels = load_features(features_path)
    # db = sys.argv[2]

    # clf = joblib.load(db + '/model.joblib')

    face_recognition(features, labels)
    
