from sklearn.decomposition import PCA
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imutils
import cv2
import numpy as np
import sys
import dlib 
import os
import time

os.environ['KMP_DUPLICATE_LIB_OK']='True'

emotion_model_path = 'models/_mini_XCEPTION.106-0.65.hdf5'

detector = dlib.get_frontal_face_detector()
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry","disgust","scared", "happy", "sad", "surprised","neutral"]
CHECK = ["AN", "DI", "FE", "HA", "SA", "SU", "NE"]

def dlib_to_opencv(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
 
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def save(save_path, image):    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("[+]Save image to file : ", save_path)
    cv2.imwrite(save_path, image)

def check_face_emotion(data_dir):

    predict_an_true = 0
    predict_an_false = 0
    predict_di_true = 0
    predict_di_false = 0
    predict_fe_true = 0
    predict_fe_false = 0
    predict_ha_true = 0
    predict_ha_false = 0
    predict_sa_true = 0
    predict_sa_false = 0
    predict_su_true = 0
    predict_su_false = 0
    predict_ne_true = 0
    predict_ne_false = 0

    directories = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]

    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f) 
                      for f in os.listdir(label_dir) if f.endswith(".tiff")]

        for f in file_names:
            img = cv2.imread(f)
            split_name = f.split("/")[-1].split(".")[1]
            label = split_name[0] + split_name[1]
            # print(label)
            dets = detector(img, 1) 
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            for k, d in enumerate(dets):
                (x, y, w, h) = dlib_to_opencv(d)
                roi = frame[y:y + h, x:x + w]
                if (roi.shape[0] != 0) and (roi.shape[1]!= 0):
                    roi = cv2.resize(roi, (48, 48))
                    roi = roi.astype("float") / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)
                    preds = emotion_classifier.predict(roi)[0]
                    check = CHECK[preds.argmax()]
                    if (check == label):
                        if(label == 'AN'):
                            predict_an_true += 1
                        elif (label == 'DI'):
                            predict_di_true += 1
                        elif (label == 'FE'):
                            predict_fe_true += 1
                        elif (label == 'HA'):
                            predict_ha_true += 1
                        elif (label == 'SA'):
                            predict_sa_true += 1
                        elif (label == 'SU'):
                            predict_su_true += 1
                        elif (label == 'NE'):
                            predict_ne_true += 1
            
                        result = EMOTIONS[preds.argmax()]
                        startX = x
                        startY = y - 15 if y - 15 > 15 else y + 15
                        cv2.putText(img, result, (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.rectangle(img, (x, y), (x + w, y + h),(0, 0, 255), 2)
                        save_path = f.replace("original", "recognized")
                        save(save_path, img)
                    else:
                        if(label == 'AN'):
                            predict_an_false += 1
                        elif (label == 'DI'):
                            predict_di_false += 1
                        elif (label == 'FE'):
                            predict_fe_false += 1
                        elif (label == 'HA'):
                            predict_ha_false += 1
                        elif (label == 'SA'):
                            predict_sa_false += 1
                        elif (label == 'SU'):
                            predict_su_false += 1
                        elif (label == 'NE'):
                            predict_ne_false += 1

                        result = EMOTIONS[preds.argmax()]
                        startX = x
                        startY = y - 15 if y - 15 > 15 else y + 15
                        cv2.putText(img, result, (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        cv2.rectangle(img, (x, y), (x + w, y + h),(0, 0, 255), 2)
                        save_path = f.replace("original", "unrecognized")
                        save(save_path, img)


    total_an = predict_an_true + predict_an_false
    total_di = predict_di_true + predict_di_false
    total_fe = predict_fe_true + predict_fe_false
    total_ha = predict_ha_true + predict_ha_false
    total_sa = predict_sa_true + predict_sa_false
    total_su = predict_su_true + predict_su_false
    total_ne = predict_ne_true + predict_ne_false

    accuracy_an = 0
    accuracy_di = 0
    accuracy_fe = 0
    accuracy_ha = 0
    accuracy_sa = 0
    accuracy_su = 0
    accuracy_ne = 0

    if(total_an != 0):
        accuracy_an = predict_an_true/total_an
    if(total_di != 0):
        accuracy_di = predict_di_true/total_di
    if(total_fe != 0):
        accuracy_fe = predict_fe_true/total_fe
    if(total_ha != 0):
        accuracy_ha = predict_ha_true/total_ha
    if(total_sa != 0):
        accuracy_sa = predict_sa_true/total_sa
    if(total_su != 0):
        accuracy_su = predict_su_true/total_su
    if(total_ne != 0):
        accuracy_ne = predict_ne_true/total_ne
    
    print("***********************************************************************")
    print("EMOTION           PREDICT TRUE            PREDICT FALSE        ACCURACY")
    print("  Angry                " + str(predict_an_true) + "                   " +  str(predict_an_false) + "               " +  str(accuracy_an) )
    print("  Disgust              " + str(predict_di_true) + "                   " +  str(predict_di_false) + "               " +  str(accuracy_di) )
    print("  Scared               " + str(predict_fe_true) + "                  " +  str(predict_fe_false) + "               " +  str(accuracy_fe) )
    print("  Happy                " + str(predict_ha_true) + "                  " +  str(predict_ha_false) + "                " +  str(accuracy_ha) )
    print("  Sad                  " + str(predict_sa_true) + "                   " +  str(predict_sa_false) + "               " +  str(accuracy_sa) )
    print("  Surprised            " + str(predict_su_true) + "                  " +  str(predict_su_false) + "                " +  str(accuracy_su) )
    print("  Neutral              " + str(predict_ne_true) + "                  " +  str(predict_ne_false) + "                " +  str(accuracy_ne) )
    print("***********************************************************************")
            

ROOT_PATH = "images"
data_dir = os.path.join(ROOT_PATH, "original")
check_face_emotion(data_dir)

