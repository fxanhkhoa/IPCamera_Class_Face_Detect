from keras.preprocessing.image import img_to_array
from keras.models import load_model
import imutils
import cv2
import numpy as np
import sys
import dlib 
import os

# Avoid kmp duplicate errors
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# parameters for loading data and images
emotion_model_path = 'models/_mini_XCEPTION.106-0.65.hdf5'

# loading models
detector = dlib.get_frontal_face_detector()
emotion_classifier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["angry","disgust","scared", "happy", "sad", "surprised","neutral"]

def dlib_to_opencv(rect):
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
 
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

def face_emotion_recognize(address):
    cap = cv2.VideoCapture(0)

    while(True):
        ret, img = cap.read()
        if ret == True:

            dets = detector(img, 1)
            frame = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            for k, d in enumerate(dets):
                (x, y, w, h) = dlib_to_opencv(d)
                roi = frame[y:y + h, x:x + w]
                print(roi.shape)
                print(roi.shape[1])
                if (roi.shape[0] != 0) and (roi.shape[1]!= 0):
                    roi = cv2.resize(roi, (48, 48))
                    roi = roi.astype("float") / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)
                    preds = emotion_classifier.predict(roi)[0]
                    emotion_probability = np.max(preds)
                    # print(emotion_probability)
                    label = EMOTIONS[preds.argmax()]
                    # print(label)
                    startX = x
                    startY = y - 15 if y - 15 > 15 else y + 15
                    cv2.putText(img, label, (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.rectangle(img, (x, y), (x + w, y + h),(0, 0, 255), 2)

            cv2.imshow('Emotion', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    address = sys.argv[1]
    face_emotion_recognize(address)
    
