import cv2
import numpy as np
import dlib 
import os

predictor_model = "./weight/shape_predictor_68_face_landmarks.dat"
face_recognition_model = './weight/dlib_face_recognition_resnet_model_v1.dat'

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_model)
facerec = dlib.face_recognition_model_v1(face_recognition_model)

name = ['Phong', 'Tuan']


def load_features(src):

    directories = [d for d in os.listdir(src) 
                   if os.path.isdir(os.path.join(src, d))]

    labels = []
    features = []
    for d in directories:
        label_dir = os.path.join(src, d)
        file_names = [os.path.join(label_dir, f) 
                      for f in os.listdir(label_dir) if f.endswith(".npy")]

        for f in file_names:
            features.append(np.load(f))
            labels.append(d)
    return features, labels


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


def save(save_path, image):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # print("[+]Save extracted feature to file : ", save_path)
    cv2.imwrite(save_path, image)


def check_face_recognition(data_dir, features, labels):

    predict_false = 0
    predict_true = 0

    directories = [d for d in os.listdir(data_dir) 
                   if os.path.isdir(os.path.join(data_dir, d))]

    for d in directories:
        label_dir = os.path.join(data_dir, d)
        file_names = [os.path.join(label_dir, f) 
                      for f in os.listdir(label_dir) if f.endswith(".jpg")]

        for f in file_names:
            img = cv2.imread(f)

            dets = detector(img, 1) 

            for k, d in enumerate(dets):
                (x, y, w, h) = dlib_to_opencv(d)
                shape = sp(img, d)
                face_descriptor = facerec.compute_face_descriptor(img, shape)
                feature = np.reshape(face_descriptor, (1, -1))
                result = compare_faces(features, labels, feature)
        
                if(result != d):
                    predict_true += 1
                    save_path = f.replace("detected", "recognized")
                    if result != 'unknown':
                        result = name[int(result)]
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    startX = x
                    startY = y - 15 if y - 15 > 15 else y + 15
                    cv2.putText(img, result, (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    save(save_path, img)
                else:
                    predict_false += 1
                    save_path = f.replace("detected", "unrecognized")
                    if result != 'unknown':
                        result = name[int(result)]
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)
                    startX = x
                    startY = y - 15 if y - 15 > 15 else y + 15
                    cv2.putText(img, result, (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    save(save_path, img)

    total = predict_true + predict_false
    accuracy = 0
    if(total != 0):
        accuracy = predict_true/total
    
    return predict_true, predict_false, accuracy
            


FEATURE_ROOT_PATH = "features"
IMAGE_ROOT_PATH = "images"
data_dir = os.path.join(IMAGE_ROOT_PATH, "detected")
features, labels = load_features(FEATURE_ROOT_PATH)
recognized, unrecognized, accuracy = check_face_recognition(data_dir, features, labels)

print("Recognized: ", recognized)
print("Unrecognized: ", unrecognized)
print("Total images: ", recognized + unrecognized)
print("Accuracy: ", accuracy)



