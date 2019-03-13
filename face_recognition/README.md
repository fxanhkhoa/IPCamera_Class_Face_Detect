# face_recognition
----
# Introduction
----
This project using HOG + SVM for face detection, finding face landmarks, affine transformations for preprocessing, encoding faces using the Resnet model and using SVM for face recognition.
# Directory Structure
----
```
face_recognition
-db
--db1
---train.txt
---text.txt
--...
-result
--db1
---result.txt
---model.joblib
--...
-feature
--face classes
-images
--face classes
-test
--test image
-extract_features.py
-generate_db.py
-face_recognition.py
-train.py
-test.py
-weight
--dlib_face_recognition_resnet_model_v1.dat
--shape_predictor_68_face_landmarks.dat
```
# Usage
----
## 1. Face dataset:
Click this [link]() to download face dataset, extract it and put into folder images.
## 2. Generate training set and test set:
Using [this script](https://github.com/fxanhkhoa/IPCamera_Class_Face_Detect/blob/master/face_recognition/generate_db.py) to generate training set and test set as following command:
```
python3 generate_db.py images db/db1
```
It will create train.txt and test.txt in folder db/db1
## 3. Feature extraction:
Using [this script](https://github.com/fxanhkhoa/IPCamera_Class_Face_Detect/blob/master/face_recognition/extract_feature.py) to extract feature of images. The systax as following:
#### To extract feature of training set:
```
python3 extract_features.py db/db1/train.txt
```
#### To extract feature of test set:
```
python3 extract_features.py db/db1/test.txt
```
## 4. Training:
Using [this script](https://github.com/fxanhkhoa/IPCamera_Class_Face_Detect/blob/master/face_recognition/train.py) to train model with SVM linear algorithm. The syntax as following:
```
python3 train.py db/db1/train.txt db/db1/test.txt result/db1
```
It will create model.joblib in folder result/db1.
## 5. Testing:
Using [this script](https://github.com/fxanhkhoa/IPCamera_Class_Face_Detect/blob/master/face_recognition/test.py) to test model. The syntax as following:
```
python3 test.py db/db1/test.txt result/db1
```
It will create result.txt in folder result/db1. This file is the label result after using trained model to recognize.
## 6. Testing with your owm image:
Using [this script](https://github.com/fxanhkhoa/IPCamera_Class_Face_Detect/blob/master/face_recognition/face_recognization.py) to predict your owm image with trained model. The syntax as following:
```
python3 face_recognization.py <PATH_OF_IMAGE> result/db1
```
It will show label of face on image.

# References
----

## Documentations

* [Support Vector Machines with scikit-learn](https://www.datacamp.com/community/tutorials/svm-classification-scikit-learn-python)
* [Support Vector Machines](http://scikit-learn.org/stable/modules/svm.html)
