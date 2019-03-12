import sys
import cv2
import dlib
import os
import numpy as np
# import openface

def pause():
    programPause = input("Press the <ENTER> key to continue...")

predictor_model = "./weight/shape_predictor_68_face_landmarks.dat"
face_recognition_model = ('./weight/dlib_face_recognition_resnet_model_v1.dat')

detector = dlib.get_frontal_face_detector()   #Load face detector
sp = dlib.shape_predictor(predictor_model)  #Load 68 landmark
# face_aligner = openface.AlignDlib(predictor_model)  
facerec = dlib.face_recognition_model_v1(face_recognition_model)


image_path = sys.argv[1]  #input parameter của script

img = cv2.imread(image_path)
win = dlib.image_window()
win.clear_overlay()
win.set_image(img)

dets = detector(img, 1)  #Xác định vị trí khuôn mặt trong bức ảnh
win.add_overlay(dets)   #Vẽ khung hình bao quanh khuôn mặt

# pause()
dlib.hit_enter_to_continue()

for k, d in enumerate(dets):
    # Xác định facial landmark trên khuôn mặt
    shape = sp(img, d)

    # Vẽ facial landmark lên bức ảnh
    win.add_overlay(shape)

    # Affine transformations using openface
    # alignedFace = face_aligner.align(534, img, d, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

    # Affine transformations using dlib
    face_chip = dlib.get_face_chip(img, shape)

    # win1 = dlib.image_window()
    # win1.clear_overlay()
    # win1.set_image(face_chip)

    dlib.hit_enter_to_continue()

    # Encoding faces
    face_descriptor = facerec.compute_face_descriptor(face_chip)

    print(face_descriptor)
# pause()

# import sys
# import dlib
# import cv2
# import openface

# # You can download the required pre-trained face detection model here:
# # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
# predictor_model = "shape_predictor_68_face_landmarks.dat"

# # Take the image file name from the command line
# file_name = sys.argv[1]

# # Create a HOG face detector using the built-in dlib class
# face_detector = dlib.get_frontal_face_detector()
# face_pose_predictor = dlib.shape_predictor(predictor_model)
# face_aligner = openface.AlignDlib(predictor_model)

# # Take the image file name from the command line
# file_name = sys.argv[1]

# # Load the image
# image = cv2.imread(file_name)

# # Run the HOG face detector on the image data
# detected_faces = face_detector(image, 1)

# print("Found {} faces in the image file {}".format(len(detected_faces), file_name))

# # Loop through each face we found in the image
# for i, face_rect in enumerate(detected_faces):

# 	# Detected faces are returned as an object with the coordinates 
# 	# of the top, left, right and bottom edges
# 	print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))

# 	# Get the the face's pose
# 	pose_landmarks = face_pose_predictor(image, face_rect)

# 	# Use openface to calculate and perform the face alignment
# 	alignedFace = face_aligner.align(534, image, face_rect, landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)

# 	# Save the aligned image to a file
#     cv2.imwrite("aligned_face_{}.jpg".format(i), alignedFace)

