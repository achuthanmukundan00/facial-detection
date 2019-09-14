# Facial detection on images using Haar cascades

import cv2
import sys

imagePath = sys.argv[1]
cascadePath = sys.argv[2]

# Create a Haar cascade using XML file passed by user
faceCascade = cv2.CascadeClassifier(cascadePath)

# Fetch image and convert to grayscale
image = cv2.imread(imagePath)
grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Facial detection
# detectMultiScale() returns a list of rectangles where it has found objects
faces = faceCascade.detectMultiScale(
    grayImage, 
    scaleFactor = 1.1, 
    minNeighbors = 4, 
    minSize = (30, 30),
    flags = cv2.CASCADE_SCALE_IMAGE
)

# Draw red rectangles corresponding to the output of detectMultiScale()
# using cv2 built in rectangle fcn
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)

cv2.imshow("Faces Found!", image)
cv2.waitKey(0)