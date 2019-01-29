#!/usr/bin/env python
"""
  Copyright 2017 BIG VISION LLC ALL RIGHTS RESERVED
  
  This program is distributed WITHOUT ANY WARRANTY to the
  Plus and Premium membership students of the online course
  titled "Computer Visionfor Faces" by Satya Mallick for
  personal non-commercial use.
  
  Sharing this code is strictly prohibited without written
  permission from Big Vision LLC.
  
  For licensing and other inquiries, please email
  spmallick@bigvisionllc.com
  
"""

import cv2

if __name__ == '__main__':
  
  # Image paths
  imagePaths = ["../data/images/left_profile_face.jpg",
                "../data/images/right_profile_face.jpg"]
    
  # Load 20x34 LBPCascade detector for profile face.
  faceCascade = cv2.CascadeClassifier('../models/lbpcascade_profileface.xml')
                

  for imagePath in imagePaths:
    # Read image and convert it to grayscale
    im = cv2.imread(imagePath, cv2.IMREAD_COLOR)
    imGray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

    # Detect right face.
    faces = faceCascade.detectMultiScale(imGray, scaleFactor=1.2, minNeighbors=3)
    if len(faces) == 0 :
      faces = [];
    
    # Note that the detector is trained to detect faces facing  right.
    # So we flip the image to detect faces facing left.
    imGrayFlipped = cv2.flip(imGray, 1)

    # Detect left face
    facesFlipped = faceCascade.detectMultiScale(imGrayFlipped, scaleFactor=1.2, minNeighbors=3)

    # The x-coordinate of the detected face need to flipped.
    if len(facesFlipped) > 0:
      imh, imw, ch = im.shape
      for face in facesFlipped:
        # flip x1. x1New = image_width - face_width - x1Old
        face[0] = imw - face[2] - face[0]
        faces.append(face)

    # Draw rectangle
    for (x, y, w, h) in faces:
      cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)

    # Show final results
    cv2.imshow('Profile Face', im)
    if cv2.waitKey(0) & 0xFF == 27:
      cv2.destroyAllWindows()
