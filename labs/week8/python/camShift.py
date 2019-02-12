#!/usr/bin/python
# Copyright 2017 BIG VISION LLC ALL RIGHTS RESERVED
# 
# This code is made available to the students of 
# the online course titled "Computer Vision for Faces" 
# by Satya Mallick for personal non-commercial use. 
#
# Sharing this code is strictly prohibited without written
# permission from Big Vision LLC. 
#
# For licensing and other inquiries, please email 
# spmallick@bigvisionllc.com 
# 
import numpy as np
import cv2,sys,dlib,time

if __name__ == '__main__':

  filename = "../data/videos/face1.mp4"
  if len(sys.argv) == 2:
    filename = sys.argv[1]
  cap = cv2.VideoCapture(filename)

  # Read a frame and find the face region using dlib
  for i in range(5):
    ret,frame = cap.read()

  # Detect faces in the image
  faceDetector = dlib.get_frontal_face_detector()
  faceRects = faceDetector(frame, 0)

  if not len(faceRects):
    print("face not found")
    sys.exit()

  bbox = faceRects[0]

  # modify the dlib rect to opencv rect 
  currWindow = (int(bbox.left()), int(bbox.top()), int(bbox.right() - bbox.left()), int(bbox.bottom() - bbox.top()))

  # get the face region from the frame
  roiObject = frame[currWindow[1]:currWindow[1] + currWindow[3], currWindow[0]:currWindow[0] + currWindow[2]]
  hsvObject =  cv2.cvtColor(roiObject, cv2.COLOR_BGR2HSV)

  # GGGet the mask for calculating histogram of the object and also remove noise
  mask = cv2.inRange(hsvObject, np.array((0., 50., 50.)), np.array((180.,255.,255.)))
  cv2.imshow("mask",mask)
  cv2.imshow("Object",roiObject)

  # Find the histogram and normalize it to have values between 0 to 255
  histObject = cv2.calcHist([hsvObject], [0], mask, [180], [0,180])           
  cv2.normalize(histObject, histObject, 0, 255, cv2.NORM_MINMAX)

  # Setup the termination criteria, either 10 iterations or move by atleast 1 pt
  term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )
  i=0
  while(1):
    ret, frame = cap.read()

    if ret == True:
      # Convert to hsv color space
      hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

      # find the back projected image with the histogram obtained earlier
      backProjectImage = cv2.calcBackProject([hsv], [0], histObject, [0,180], 1)
      cv2.imshow("Back Projected Image", backProjectImage)

      # Compute the new window using CAM shift in the present frame
      rotatedWindow, currWindow = cv2.CamShift(backProjectImage, currWindow, term_crit)

      # Get the window used by mean shift
      x,y,w,h = currWindow
      
      # Get the rotatedWindow vertices
      rotatedWindow = cv2.boxPoints(rotatedWindow)
      rotatedWindow = np.int0(rotatedWindow)
      frameClone = frame.copy()

      # Display the current window used for mean shift
      cv2.rectangle(frameClone, (x,y), (x+w,y+h), (255, 0, 0), 2, cv2.LINE_AA)

      # Display the rotated rectangle with the orientation information
      frameClone = cv2.polylines(frameClone, [rotatedWindow], True, (0, 255, 0), 2, cv2.LINE_AA)
      cv2.putText(frameClone, "{},{},{},{}".format(x,y,w,h), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2, cv2.LINE_AA)
      cv2.putText(frameClone, "{}".format(rotatedWindow), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 255), 2, cv2.LINE_AA)
      cv2.imshow('CAM Shift Object Tracking Demo',frameClone)
      cv2.imwrite('{:03d}.jpg'.format(i),frameClone)
      i+=1

      k = cv2.waitKey(10) & 0xff
      if k == 27:
        break
    else:
      break

  cv2.destroyAllWindows()
  cap.release()
