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

import faceBlendCommon as fbc
import cv2,argparse,dlib,time
import numpy as np

leftEye = [36, 37, 38, 39, 40, 41]
rightEye = [42, 43, 44, 45, 46, 47]
mouth = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59 ]
leftBrows = [17, 18, 19, 20, 21]
rightBrows = [22, 23, 24, 25, 26]

# kernal size for morphological opening
k = 7

# Mask the mouth, eye and brows
def applyMask(skinImage, points):

  tempMask = np.ones((skinImage.shape[0], skinImage.shape[1]), dtype = np.uint8)

  temp = []
  for p in leftEye:
    temp.append(( points[p][0], points[p][1] ))

  cv2.fillConvexPoly(tempMask, np.int32(temp), 0, 16, 0)

  temp = []
  for p in rightEye:
    temp.append(( points[p][0],points[p][1] ))

  cv2.fillConvexPoly(tempMask, np.int32(temp), 0, 16, 0)

  temp = []
  for p in leftBrows:
    temp.append(( points[p][0],points[p][1] ))

  cv2.fillConvexPoly(tempMask, np.int32(temp), 0, 16, 0)

  temp = []
  for p in rightBrows:
    temp.append(( points[p][0],points[p][1] ))

  cv2.fillConvexPoly(tempMask, np.int32(temp), 0, 16, 0)

  temp = []
  for p in mouth:
    temp.append(( points[p][0],points[p][1] ))

  cv2.fillConvexPoly(tempMask, np.int32(temp), 0, 16, 0)

  return cv2.bitwise_and(skinImage, skinImage, mask = tempMask)


def findSkinHSV(meanimg, frame):

  # Specify the offset around the mean value
  hsvHueOffset = 10
  hsvSatOffset = 50
  hsvValOffset = 150

  # Convert to the HSV color space
  hsv = cv2.cvtColor(meanimg,cv2.COLOR_BGR2HSV)[0][0]
  frameHSV = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

  # Find the range of pixel values to be taken as skin region
  minHSV = np.array([hsv[0] - hsvHueOffset, hsv[1] - hsvSatOffset, hsv[2] - hsvValOffset])
  maxHSV = np.array([hsv[0] + hsvHueOffset, hsv[1] + hsvSatOffset, hsv[2] + hsvValOffset])

  # Apply the range function to find the pixel values in the specific range
  skinRegionhsv = cv2.inRange(frameHSV,minHSV,maxHSV)

  # Apply Gaussian blur to remove noise
  skinRegionhsv = cv2.GaussianBlur(skinRegionhsv, (5, 5), 0)

  # Get the kernel for performing morphological opening operation
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
  skinRegionhsv = cv2.morphologyEx(skinRegionhsv, cv2.MORPH_OPEN, kernel, iterations = 3 )

  # Apply the mask to the image
  skinhsv = cv2.bitwise_and(frame, frame, mask = skinRegionhsv)

  return skinhsv

def findSkinYCB(meanimg, frame):

  # Specify the offset around the mean value
  CrOffset = 15
  CbOffset = 15
  YValOffset = 100

  # Convert to the YCrCb color space
  ycb = cv2.cvtColor(meanimg,cv2.COLOR_BGR2YCrCb)[0][0]
  frameYCB = cv2.cvtColor(frame,cv2.COLOR_BGR2YCrCb)

  # Find the range of pixel values to be taken as skin region
  minYCB = np.array([ycb[0] - YValOffset,ycb[1] - CrOffset, ycb[2] - CbOffset])
  maxYCB = np.array([ycb[0] + YValOffset,ycb[1] + CrOffset, ycb[2] + CbOffset])

  # Apply the range function to find the pixel values in the specific range
  skinRegionycb = cv2.inRange(frameYCB,minYCB,maxYCB)

  # Apply Gaussian blur to remove noise
  skinRegionycb = cv2.GaussianBlur(skinRegionycb, (5, 5), 0)

  # Get the kernel for performing morphological opening operation
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
  skinRegionycb = cv2.morphologyEx(skinRegionycb, cv2.MORPH_OPEN, kernel, iterations = 3 )

  # Apply the mask to the image
  skinycb = cv2.bitwise_and(frame, frame, mask = skinRegionycb)

  return skinycb


if __name__ == '__main__':

  # Load face detector
  faceDetector = dlib.get_frontal_face_detector()

  # Load landmark detector.
  landmarkDetector = dlib.shape_predictor("../../common/shape_predictor_68_face_landmarks.dat")

  ap = argparse.ArgumentParser()
  ap.add_argument("-f", "--filename",help="filename")
  args = vars(ap.parse_args())

  filename = "../data/images/hillary_clinton.jpg"

  if args["filename"]:
    filename = args["filename"]

  frame = cv2.imread(filename)

  # Find landmarks.
  landmarks = fbc.getLandmarks(faceDetector, landmarkDetector, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) 
  landmarks = np.array(landmarks)

  # specify the points for taking a square patch
  ix = landmarks[32][0]
  fx = landmarks[34][0]
  iy = landmarks[29][1]
  fy = landmarks[30][1]

  # Take a patch on the nose
  tempimg = frame[iy:fy,ix:fx,:]
  # showimage = np.copy(frame)
  # cv2.rectangle(showimage,(ix,iy),(fx,fy),(0,255,0),-1)

  # Compute the mean image from the patch
  meanimg = np.uint8([[cv2.mean(tempimg)[:3]]])

  # Find skin using HSV color space
  skinhsv = findSkinHSV(meanimg, frame)
  maskedskinhsv = applyMask(skinhsv, landmarks)
  cv2.putText(maskedskinhsv, "HSV", (50, 50), cv2.FONT_HERSHEY_COMPLEX, .9, (255,255,255), 1, cv2.LINE_AA)

  # Find skin using YCrCb color space
  skinycb = findSkinYCB(meanimg, frame)
  maskedskinycb = applyMask(skinycb, landmarks)
  cv2.putText(maskedskinycb, "YCrCb", (50, 50), cv2.FONT_HERSHEY_COMPLEX, .9, (255,255,255), 1, cv2.LINE_AA)

  # Display the results and save
  combined_masked = np.hstack([maskedskinhsv, maskedskinycb])
  cv2.imshow("MASKED OUTPUT",combined_masked)
  k = cv2.waitKey(0) & 0xFF
  cv2.imwrite("results/skinDetectThreshold.jpg",combined_masked)
