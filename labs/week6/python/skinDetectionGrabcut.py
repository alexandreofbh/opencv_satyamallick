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
import numpy as np
import cv2,argparse
import dlib

# Draws a polyline on an image (mask). The polyline is specified by a list of indices (pointsIndex) in a vector of points
def drawPolyline(mask, landmark, pointsIndex, color, thickness):
  linePoints = []
  for index in pointsIndex:
    linePoints.append((landmark[index][0], landmark[index][1]))
    # Fill face region with foreground indicator GC_FGD
  cv2.polylines(mask, np.int32([linePoints]), False, color, int(thickness))

# Draws a polygon on an image (mask). The polygon is specified by a list of indices (pointsIndex) in a vector of points
def drawPolygon(mask, landmark, pointsIndex, color):
  polygonPoints = []
  for index in pointsIndex:
    polygonPoints.append((landmark[index][0], landmark[index][1]))
    # Fill face region with foreground indicator GC_FGD
  cv2.fillConvexPoly(mask, np.int32(polygonPoints), color)


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
  img = cv2.imread(filename)

  # Find landmarks.
  landmarks = fbc.getLandmarks(faceDetector, landmarkDetector, cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
  landmarks = np.array(landmarks)

  # Calculate face mask by finding the convex hull and filling it GC_FGD
  faceMask = np.zeros(img.shape[:2],np.uint8)
  hull = cv2.convexHull(landmarks, False, True)

  # Convert to array for fillConvexPoly
  hullInt = []
  for i in range(0, len(hull)):
    hullInt.append((hull[i][0][0],hull[i][0][1]))

  # Fill face region with foreground indicator GC_FGD
  cv2.fillConvexPoly(faceMask, np.int32(hullInt), cv2.GC_FGD)
  # cv2.imshow("facemask", faceMask*60)

  # Create a mask of possible foreground and possible background regions.
  # This regions will be partial ellipses.
  mask = np.zeros(faceMask.shape[:2], dtype=faceMask.dtype)

  # The center of face is defined as the center of the
  # two points connecting the two ends of the jaw line.
  # This point serves as the center of the ellipse.
  faceCenter = (landmarks[16] + landmarks[0]) / 2

  # The two radii of the ellipse will be defined as a factor
  # the radius defined below.
  radius = np.linalg.norm(faceCenter - landmarks[0])

  # The angle of the ellipse is determined by the line
  # connecting the corners of the eyes.
  eyeVector = landmarks[45] - landmarks[36]
  angle = 180.0 * np.arctan2(eyeVector[1], eyeVector[0]) / np.pi

  # convert all values to int for using cv.ellipse in python
  center = (int(round(faceCenter[0] )),int(round(faceCenter[1] )))

  # Draw outer elliptical area that indicates probable background (GC_PR_BGD)
  cv2.ellipse(mask, center, (int(round(1.3*radius)),int(round(1.2*radius))), angle, 150, 390, cv2.GC_PR_BGD, -1)

  # Draw a smaller elliptical area that indicates probable foreground region(GC_PR_FGD)
  cv2.ellipse(mask, center, (int(round(0.9*radius)),int(round(radius))), angle, 150, 390, cv2.GC_PR_FGD,-1)
  # cv2.imshow("mask1", mask*60)

  # Copy the faceMask over this mask
  locs = np.where(faceMask != 0)
  mask[locs[0], locs[1]] = faceMask[locs[0], locs[1]]
  # cv2.imshow("mask after copying", mask*60)

  # Define relevant parts of the face for mask calculation.
  # Indices of left eye points in landmarks
  leftEyeIndex = [36, 37, 38, 39, 40, 41]

  # Indices of right eye points in landmarks
  rightEyeIndex = [42, 43, 44, 45, 46, 47]

  # Indices of mouth points in landmarks
  mouthIndex = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

  # Indices of left eyebrow points in landmarks
  leftEyeBrowIndex = [17, 18, 19, 20, 21]

  # Indices of right eyebrow points in landmarks
  rightEyeBrowIndex = [22, 23, 24, 25, 26]

  backgroundColor = cv2.GC_BGD

  # Remove eyes and mouth region by setting the mask to GC_BGD
  drawPolygon(mask, landmarks, leftEyeIndex, backgroundColor)
  drawPolygon(mask, landmarks, rightEyeIndex, backgroundColor)
  drawPolygon(mask, landmarks, mouthIndex, backgroundColor)

  # Remove eyebrows by setting the mask to GC_BGD
  # The eyebrows are defined by a polyline. So we have to specify a thickness
  # The thickness is chosen as a factor of the distance between the eye corners
  thickness = 0.1 * np.linalg.norm((landmarks[36] - landmarks[45]))
  drawPolyline(mask, landmarks, leftEyeBrowIndex, backgroundColor, thickness)
  drawPolyline(mask, landmarks, rightEyeBrowIndex, backgroundColor, thickness)

  # Array for storing background and foreground models
  bgdModel = np.zeros((1, 65), np.float64)
  fgdModel = np.zeros((1, 65), np.float64)

  # Apply grabcut
  cv2.grabCut(img, mask, None, bgdModel, fgdModel, 1, cv2.GC_INIT_WITH_MASK)

  # Final mask is the union of definitely foreground and probably foreground
  # mask such that all 1-pixels (cv2.GC_FGD) and 3-pixels (cv2.GC_PR_FGD) are put to 1 (ie foreground) and
  # all rest are put to 0(ie background pixels)
  mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),1,0).astype('uint8')

  # Copy the skin region to output
  output = img *(mask[:,:,np.newaxis])

  # Display mask
  cv2.namedWindow("mask", cv2.WINDOW_NORMAL)
  # Since the mask values are between 0 and 3, we need to scale it for display
  cv2.imshow("mask", mask*60)

  # Display extracted skin region
  cv2.namedWindow("Skin Detection", cv2.WINDOW_NORMAL)
  cv2.imshow("Skin Detection", output)
  cv2.imwrite("results/skinDetectionGrabcut.jpg", output)


  ################## Skin Smoothing  ###################
  # diameter of the pixel neighbourhood used during filtering
  dia = 15;

  # Larger the value the distant colours will be mixed together
  # to produce areas of semi equal colors
  sigmaColor = 70;

  # Larger the value more the influence of the farther placed pixels
  # as long as their colors are close enough
  sigmaSpace = 70;

  #Apply bilateralFilter
  smoothMask = cv2.bilateralFilter(output, dia, sigmaColor, sigmaSpace)

  smoothImage = np.copy(img)

  smoothImage[output != 0] = smoothMask[output != 0]

  cv2.imshow("Skin Smoothing", smoothImage)
  cv2.imwrite("results/skinSmoothing.jpg", smoothImage)

  cv2.waitKey(0)
  cv2.destroyAllWindows()
