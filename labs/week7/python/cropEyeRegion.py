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

import glob,os,dlib,cv2
import numpy as np
import faceBlendCommon as fbc

# Load face detection and pose estimation models.
modelPath = "../../common/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(modelPath)

def getCroppedEyeRegion(targetImage):

  landmarks = fbc.getLandmarks(detector, predictor, cv2.cvtColor(targetImage, cv2.COLOR_BGR2RGB))

  #Get points from landmarks detector
  x1 = landmarks[0][0]
  x2 = landmarks[16][0]
  y1 = min(landmarks[24][1], landmarks[19][1])
  y2 = landmarks[29][1]


  cropped = targetImage[y1:y2,x1:x2,:]
  cropped = cv2.resize(cropped,(96, 32), interpolation = cv2.INTER_CUBIC)
  return cropped
