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

import os
import sys
import cv2
import random
import numpy as np

random.seed(42)
debug = False


def createDir(folder):
  try:
    os.makedirs(folder)
  except OSError:
    print('{}: already exists'.format(folder))
  except Exception as e:
    print(e)


def readImagePaths(folder, imgExts):
  imagePaths = []
  for x in os.listdir(folder):
    xPath = os.path.join(folder, x)
    if os.path.splitext(xPath)[1] in imgExts:
      imagePaths.append(xPath)
  return imagePaths


def genPatches(imagePath, patchSize, numPatches):
  im = cv2.imread(imagePath, cv2.IMREAD_COLOR)
  imh, imw, imch = im.shape
  pw, ph = patchSize
  patches = np.zeros((numPatches, ph, pw, imch), dtype=np.uint8)
  for i in range(numPatches):
    px1 = random.randint(0, imw - pw - 1)
    py1 = random.randint(0, imh - ph - 1)
    patches[i, :, :, :] = im[py1:py1 + ph, px1:px1 + pw, :]
  return patches


def writePatches(imagePath, patchesDir, patches):
  for i, patch in enumerate(patches):
    # print('{} - {}'.format(i, patch.shape))
    patchName = '{}_{}.jpg'.format(os.path.splitext(os.path.basename(imagePath))[0], str(i))
    patchPath = os.path.join(patchesDir, patchName)
    # print(patchPath)
    cv2.imwrite(patchPath, patch)


def cropImage(imagePath, margin=0):
  im = cv2.imread(imagePath, cv2.IMREAD_COLOR)
  imh, imw = im.shape[0:2]
  imCrop = im[margin:imh - margin, margin:imw - margin, :]
  return imCrop


if __name__ == '__main__':
  print("Run this script twice by passing following arguments\n" +
        "python generateCropPatches path_to_INRIA_folder/train_64x128_H96 16" + '\n'
        "python generateCropPatches path_to_INRIA_folder/test_64x128_H96 3" + '\n'
        )

  patchSize = (64, 128)
  rootDir = sys.argv[1]  # Train or train_64x128_H96
  margin = int(sys.argv[2])  # 16 for train. 3 for test

  numPatches = 10
  negImagesDir = os.path.join(rootDir, 'neg')
  negPatchesDir = os.path.join(rootDir, 'negPatches')
  createDir(negPatchesDir)

  # get all images from neg image directory
  negImagePaths = readImagePaths(negImagesDir, ['.jpg', '.png', '.jpeg'])
  numNegImages = len(negImagePaths)
  print(numNegImages)

  for index, negImagePath in enumerate(negImagePaths):
    print('{}:{} - {}'.format(index, numNegImages, negImagePath))
    negPatches = genPatches(negImagePath, patchSize, numPatches)
    writePatches(negImagePath, negPatchesDir, negPatches)

  posImagesDir = os.path.join(rootDir, 'pos')
  posPatchesDir = os.path.join(rootDir, 'posPatches')
  createDir(posPatchesDir)

  # get all images from neg image directory
  posImagePaths = readImagePaths(posImagesDir, ['.jpg', '.png', '.jpeg'])
  numPosImages = len(posImagePaths)
  print(numPosImages)

  for index, posImagePath in enumerate(posImagePaths):
    print('{}:{} - {}'.format(index, numPosImages, posImagePath))
    imCrop = cropImage(posImagePath, margin=margin)
    imCropName = os.path.join(posPatchesDir, os.path.splitext(os.path.basename(posImagePath))[0] + '.jpg')
    cv2.imwrite(imCropName, imCrop)
