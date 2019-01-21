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
import numpy as np

# (R, G, B) is classified as skin if
# R > 95 and G > 40 and B > 20 and
# R > G and R > B and |R-G| > 15 and
# max{R, G, B} - min{R, G, B} > 15

if __name__ == '__main__':

  # Start webcam
  cap = cv2.VideoCapture(0)

  # Check if webcam opens
  if (cap.isOpened()== False):
    print("Error opening video stream or file")
  
  # Window for displaying output
  cv2.namedWindow("Skin Detection")

  while(1):

    # Read frame
    ret, image = cap.read()

    # Split frame into r, g and b channels
    b,g,r = cv2.split(image)

    # Set output to all zeros
    output = np.zeros(image.shape, dtype=np.uint8)

    # Specifying the rules
    rule1 = np.uint8(r>95)                  # R>95
    rule2 = np.uint8(g>40)                  # G > 40
    rule3 = np.uint8(b>20)                  # B > 20
    rule4 = np.uint8(r>g)                   # R > G
    rule5 = np.uint8(r>b)                   # R > B
    rule6 = np.uint8(abs(r-g)>15)           # |R-G| > 15

    # max{R, G, B} - min{R, G, B} > 15
    rule7 = np.uint8( ( np.maximum(np.maximum(b,g),r) - np.minimum(np.minimum(b,g),r) ) > 15)

    # Apply (AND) all the rules to get the skin mask
    skinMask = rule1 * rule2 * rule3 * rule4 * rule5 * rule6 * rule7

    # Using the mask to get the skin pixels
    output[ skinMask ==1 ] = image[ skinMask == 1]

    # Display results
    cv2.imshow("Skin Detection",output)
    if cv2.waitKey(1) & 0xFF == 27:
      break

  cv2.destroyAllWindows()
