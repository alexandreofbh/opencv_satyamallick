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

import cv2
import numpy as np
from utils import mouse_handler
from utils import get_four_points
import sys


if __name__ == '__main__' :

    # Read source image.
    im_src = cv2.imread('../data/images/first-image.jpg');
    size = im_src.shape
   
    # Create a vector of source points.
    pts_src = np.array(
                       [
                        [0,0],
                        [size[1] - 1, 0],
                        [size[1] - 1, size[0] -1],
                        [0, size[0] - 1 ]
                        ],dtype=float
                       );

    
    # Read destination image
    im_dst = cv2.imread('../data/images/times-square.jpg');

    # Get four corners of the billboard
    print('Click on four corners of a billboard and then press ENTER')
    pts_dst = get_four_points(im_dst)
    
    # Calculate Homography between source and destination points
    h, status = cv2.findHomography(pts_src, pts_dst);
    
    # Warp source image
    im_temp = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))

    # Black out polygonal area in destination image.
    cv2.fillConvexPoly(im_dst, pts_dst.astype(int), 0, 16);
    
    # Add warped source image to destination image.
    im_dst = im_dst + im_temp;
    
    # Display image.
    cv2.imshow("Image", im_dst);
    cv2.waitKey(0);

