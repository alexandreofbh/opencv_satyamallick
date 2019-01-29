/*
Copyright 2017 BIG VISION LLC ALL RIGHTS RESERVED

This program is distributed WITHOUT ANY WARRANTY to the
Plus and Premium membership students of the online course
titled "Computer Visionfor Faces" by Satya Mallick for
personal non-commercial use.

Sharing this code is strictly prohibited without written
permission from Big Vision LLC.

For licensing and other inquiries, please email
spmallick@bigvisionllc.com

*/

#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

int main(int argc, char **argv)
{
  // Image paths
  vector <string> imagePaths;
  imagePaths.push_back("../data/images/left_profile_face.jpg");
  imagePaths.push_back("../data/images/right_profile_face.jpg");

  // Load 20x34 LBPCascade detector for profile face.
  CascadeClassifier faceCascade("../models/lbpcascade_profileface.xml");

  for (int i = 0; i < imagePaths.size(); i++)
  {
    // Read image and convert it to grayscale
    Mat im = imread(imagePaths[i]);
    Mat imGray;
    cvtColor(im, imGray, COLOR_BGR2GRAY);


    // Detect right face
    vector <Rect> faces;
    faceCascade.detectMultiScale(imGray, faces, 1.2, 3);

    // Note that the detector is trained to detect faces facing  right.
    // So we flip the image to detect faces facing left.
    Mat imGrayFlipped;
    flip(imGray, imGrayFlipped, 1);

    // Detect left face
    vector <Rect> facesFlipped;
    faceCascade.detectMultiScale(imGrayFlipped, facesFlipped, 1.2, 3);

    // The x-coordinate of the detected face need to flipped.
    if (facesFlipped.size() > 0)
    {
      Size size = im.size();
      for (int j = 0; j < facesFlipped.size(); j++)
      {
        // flip x1. x1New = image_width - face_width - x1Old
        facesFlipped[j].x = size.width - facesFlipped[j].width - facesFlipped[j].x;
        faces.push_back(facesFlipped[j]);
      }
    }

    // Draw rectangle
    for (int j = 0; j < faces.size(); j++)
    {
      rectangle(im, faces[j], Scalar(255, 0, 0), 2);
    }

    // Show final results
    imshow("Profile Face", im);
    int key = waitKey(0);
  }

  destroyAllWindows();
  return EXIT_SUCCESS;

}

