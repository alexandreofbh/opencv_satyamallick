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
#include <iostream>

using namespace cv;
using namespace std;

Mat src,cloneimg,blemish;
cv::Rect roi;

// Default radius
int radius = 15;
Point center;

Rect findBestSquare(Mat img)
{
  float minVal = 1000;
  cv::Rect bestSquare, checkSquare;
  int nSquares = 3;
  Point startPosition (center.x - nSquares*radius, center.y - nSquares*radius);
  Point newPosition(startPosition);

  // check all the squares in a nSquares x nSquares neighborhood around the required point
  for (int i = 0; i < nSquares; i++)
  {
    for (int j = 0; j < nSquares; j++)
    {
      // take a square from the neighborhood of the center
      newPosition.x = startPosition.x + i*2*radius;
      newPosition.y = startPosition.y + j*2*radius;

      // check for out of bounds
      if(newPosition.x + 2*radius > img.cols || newPosition.x < 0 || newPosition.y + 2*radius > img.rows || newPosition.y < 0)
        continue;

      checkSquare = cv::Rect( newPosition.x, newPosition.y, 2*radius, 2*radius );

      //find the gradient image
      Mat sobelX,SobelY;
      Sobel(cloneimg(checkSquare), sobelX, CV_32F, 1, 0);
      Sobel(cloneimg(checkSquare), SobelY, CV_32F, 0, 1);

      //find a measure of roughness of the square block
      float meanSobelX = cv::mean(cv::mean(abs(sobelX))).val[0];
      float meanSobelY = cv::mean(cv::mean(abs(SobelY))).val[0];

      //if it is smoother than previous ones update the best square
      if ((meanSobelX + meanSobelY) < minVal)
      {
        minVal = meanSobelX + meanSobelY;
        bestSquare = checkSquare;
      }
      else
        continue;
    }
  }
  return bestSquare;
}


void onMouse( int event, int x, int y, int flags, void* userdata )
{
  // get left click from the mouse
  if( event == EVENT_LBUTTONDOWN )
  {

    center = Point(x,y);
    roi = Rect( center.x - radius, center.y - radius, 2 * radius, 2 * radius );

    // check if the selected region is on the boundaries
    if(roi.x + roi.width > cloneimg.cols || roi.x < 0 || roi.y + roi.height > cloneimg.rows || roi.y < 0)
      return;

    blemish = cloneimg(roi).clone();

    // find the smoothest region around the marked point
    cv::Rect bestSquare = findBestSquare(cloneimg);

    Mat smoothRegion = cloneimg(bestSquare);

    // Create a circular white mask of the same size as the smooth region
    Mat mask = Mat::zeros(blemish.rows, blemish.cols, blemish.depth());
    circle(mask, Point(radius, radius), radius, Scalar(255, 255, 255), -1, 8);

    // Perform Seamless Cloning
    seamlessClone(smoothRegion,cloneimg,mask,center,cloneimg,NORMAL_CLONE);

    imshow("Blemish Remover",cloneimg);
  }

  // added functionality for UNDO-ing the last modification
  if( event == EVENT_RBUTTONDOWN )
  {
    Mat mask = 255 * Mat::ones(blemish.rows, blemish.cols, blemish.depth());
    seamlessClone(blemish,cloneimg,mask,center,cloneimg,NORMAL_CLONE);
    imshow("Blemish Remover",cloneimg);
  }
}


int main( int argc, const char** argv )
{
  // load the image
  string filename = "../data/images/blemish.png";

  if (argc == 2)
  {
    filename = argv[1];
  }

  if (argc == 3)
  {
    filename = argv[1];
    radius = atoi(argv[2]);
  }

  src = imread(filename);

  if(src.empty())
  {
    return -1;
  }

  // setup the mouse callback function
  namedWindow("Blemish Remover", WINDOW_NORMAL);   // changed from WINDOW_AUTOSIZE

  cloneimg = src.clone();

  setMouseCallback( "Blemish Remover", onMouse, &cloneimg );
  while(1)
  {
    imshow( "Blemish Remover", cloneimg );

    char k = waitKey(0);
    if ( k == 27)
      break;
    if ( k == 's' )
      imwrite("results/clean_blemish.jpg", cloneimg);
  }

  return 0;
}
