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

using namespace cv;
using namespace std;

/*
 (R, G, B) is calssified as skin if
 R > 95 and G > 40 and B > 20 and
 R > G and R > B and |R-G| > 15 and
 max{R, G, B} - min{R, G, B} > 15
*/

int main(int argc, char **argv)
{

  // Start webcam
  VideoCapture cap(0);

  // Check if webcam opens
  if(!cap.isOpened())
  {
    cout << "Error opening video stream or file" << endl;
    return EXIT_FAILURE;
  }

  // Variables used inside the loop.
  Mat image, channels[3], b, g, r, output, skinMask;
  Mat rule1, rule2, rule3, rule4, rule5, rule6, rule7;

  // Window for displaying output
  namedWindow("Skin Detection");

  while(1)
  {
    // Read frame
    cap >> image;

    // Split frame into r, g and b channels
    split(image, channels);
    b = channels[0];
    g = channels[1];
    r = channels[2];

    // Specifying the rules to get binary masks

    rule1 = r > 95;                  // R>95
    rule2 = g > 40;                  // G > 40
    rule3 = b > 20;                  // B > 20
    rule4 = r > g;                   // R > G
    rule5 = r > b;                   // R > B
    rule6 = abs(r-g) > 15;           // |R-G| > 15

    // max{R, G, B} - min{R, G, B} > 15
    rule7 = (max(max(b,g),r) - min(min(b,g),r)) > 15;

    // Apply (AND) all the rules to get the skin mask
    skinMask = rule1.mul(rule2);
    skinMask = skinMask.mul(rule3);
    skinMask = skinMask.mul(rule4);
    skinMask = skinMask.mul(rule5);
    skinMask = skinMask.mul(rule6);
    skinMask = skinMask.mul(rule7);

    // Using the mask to get the skin pixels
    image.copyTo(output, skinMask);

    // Display results
    imshow("Skin Detection",output);
    int key = waitKey(1);
    if ( key == 27)
    {
      break; // ESC Pressed
    }
    output.setTo(Scalar(0,0,0));
  }
  destroyAllWindows();
  return EXIT_SUCCESS;
}




