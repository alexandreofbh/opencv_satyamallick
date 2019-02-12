#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <dlib/opencv.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing.h>

using namespace std;
using namespace cv;

int main(int argc, char ** argv)
{
  string filename = "../data/videos/face1.mp4";
  if (argc == 2)
  {
    filename = argv[1];
  }
  VideoCapture cap(filename);

  // Check if webcam opens
  if(!cap.isOpened())
  {
    cout << "Error opening video stream or file" << endl;
    return EXIT_FAILURE;
  }

  // Declare the variables for using later
  Mat frame, roiObject, hsvObject, mask, histObject;
  Mat hsv, backProjectImage, frameClone;

  // Read a frame and find the face region using dlib
  for(int i = 0; i< 10; i++)
    cap >> frame;

  dlib::frontal_face_detector faceDetector = dlib::get_frontal_face_detector();
  dlib::cv_image<dlib::bgr_pixel> dlibIm(frame);

  // Detect faces in the image
  std::vector<dlib::rectangle> faceRects = faceDetector(dlibIm);
  Rect currWindow;
  if(faceRects.size() > 0)
  {
    dlib::rectangle bbox = faceRects[0];
    // modify the dlib rect to opencv rect
    currWindow = Rect(
                    (long)bbox.left(),
                    (long)bbox.top() ,
                    (long)bbox.width(),
                    (long)bbox.height()
                    );
  }
  else
  {
    cout << "Face Not Found " << endl;
    cap.release();
    return 0;
  }

  // Create and image with the face region
  frame(currWindow).copyTo(roiObject);
  cvtColor(roiObject, hsvObject, COLOR_BGR2HSV);

  // Get the mask for calculating histogram of the object and also remove noise
  inRange(hsvObject, Scalar(0, 50, 50), Scalar(180, 256, 256), mask);

  // Split the image into channels for finding the histogram
  vector<Mat> channels(3);
  split(hsvObject, channels);
  imshow("Mask", mask);
  imshow("Object", roiObject);

  // Initialize parameters for histogram
  int histSize = 180;
  float range[] = { 0, 179 };
  const float *ranges[] = { range };

  // Find the histogram and normalize it to have values between 0 to 255
  calcHist( &channels[0], 1, 0, mask, histObject, 1, &histSize, ranges, true, false );
  normalize(histObject, histObject, 0, 255, NORM_MINMAX);

  while(1)
  {
    // Read frame
    cap >> frame;
    if( frame.empty() )
      break;

    // Convert to hsv color space
    cvtColor(frame, hsv, COLOR_BGR2HSV);
    split(hsv, channels);

    // find the back projected image with the histogram obtained earlier
    calcBackProject(&channels[0], 1, 0, histObject, backProjectImage, ranges);
    imshow("Back Projected Image", backProjectImage);

    // Compute the new window using CAM shift in the present frame
    RotatedRect rotatedWindow = CamShift(backProjectImage, currWindow, TermCriteria( TermCriteria::EPS | TermCriteria::COUNT, 10, 1 ));

    // Get the rotatedWindow vertices
    Point2f rotatedWindowVertices[4];
    rotatedWindow.points(rotatedWindowVertices);

    frameClone = frame.clone();

    // Display the current window used for mean shift
    rectangle(frameClone, Point(currWindow.x, currWindow.y), Point(currWindow.x + currWindow.width, currWindow.y + currWindow.height), Scalar(255, 0, 0), 2, LINE_AA);

    // Display the rotated rectangle with the orientation information
    for (int i = 0; i < 4; i++)
      line(frameClone, rotatedWindowVertices[i], rotatedWindowVertices[(i+1)%4], Scalar(0,255,0), 2, LINE_AA);
    imshow("CAMShift Object Tracking Demo", frameClone);

    int k = cv::waitKey(10);
    if (k == 27)
      break;
  }

  cap.release();
  destroyAllWindows();
  return 0;
}
