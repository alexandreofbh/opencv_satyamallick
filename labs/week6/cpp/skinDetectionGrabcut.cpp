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

#include "faceBlendCommon.hpp"

using namespace cv;
using namespace std;
using namespace dlib;

#define FACE_DOWNSAMPLE_RATIO 1

// Draws a polyline on an image (mask). The polyline is specified by a list of indices (pointsIndex) in a vector of points
void drawPolyline(Mat &mask, std::vector<Point2f> points, std::vector<int> pointsIndex, Scalar color, int thickness)
{
  std::vector<Point> linePoints;
  for(int i = 0; i < pointsIndex.size(); i++)
  {
    Point pt( points[pointsIndex[i]].x , points[pointsIndex[i]].y );
    linePoints.push_back(pt);
  }
  polylines(mask, linePoints, false, color, thickness);
}

// Draws a polygon on an image (mask). The polygon is specified by a list of indices (pointsIndex) in a vector of points
void drawPolygon(Mat &mask, std::vector<Point2f> points, std::vector<int> pointsIndex, Scalar color)
{
  std::vector<Point> polygonPoints;
  for(int i = 0; i < pointsIndex.size(); i++)
  {
    Point pt( points[pointsIndex[i]].x , points[pointsIndex[i]].y );
    polygonPoints.push_back(pt);
  }
  fillConvexPoly(mask,&polygonPoints[0], polygonPoints.size(), color);
}


int main( int argc, char** argv)
{

  // Load face detector
  frontal_face_detector faceDetector = get_frontal_face_detector();

  // Load landmark detector.
  shape_predictor landmarkDetector;
  deserialize("../../common/shape_predictor_68_face_landmarks.dat") >> landmarkDetector;

  // load the image
  string filename = "../data/images/hillary_clinton.jpg";

  if (argc == 2)
  {
    filename = argv[1];
  }
  // Read file for applying skin detection.
  Mat img = imread(filename);

  // Find landmarks.
  std::vector<Point2f> landmarks;
  landmarks = getLandmarks(faceDetector, landmarkDetector, img, (float)FACE_DOWNSAMPLE_RATIO);

  // Calculate face mask by finding the convex hull and filling it GC_FGD
  Mat faceMask = Mat::zeros(img.size(), CV_8UC1);
  std::vector<Point2f> hull;
  convexHull(landmarks, hull, false, true);

  // Convert vector of Point2f to vector of Point for fillConvexPoly
  std::vector<Point> hullInt;
  for( int i = 0; i < hull.size(); i++)
  {
    hullInt.push_back(Point(hull[i].x, hull[i].y));
  }
  // Fill face region with foreground indicator GC_FGD
  fillConvexPoly(faceMask, &hullInt[0], hull.size(), Scalar(GC_FGD));
  // imshow("facemask", faceMask*60);

  // Create a mask of possible foreground and possible background regions.
  // This regions will be partial ellipses.
  Mat mask = Mat::zeros(faceMask.size(), faceMask.type());

  // The center of face is defined as the center of the
  // two points connecting the two ends of the jaw line.
  // This point serves as the center of the ellipse.
  Point2f faceCenter = (landmarks[16] + landmarks[0])/2;

  // The two radii of the ellipse will be defined as a factor
  // the radius defined below.
  double radius = norm(faceCenter-landmarks[0]);

  // The angle of the ellipse is determined by the line
  // connecting the corners of the eyes.
  Point2f eyeVector = landmarks[45] - landmarks[36];
  double angle = 180.0 * atan2(eyeVector.y, eyeVector.x) / M_PI;

  // Draw outer elliptical area that indicates probable background (GC_PR_BGD)
  ellipse(mask, Point(faceCenter.x, faceCenter.y), Size (1.3 * radius, 1.2 * radius), angle, 150, 390, Scalar(GC_PR_BGD), -1);

  // Draw a smaller elliptical area that indicates probable foreground region(GC_PR_FGD)
  ellipse(mask, Point(faceCenter.x, faceCenter.y), Size ( 0.9 * radius, radius), angle, 150, 390, Scalar(GC_PR_FGD), -1);
  // imshow("mask1", mask*60);

  // Copy the faceMask over this mask
  faceMask.copyTo(mask,faceMask);
  // imshow("mask after copyting", mask*60 );

  // Define relevant parts of the face for mask calculation.

  // Indices of left eye points in landmarks
  static int leftEye[] = {36, 37, 38, 39, 40, 41};
  std::vector<int> leftEyeIndex (leftEye, leftEye + sizeof(leftEye) / sizeof(leftEye[0]) );

  // Indices of right eye points in landmarks
  static int rightEye[] = {42, 43, 44, 45, 46, 47};
  std::vector<int> rightEyeIndex (rightEye, rightEye + sizeof(rightEye) / sizeof(rightEye[0]) );

  // Indices of mouth points in landmarks
  static int mouth[] = {48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59};
  std::vector<int> mouthIndex (mouth, mouth + sizeof(mouth) / sizeof(mouth[0]) );

  // Indices of left eyebrow points in landmarks
  static int leftEyeBrow[] = {17, 18, 19, 20, 21};
  std::vector<int> leftEyeBrowIndex (leftEyeBrow, leftEyeBrow + sizeof(leftEyeBrow) / sizeof(leftEyeBrow[0]) );

  // Indices of right eyebrow points in landmarks
  static int rightEyeBrow[] = {22, 23, 24, 25, 26};
  std::vector<int> rightEyeBrowIndex (rightEyeBrow, rightEyeBrow + sizeof(rightEyeBrow) / sizeof(rightEyeBrow[0]) );

  Scalar backgroundColor = Scalar(GC_BGD);

  // Remove eyes and mouth region by setting the mask to GC_BGD
  drawPolygon(mask, landmarks, leftEyeIndex, backgroundColor);
  drawPolygon(mask, landmarks, rightEyeIndex, backgroundColor);
  drawPolygon(mask, landmarks, mouthIndex, backgroundColor);

  // Remove eyebrows by setting the mask to GC_BGD
  // The eyebrows are defined by a polyline. So we have to specify a thickness
  // The thickness is chosen as a factor of the distance between the eye corners
  int thickness = 0.1 * norm(landmarks[36] - landmarks[45]);
  drawPolyline(mask, landmarks, leftEyeBrowIndex, backgroundColor, thickness);
  drawPolyline(mask, landmarks, rightEyeBrowIndex, backgroundColor, thickness);

  // Variables for storing background and foreground models
  Mat bgdModel, fgdModel;

  // Apply grabcut
  grabCut(img, mask, Rect(), bgdModel, fgdModel, 1, GC_INIT_WITH_MASK);

  // Final mask is the union of definitely foreground and probably foreground
  mask = ( mask == cv::GC_FGD) | ( mask == cv::GC_PR_FGD);

  // Copy the skin region
  Mat output;
  img.copyTo(output, mask);

  // Display mask
  namedWindow("mask", WINDOW_NORMAL);
  // Since the mask values are between 0 and 3, we need to scale it for display
  imshow("mask", mask*60);

  // Display extracted skin region
  namedWindow("Skin Detection", WINDOW_NORMAL);
  imshow("Skin Detection",output);
  imwrite("results/skinDetectionGrabcut.jpg", output);

  //diameter of the pixel neighbourhood used during filtering
  int dia=15;

  // Larger the value the distant colours will be mixed together
  // to produce areas of semi equal colors
  double sigmaColor=80;

  // Larger the value more the influence of the farther placed pixels
  // as long as their colors are close enough
  double sigmaSpace=80;

  // Apply bilateral filter
  Mat smoothMask;
  bilateralFilter(output, smoothMask, dia, sigmaColor, sigmaSpace);

  Mat smoothImage = img.clone();
  smoothMask.copyTo(smoothImage, mask);

  imshow("smoothImage",smoothImage);
  imwrite("results/skinSmoothing.jpg", smoothImage);

  waitKey(0);

  //Successful exit
  return EXIT_SUCCESS;
}
