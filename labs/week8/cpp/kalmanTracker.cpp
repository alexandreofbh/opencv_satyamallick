/*
 Copyright 2017 BIG VISION LLC ALL RIGHTS RESERVED

 This program is distributed WITHOUT ANY WARRANTY to the
 students of the online course titled

 "Computer Visionfor Faces" by Satya Mallick

 for personal non-commercial use.

 Sharing this code is strictly prohibited without written
 permission from Big Vision LLC.

 For licensing and other inquiries, please email
 spmallick@bigvisionllc.com

 */

#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;

// Compare rectangles
bool rectAreaComparator(Rect &r1, Rect &r2)
{
  return r1.area() < r2.area();
}

// Colors for display
Scalar blue(255,128,0);
Scalar red(64,64,255);

int main(int, char**)
{

  // Initialize hog descriptor for people detection
  HOGDescriptor hog(
                    Size(64, 128), //winSize
                    Size(16, 16),  //blocksize
                    Size(8, 8),    //blockStride,
                    Size(8, 8),    //cellSize,
                    9,             //nbins,
                    0,             //derivAperture,
                    -1,            //winSigma,
                    0,             //histogramNormType,
                    0.2,           //L2HysThresh,
                    1,             //gammal correction,
                    64,            //nlevels=64
                    0);            //signedGradient
  // Setup people detector.
  vector< float > svmDetector = hog.getDefaultPeopleDetector();
  hog.setSVMDetector(svmDetector);
  float hitThreshold = 0.0;
  Size winStride = Size(8, 8);
  Size padding = Size(32, 32);
  float scale = 1.05;
  float finalThreshold = 1;
  bool useMeanshiftGrouping = 0;
  vector<double> weights;


  // Vector for storing objects rectangles
  std::vector<Rect> objects;

  // Load video
  VideoCapture cap("../data/videos/boy-walking.mp4");

  // Confirm video is open
  if (!cap.isOpened())
  {
    cerr << "Unable to read video" << endl;
    return EXIT_FAILURE;
  }

  // Variables for storing frames
  Mat frame, frameDisplayDetection, frameDisplay, output;

  // Specify Kalman filter type
  int type = CV_32F;

  // Initialize Kalman filter.
  // Internal state has 6 elements (x, y, width, vx, vy, vw)
  // Measurement has 3 elements (x, y, width ).
  // Note: Height = 2 x width, so it is not part of the state
  // or measurement.
  KalmanFilter KF(6, 3, 0, type);

  /*
   Transition matrix is of the form
   [
     1, 0, 0, dt, 0,  0,
     0, 1, 0, 0,  dt, 0,
     0, 0, 1, 0,  0,  dt,
     0, 0, 0, 1,  0,  0,
     0, 0, 0, 0,  1,  0,
     0, 0, 0, dt, 0,  1
   ]

   because

   x = x + vx * dt
   y = y + vy * dt
   w = y + vw * dt

   vx = vx
   vy = vy
   vw = vw

   */
  setIdentity(KF.transitionMatrix);

  /*
   Measurement matrix is of the form
   [
    1, 0, 0, 0, 0,  0,
    0, 1, 0, 0, 0,  0,
    0, 0, 1, 0, 0,  0,
   ]

   because we are detecting only x, y and w.
   These quantities are updated.

  */
  setIdentity(KF.measurementMatrix);

  // Variable to store detected x, y and w
  Mat measurement = Mat::zeros(3, 1, type);

  // Variables to store detected object and tracked object
  Rect objectTracked, objectDetected;

  // Variables to store results of the predict and update ( a.k.a correct step ).
  Mat updatedMeasurement, predictedMeasurement;

  // Variable to indicate measurement was updated.
  bool measurementWasUpdated = false;

  // Timing variable
  double ticks, preTicks;

  // Read frames until object is detected for the first time
  while(cap.read(frame))
  {

    // Detect object
    hog.detectMultiScale(frame, objects, weights, hitThreshold, winStride, padding,
                         scale, finalThreshold, useMeanshiftGrouping);

    // Update timer
    ticks = (double) cv::getTickCount();


    if(objects.size() > 0 )
    {
      // First object detected
      objectDetected = *std::max_element(objects.begin(), objects.end(), rectAreaComparator);

      // Update measurement
      measurement.at<float>(0) = objectDetected.x;
      measurement.at<float>(1) = objectDetected.y;
      measurement.at<float>(2) = objectDetected.width;

      // Update state. Note x, y, w are set to measured values.
      // vx = vy = vw because we have no idea about the velocities yet.
      KF.statePost.at<float>(0) = measurement.at<float>(0);
      KF.statePost.at<float>(1) = measurement.at<float>(1);
      KF.statePost.at<float>(2) = measurement.at<float>(2);
      KF.statePost.at<float>(3) = 0;
      KF.statePost.at<float>(4) = 0;
      KF.statePost.at<float>(5) = 0;

      // Set diagonal values for covariance matrices.
      // processNoiseCov is Q
      setIdentity(KF.processNoiseCov, Scalar::all(1e-2));

      // measurementNoiseCov is R
      setIdentity(KF.measurementNoiseCov, Scalar::all(1e-2));
      break;
    }
  }

  // dt for Transition matrix
  double dt = 0;

  // Random number generator for randomly selecting frames for update
  RNG rng( 0xFFFFFFFF );

  // Loop over all frames
  while(cap.read(frame))
  {
    // Variable for displaying tracking result
    frameDisplay = frame.clone();

    // Variable for displaying detection result
    frameDisplayDetection = frame.clone();

    // Update dt for transition matrix.
    // dt = time elapsed.

    preTicks = ticks;
    ticks = (double) cv::getTickCount();
    dt = (ticks - preTicks) / cv::getTickFrequency();

    KF.transitionMatrix.at<float>(3) = dt;
    KF.transitionMatrix.at<float>(10) = dt;
    KF.transitionMatrix.at<float>(17) = dt;

    // Kalman filter predict step
    predictedMeasurement = KF.predict();


    // Clear objects detected in previous frame.
    objects.clear();

    // Detect objects in current frame
    hog.detectMultiScale(frame, objects, weights, hitThreshold, winStride, padding,
                         scale, finalThreshold, useMeanshiftGrouping);

    // Find largest object
    objectDetected = *std::max_element(objects.begin(), objects.end(), rectAreaComparator);

    // Display detected rectangle
    rectangle(frameDisplayDetection, objectDetected, red, 2, 4);

    // We will update measurements 15% of the time.
    // Frames are randomly chosen.
    bool update = rng.uniform( 0.0, 1.0) < 0.15;

    if (update)
    {
      // Kalman filter update step
      if(objects.size() > 0 )
      {
        // Copy x, y, w from the detected rectangle
        measurement.at<float>(0) = objectDetected.x;
        measurement.at<float>(1) = objectDetected.y;
        measurement.at<float>(2) = objectDetected.width;

        // Perform Kalman update step
        updatedMeasurement = KF.correct(measurement);
        measurementWasUpdated = true;
      }
      else
      {
        // Measurement not updated because no object detected
        measurementWasUpdated = false;
      }

    }
    else
    {
      // Measurement not updated
      measurementWasUpdated = false;

    }

    if(measurementWasUpdated)
    {
      // Use updated measurement if measurement was updated
      objectTracked = Rect(updatedMeasurement.at<float>(0), updatedMeasurement.at<float>(1),updatedMeasurement.at<float>(2),2 * updatedMeasurement.at<float>(2));
    }
    else
    {
      // If measurement was not updated, use predicted values.
      objectTracked = Rect(predictedMeasurement.at<float>(0), predictedMeasurement.at<float>(1),predictedMeasurement.at<float>(2),2 * predictedMeasurement.at<float>(2));
    }

    // Draw tracked object
    rectangle(frameDisplay, objectTracked, blue, 2, 4);

    // Text indicating Tracking or Detection.
    putText(frameDisplay,"Tracking", Point(20,40),FONT_HERSHEY_SIMPLEX, 0.75, blue, 2);
    putText(frameDisplayDetection,"Detection", Point(20,40),FONT_HERSHEY_SIMPLEX, 0.75, red, 2);

    // Concatenate detected result and tracked result vertically
    vconcat(frameDisplayDetection, frameDisplay, output);

    // Display result.
    imshow("object Tracker", output);
    int key = waitKey(5);
    // Break if ESC pressed
    if ( key == 27 )
    {
      break;
    }

  }
  return EXIT_SUCCESS;
}

