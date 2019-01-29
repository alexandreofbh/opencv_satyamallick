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

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include "opencv2/objdetect.hpp"
#include <opencv2/ml.hpp>


#ifdef _WIN32
  #include "dirent.h"
#elif __APPLE__
  #include "TargetConditionals.h"
  #if TARGET_OS_MAC
  #include <dirent.h>
  #else
  #error "Not Mac. Find al alternative to dirent"
  #endif
#elif __linux__
  #include <dirent.h>
#elif __unix__ // all unices not caught above
  #include <dirent.h>
#else
  #error "Unknown compiler"
#endif

using namespace cv::ml;
using namespace cv;
using namespace std;

void getFileNames(string dirName, vector<string> &imageFnames, string imgExt)
{
  DIR *dir;
  struct dirent *ent;
  int count = 0;

  vector<string> files;
  if ((dir = opendir (dirName.c_str())) != NULL) {
    while ((ent = readdir (dir)) != NULL) {
    // Avoiding dummy names which are read by default
    if(strcmp(ent->d_name,".") == 0 | strcmp(ent->d_name, "..") == 0) { continue; }
    string temp_name = ent->d_name;
    files.push_back(temp_name);
    }

    // Sort file names
    std::sort(files.begin(),files.end());
    for(int it=0;it<files.size();it++) {
      string path = dirName;
      string fname = files[it];
      if (fname.find(imgExt, (fname.length() - imgExt.length())) != std::string::npos) {
        path.append(fname);
        // imageFnames.push_back(path);
        imageFnames.push_back(fname);
      }
    }
    closedir (dir);
  }
}

int main() {
  // Initialize HOG
  HOGDescriptor hog(
    Size(64, 128), //winSize
    Size(16, 16),  //blocksize
    Size(8, 8),    //blockStride,
    Size(8, 8),    //cellSize,
        9,     //nbins,
        0,     //derivAperture,
        -1,    //winSigma,
        0,     //histogramNormType,
        0.2,   //L2HysThresh,
        1,     //gammal correction,
        64,    //nlevels=64
        0);    //signedGradient

  Ptr<SVM> svm = ml::SVM::load("../models/pedestrian.yml");
  // get the support vectors
  Mat sv = svm->getSupportVectors();
  // get the decision function
  Mat alpha, svidx;
  double rho = svm->getDecisionFunction(0, alpha, svidx);

  vector<float> svmDetectorTrained;
  svmDetectorTrained.clear();
  svmDetectorTrained.resize(sv.cols + 1);
  for (int j = 0; j < sv.cols; j++) {
    svmDetectorTrained[j] = -sv.at<float>(0, j);
  }
  svmDetectorTrained[sv.cols] = (float)rho;

  // set SVMDetector trained by us in HOG
  hog.setSVMDetector(svmDetectorTrained);

  // OpenCV's HOG based Pedestrian Detector
  HOGDescriptor hogDefault(
    Size(64, 128), //winSize
    Size(16, 16),  //blocksize
    Size(8, 8),    //blockStride,
    Size(8, 8),    //cellSize,
        9,     //nbins,
        0,     //derivAperture,
        -1,    //winSigma,
        0,     //histogramNormType,
        0.2,   //L2HysThresh,
        1,     //gammal correction,
        64,    //nlevels=64
        0);    //signedGradient
  // Set the people detector.
  vector< float > svmDetectorDefault = hog.getDefaultPeopleDetector();
  hogDefault.setSVMDetector(svmDetectorDefault);

  string imageDir = "../data/images/pedestrians/";
  vector<string> imagePaths;
  getFileNames(imageDir, imagePaths, "jpg");

  float finalHeight = 800.0;
  for (int i = 0; i < imagePaths.size(); i++) {
    cout << "processing: " << imagePaths[i] << endl;
    Mat im = imread(imageDir + imagePaths[i], IMREAD_COLOR);

    float finalWidth = (finalHeight * im.cols) / im.rows;
    resize(im, im, Size(finalWidth, finalHeight));

    vector<Rect> bboxes, bboxes2;
    vector<double> weights, weights2;

    float hitThreshold = 1.0;
    Size winStride = Size(8, 8);
    Size padding = Size(32, 32);
    float scale = 1.05;
    float finalThreshold = 2;
    bool useMeanshiftGrouping = 0;
    hog.detectMultiScale(im, bboxes, weights, hitThreshold, winStride, padding,
               scale, finalThreshold, useMeanshiftGrouping);

    hogDefault.detectMultiScale(im, bboxes2, weights2, 0, winStride, padding,
               scale, finalThreshold, useMeanshiftGrouping);

    if (!bboxes.empty()) {
      cout << "Trained Detector :: pedestrians detected: " << bboxes.size() << endl;
      vector< Rect >::const_iterator loc = bboxes.begin();
      vector< Rect >::const_iterator end = bboxes.end();
      for( ; loc != end ; ++loc ) {
        rectangle(im, *loc, Scalar(0, 255, 0), 2);
      }
    }
    if (!bboxes2.empty()) {
      cout << "Default Detector :: pedestrians detected: " << bboxes2.size() << endl;
      vector< Rect >::const_iterator loc = bboxes2.begin();
      vector< Rect >::const_iterator end = bboxes2.end();
      for( ; loc != end ; ++loc ) {
        rectangle(im, *loc, Scalar(0, 0, 255), 2);
      }
    }

    imshow("pedestrians", im);
    imwrite("results/" + imagePaths[i], im);
    waitKey(0);
  }
  return 0;
}
