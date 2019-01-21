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

#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ml.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <string>
#include <stdlib.h>
#include <stdio.h>

// dirent.h is pre-included with *nix like systems
// but not for Windows. So we are trying to include
// this header files based on Operating System
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

using namespace std;
using namespace cv;
using namespace cv::ml;

string colorspace = "y";

void getFileNames(string dirName, vector<string> &imageFnames)
{
  DIR *dir;
  struct dirent *ent;
  int count = 0;

  //image extensions to be found
  string imgExt1 = "png";
  string imgExt2 = "jpg";
  string imgExt3 = "jpeg";
  string imgExt4 = "JPG";
  string imgExt5 = "JPEG";
  string imgExt6 = "PNG";

  vector<string> files;

  if ((dir = opendir (dirName.c_str())) != NULL)
  {
    while ((ent = readdir (dir)) != NULL)
    {
      // Avoiding dummy names which are read by default
      if(strcmp(ent->d_name,".") == 0 | strcmp(ent->d_name, "..") == 0)
      {
        continue;
      }
      string temp_name = ent->d_name;
      files.push_back(temp_name);
    }

    // Sort file names
    std::sort(files.begin(),files.end());
    for(int it=0;it<files.size();it++)
    {
      string path = dirName;
      string fname=files[it];

      if (fname.find(imgExt1, (fname.length() - imgExt1.length())) != std::string::npos)
      {
        path.append(fname);
        imageFnames.push_back(path);
      }
      else if (fname.find(imgExt2, (fname.length() - imgExt2.length())) != std::string::npos)
      {
        path.append(fname);
        imageFnames.push_back(path);
      }
      else if (fname.find(imgExt3, (fname.length() - imgExt3.length())) != std::string::npos)
      {
        path.append(fname);
        imageFnames.push_back(path);
      }
      else if (fname.find(imgExt4, (fname.length() - imgExt4.length())) != std::string::npos)
      {
        path.append(fname);
        imageFnames.push_back(path);
      }
      else if (fname.find(imgExt5, (fname.length() - imgExt5.length())) != std::string::npos)
      {
        path.append(fname);
        imageFnames.push_back(path);
      }
      else if (fname.find(imgExt6, (fname.length() - imgExt6.length())) != std::string::npos)
      {
        path.append(fname);
        imageFnames.push_back(path);
      }
    }
    closedir (dir);
  }
}

void addTrainPixels(vector<string> imageFiles, vector<string> gtFiles, std::vector<Vec3b> &trainPixels, std::vector<int> &trainLabels)
{
  // Clear the arrays
  trainPixels.clear();
  trainLabels.clear();
  Mat image, mask, convertedImage;

  // Read each file and add the skin pixels with labels to data arrays
  for(size_t i = 0; i < imageFiles.size(); i++)
  {
    // Read the image file and mask
    image = imread(imageFiles[i],1);
    mask = imread(gtFiles[i],0);

    // Convert to required color space
    if (colorspace == "l")
    {
      cvtColor(image, convertedImage, COLOR_BGR2Lab);
    }
    else if( colorspace == "y")
    {
      cvtColor(image, convertedImage, COLOR_BGR2YCrCb);
    }
    else
    {
      convertedImage = image.clone();
    }


    // Go over each pixel and assign the label on the basis of ground truth.
    for(int y = 0; y < image.rows; y++)
    {
      //cout << y<<endl;
      for(int x = 0; x < image.cols; x++)
      {
        // In case the ground images is not absolutely binary
        if(mask.at<uchar>(y,x) > 180)
        {
          trainPixels.push_back(convertedImage.at<Vec3b>(y,x));
          // Marking it as skin
          trainLabels.push_back(1);
          // classCounters[1]++;
        }
        else
        {
          trainPixels.push_back(convertedImage.at<Vec3b>(y,x));
          // Marking it as non skin
          trainLabels.push_back(0);
          // classCounters[0]++;
        }
      }
    }
  }
}

void addTestPixels(Mat &testImage, std::vector<Vec3b> &testPixels, std::vector<Point> &testPoints )
{

  Mat convertedImage;

  // Convert the image to required color space
  if (colorspace == "l" )
  {
    cvtColor(testImage, convertedImage, COLOR_BGR2Lab);
    cout << "USING LAB COLOR SPACE" << endl;

  }
  else if( colorspace == "y")
  {
    cvtColor(testImage, convertedImage, COLOR_BGR2YCrCb);
    cout << "USING YCrCb COLOR SPACE" << endl;
  }
  else
  {
    convertedImage = testImage.clone();
    cout << "USING BGR COLOR SPACE" << endl;

  }

  // Go over each pixel and put them in vector
  for(int y = 0; y < testImage.rows; y++)
  {
    for(int x = 0; x < testImage.cols; x++)
    {
      // Push the colours and points of the test image
      testPixels.push_back(convertedImage.at<Vec3b>(y,x));
      // Push the points in another array for displaying later
      testPoints.push_back(Point(x,y));
    }
  }
}

Mat prepareSamples(const std::vector<Vec3b>& pixelArray)
{
  // Reshape the vector to get it in the format of training,
  // that is a single row of entries and of the format 32-bit float
  Mat samples;
  Mat(pixelArray).reshape(1, (int)pixelArray.size()).convertTo(samples, CV_32F, 1/255.0);
  return samples;
}

void testNBC(const Ptr<StatModel>& model, Mat &testImage, Mat& dst)
{

  std::vector<Point> testPoints;
  std::vector<Vec3b> testPixels;

  // Push the pixels of the test image
  addTestPixels(testImage, testPixels, testPoints);

  // Create a matrix for pixel wise prediction
  Mat testSample( 1, 3, CV_32FC1 );

  Mat testColor = prepareSamples(testPixels);
  for(int i = 0; i < testPoints.size(); i++)
  {
    testSample.at<float>(0) = (float)testColor.at<float>(i,0);
    testSample.at<float>(1) = (float)testColor.at<float>(i,1);
    testSample.at<float>(2) = (float)testColor.at<float>(i,2);

    // Predict skin or non skin
    int predY = (int)model->predict( testSample );
    if(predY == 1)
      dst.at<Vec3b>(testPoints[i].y, testPoints[i].x) = testImage.at<Vec3b>(testPoints[i].y, testPoints[i].x);
    else
      dst.at<Vec3b>(testPoints[i].y, testPoints[i].x) = Vec3b(0,0,0);
  }
}

int main( int argc, char** argv)
{

  cout << endl << "NOTE : Please download the data from the link below. " << endl <<
          "http://cs-chan.com/downloads_skin_dataset.html" << endl <<
          "Extract the folder and put it in the cv4faces/data/images/ folder." << endl <<
          "You should get a folder structure like cv4faces/week6/data/images/Face_Dataset/ ." << endl;

  // Names of the two directories containing the training images and ground truth images
  string imagesDirectory1 = "../data/images/Face_Dataset/Pratheepan_Dataset/FacePhoto/";
  string imagesDirectory2 = "../data/images/Face_Dataset/Pratheepan_Dataset/FamilyPhoto/";

  string groundTruthDirectory1 = "../data/images/Face_Dataset/Ground_Truth/GroundT_FacePhoto/";
  string groundTruthDirectory2 = "../data/images/Face_Dataset/Ground_Truth/GroundT_FamilyPhoto/";
  // load the image
  string filename = "../data/images/hillary_clinton.jpg";

  if (argc == 2)
  {
    filename = argv[1];
  }

  if (argc == 3)
  {
    filename = argv[1];
    colorspace = argv[2];
  }

  vector<string> imageFiles, gtFiles;

  // Read images and ground truth images from the two directories
  getFileNames(imagesDirectory1, imageFiles);
  getFileNames(groundTruthDirectory1, gtFiles);
  getFileNames(imagesDirectory2, imageFiles);
  getFileNames(groundTruthDirectory2, gtFiles);

  if( !imageFiles.size() || !gtFiles.size() )
  {
    cout << endl << "Please download and keep the data as mentioned above" << endl;
    return 0;
  }

  cout << "Training . . ." << endl;
  //////////////////////   TRAINING   ////////////////////////////
  std::vector<Vec3b> trainPixels;
  std::vector<int> trainLabels;

  // Adding the training pixels to the data array
  addTrainPixels(imageFiles, gtFiles, trainPixels, trainLabels);

  // Prepare data for training
  Mat trainSamples = prepareSamples(trainPixels);

  // Use the TrainData class to relate the trainLabels and the colours
  Ptr<TrainData> trainData = TrainData::create(trainSamples, ROW_SAMPLE, Mat(trainLabels));

  // train classifier
  Ptr<NormalBayesClassifier> nbc = StatModel::train<NormalBayesClassifier>(trainData);


  cout << "Testing . . ." << endl;
  //////////////////////   TESTING   ////////////////////////////
  Mat testImage = imread(filename);
  // cv::GaussianBlur(testImage, testImage, Size(7, 7), 0);

  Mat imgDst;
  imgDst.create(testImage.rows, testImage.cols, CV_8UC3);

  // Predict on the test image
  testNBC(nbc, testImage, imgDst);

  namedWindow("Skin Detection Demo",WINDOW_AUTOSIZE);
  imshow("Skin Detection Demo", imgDst);

  if(colorspace == "l")
    imwrite("results/skinDetectNBC_LAB.jpg", imgDst);
  else if(colorspace == "y")
    imwrite("results/skinDetectNBC_YCrCb.jpg", imgDst);
  else
    imwrite("results/skinDetectNBC_BGR.jpg", imgDst);


  waitKey(0);
  destroyAllWindows();

  return 0;
}
