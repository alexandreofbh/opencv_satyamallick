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
#include "faceBlendCommon.hpp"

using namespace cv::ml;
using namespace cv;
using namespace std;

#define FACE_DOWNSAMPLE_RATIO 1

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

void getFileNames(string dirName, vector<string> &imageFnames)
{
  DIR *dir;
  struct dirent *ent;
  int count = 0;

  //image extensions to be found
  string imgExt = "jpg";

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
      string fname = files[it];

      if (fname.find(imgExt, (fname.length() - imgExt.length())) != std::string::npos)
      {
        path.append(fname);
        imageFnames.push_back(path);
      }
    }
    closedir (dir);
  }
}

void loadTrainTestLabel(string &pathName, vector<Mat> &trainImages, vector<Mat> &testImages, vector<int> &trainLabels, vector<int> &testLabels, int classVal, float testFraction = 0.2)
{
  vector<string> imageFiles;

  getFileNames(pathName, imageFiles);

  int totalImages = imageFiles.size();
  int nTest = (int)(testFraction*totalImages);

  for (int counter = 0; counter < totalImages ; counter++)
  {
    Mat image = imread(imageFiles[counter]);
    if(counter < nTest)
    {
        testImages.push_back(image);
        testLabels.push_back(classVal);
    }
    else
    {
        trainImages.push_back(image);
        trainLabels.push_back(classVal);
    }
  }
}

HOGDescriptor hog(
        Size(96,32), //winSize
        Size(8,8), //blocksize
        Size(8,8), //blockStride,
        Size(4,4), //cellSize,
                9, //nbins,
                0, //derivAper,
                4, //winSigma,
                1, //histogramNormType,
                0.2, //L2HysThresh,
                1,//gammal correction,
                64,//nlevels=64
                1);//signedGradient

void CreateHOG(vector<vector<float> > &HOG, vector<Mat> &Images)
{
  for(int y = 0; y < Images.size(); y++)
  {
    vector<float> descriptors;
    hog.compute(Images[y],descriptors);
    HOG.push_back(descriptors);
  }
}

void ConvertVectortoMatrix(vector<vector<float> > &HOG, Mat &samples)
{
  int descriptor_size = HOG[0].size();

  for(int i = 0; i < HOG.size(); i++)
    for(int j = 0; j < descriptor_size; j++)
      samples.at<float>(i,j) = HOG[i][j];
}

void getSVMParams(Ptr<SVM> svm)
{
  cout << "Kernel type     : " << svm->getKernelType() << endl;
  cout << "Type            : " << svm->getType() << endl;
  cout << "C               : " << svm->getC() << endl;
  cout << "Degree          : " << svm->getDegree() << endl;
  cout << "Nu              : " << svm->getNu() << endl;
  cout << "Gamma           : " << svm->getGamma() << endl;
}

Ptr<SVM> svmInit(float C, float gamma)
{
  Ptr<SVM> svm = SVM::create();
  svm->setGamma(gamma);
  svm->setC(C);
  svm->setKernel(SVM::RBF);
  svm->setType(SVM::C_SVC);

  return svm;
}

void svmTrain(Ptr<SVM> svm, Mat &trainMat, vector<int> &trainLabels)
{
  Ptr<TrainData> td = TrainData::create(trainMat, ROW_SAMPLE, trainLabels);
  svm->train(td);
  svm->save("results/eyeGlassClassifierModel.yml");
}

void svmPredict(Ptr<SVM> svm, Mat &testResponse, Mat &testMat )
{
  svm->predict(testMat, testResponse);
}

void SVMevaluate(Mat &testResponse, float &count, float &accuracy, vector<int> &testLabels)
{
  for(int i = 0; i < testResponse.rows; i++)
  {
    // cout << testResponse.at<float>(i,0) << " " << testLabels[i] << endl;
    if(testResponse.at<float>(i,0) == testLabels[i])
      count = count + 1;
  }
  accuracy = (count/testResponse.rows)*100;
}

Mat getCroppedEyeRegion(dlib::frontal_face_detector faceDetector, dlib::shape_predictor landmarkDetector, Mat image)
{
    // Find landmarks.
    std::vector<Point2f> landmarks;
    landmarks = getLandmarks(faceDetector, landmarkDetector, image, (float)FACE_DOWNSAMPLE_RATIO);

    // specify the points for taking a square patch
    int ix = landmarks[0].x;
    int fx = landmarks[16].x;
    int iy = min(landmarks[24].y, landmarks[19].y);
    int fy = landmarks[29].y;

    // Take a patch over the eye region
    Mat cropped;
    cv::Rect selectedRegion = cv::Rect( ix, iy, fx-ix, fy-iy );
    cropped = image(selectedRegion);
    resize(cropped, cropped, Size(96, 32), 0, 0, INTER_CUBIC);
    return cropped;
}

int main(int argc, char **argv)
{
  vector<Mat> trainImages;
  vector<Mat> testImages;
  vector<int> trainLabels;
  vector<int> testLabels;

  // Path1 is class 0 and Path2 is class 1
  string path1 = "../data/images/glassesDataset/cropped_withoutGlasses2/";
  string path2 = "../data/images/glassesDataset/cropped_withGlasses2/";

  // Get training and testing images for both classes
  loadTrainTestLabel(path1, trainImages, testImages, trainLabels, testLabels, 0);
  loadTrainTestLabel(path2, trainImages, testImages, trainLabels, testLabels, 1);

  ////////// Feature computation for the data  ///////////
  std::vector<std::vector<float> > trainHOG;
  std::vector<std::vector<float> > testHOG;
  CreateHOG(trainHOG, trainImages);
  CreateHOG(testHOG, testImages);

  int descriptor_size = trainHOG[0].size();
  cout << "Descriptor Size : " << descriptor_size << endl;

  // Convert hog data into features recognized by SVM model
  Mat trainMat(trainHOG.size(), descriptor_size, CV_32FC1);
  Mat testMat(testHOG.size(), descriptor_size, CV_32FC1);

  ConvertVectortoMatrix(trainHOG, trainMat);
  ConvertVectortoMatrix(testHOG, testMat);

  float C = 2.5, gamma = 0.02;

  Mat testResponse;
  Ptr<SVM> model = svmInit(C, gamma);

  ///////////  SVM Training  ////////////////
  svmTrain(model, trainMat, trainLabels);

  ///////////  SVM Testing  ////////////////
  svmPredict(model, testResponse, testMat);

  ////////////// Find Accuracy   ///////////
  float count = 0;
  float accuracy = 0 ;
  getSVMParams(model);
  SVMevaluate(testResponse, count, accuracy, testLabels);

  /////////// Testing a single image   ////////////////////
  // Load face detector
  dlib::frontal_face_detector faceDetector = dlib::get_frontal_face_detector();

  // Load landmark detector.
  dlib::shape_predictor landmarkDetector;
  dlib::deserialize("../../common/shape_predictor_68_face_landmarks.dat") >> landmarkDetector;
  string filename = "../data/images/glassesDataset/glasses_3.jpg";
  if (argc == 2)
  {
    filename = argv[1];
  }

  // Read the image
  Mat testImage = imread(filename);

  // Get the eye region
  Mat cropped = getCroppedEyeRegion(faceDetector, landmarkDetector, testImage);

  // Create vector of images for testing
  vector<Mat> testImageArray;
  testImageArray.push_back(cropped);

  // Compute HOG descriptors
  std::vector<std::vector<float> > testHOGArray;
  CreateHOG(testHOGArray, testImageArray);

  // Convert the descriptors to Mat
  Mat testSample(testHOGArray.size(), descriptor_size, CV_32FC1);
  ConvertVectortoMatrix(testHOGArray, testSample);

  // We will load the model again and test the model
  // This is just to explain how to load an SVM model
  // You can use the model directly too

  Ptr<SVM> savedModel = StatModel::load<SVM>("results/eyeGlassClassifierModel.yml");
  Mat pred;
  svmPredict(savedModel, pred, testSample);

  cout << "Prediction : " << pred.at<float>(0,0) << endl;
  imshow("Test Image", testImage);
  imshow("cropped", cropped);
  waitKey(0);
  destroyAllWindows();

  return 0;
}
