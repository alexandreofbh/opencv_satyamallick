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

// returns jpg image paths in given folder
void getFileNames(string dirName, vector<string> &imageFnames)
{
  DIR *dir;
  struct dirent *ent;
  int count = 0;

  //image extensions to be found
  string imgExt = "jpg";

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
        imageFnames.push_back(path);
      }
    }
    closedir (dir);
  }
}

// read images in a folder
// return vector of images and labels
void getDataset(string &pathName, int classVal, vector<Mat> &images, vector<int> &labels) {
  vector<string> imageFiles;
  getFileNames(pathName, imageFiles);
  for (int i = 0; i < imageFiles.size(); i++) {
    Mat im = imread(imageFiles[i]);
    images.push_back(im);
    labels.push_back(classVal);
  }
}

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

// compute HOG features for given images
void computeHOG(vector<vector<float> > &hogFeatures, vector<Mat> &images)
{
  for(int y = 0; y < images.size(); y++)
  {
    vector<float> descriptor;
    hog.compute(images[y], descriptor);
    hogFeatures.push_back(descriptor);
  }
}

// Convert HOG descriptors to format recognized by SVM
void prepareData(vector<vector<float> > &hogFeatures, Mat &data)
{
  int descriptorSize = hogFeatures[0].size();

  for(int i = 0; i < hogFeatures.size(); i++)
    for(int j = 0; j < descriptorSize; j++)
      data.at<float>(i,j) = hogFeatures[i][j];
}

// Initialize SVM with parameters
Ptr<SVM> svmInit(float C, float gamma)
{
  Ptr<SVM> svm = SVM::create();
  svm->setGamma(gamma);
  svm->setC(C);
  svm->setKernel(SVM::LINEAR);
  svm->setType(SVM::C_SVC);
  return svm;
}

// Train SVM on data and labels
void svmTrain(Ptr<SVM> svm, Mat &trainMat, vector<int> &trainLabels)
{
  Ptr<TrainData> td = TrainData::create(trainMat, ROW_SAMPLE, trainLabels);
  svm->train(td);
}

// predict labels for given samples
void svmPredict(Ptr<SVM> svm, Mat &testMat, Mat &testResponse)
{
  svm->predict(testMat, testResponse);
}

// evaluate a model by comparing
// predicted labels and ground truth
void svmEvaluate(Mat &testResponse, vector<int> &testLabels, int &correct, float &error)
{
  for(int i = 0; i < testResponse.rows; i++)
  {
    if(testResponse.at<float>(i,0) == testLabels[i])
      correct = correct + 1;
  }
  error = (testLabels.size() - correct)*100.0/testLabels.size();
}

int main()
{
  // Flags to turn on/off training or testing
  bool trainModel = 1;
  bool testModel = 1;

  // Path to INRIA Person dataset
  string rootDir = "../data/images/INRIAPerson/";
  // string rootDir = "/work/datasets/INRIAPerson/";

  // set Train and Test directory paths
  string trainDir = rootDir + "train_64x128_H96/";
  string testDir = rootDir + "test_64x128_H96/";

  // ================================ Train Model =============================================
  if (trainModel == 1) {
    string trainPosDir = trainDir + "posPatches/";
    string trainNegDir = trainDir + "negPatches/";

    vector<Mat> trainPosImages, trainNegImages;
    vector<int> trainPosLabels, trainNegLabels;

    // Label 1 for positive images and -1 for negative images
    getDataset(trainPosDir, 1, trainPosImages, trainPosLabels);
    getDataset(trainNegDir, -1, trainNegImages, trainNegLabels);

    // Print total number of positive and negative examples
    cout << "positive - " << trainPosImages.size() << " , " << trainPosLabels.size() << endl;
    cout << "negative - " << trainNegImages.size() << " , " << trainNegLabels.size() << endl;

    // Append Positive/Negative Images/Labels for Training
    vector<Mat> trainImages;
    vector<int> trainLabels;
    trainImages = trainPosImages;
    trainImages.insert(trainImages.end(), trainNegImages.begin(), trainNegImages.end());

    trainLabels = trainPosLabels;
    trainLabels.insert(trainLabels.end(), trainNegLabels.begin(), trainNegLabels.end());

    // Compute HOG features for images
    vector<vector<float> > hogTrain;
    computeHOG(hogTrain, trainImages);

    // Convert hog features into data format recognized by SVM
    int descriptorSize = hogTrain[0].size();
    cout << "Descriptor Size : " << descriptorSize << endl;
    Mat trainData(hogTrain.size(), descriptorSize, CV_32FC1);
    prepareData(hogTrain, trainData);

    // Initialize SVM object
    float C = 0.01, gamma = 0;
    Ptr<SVM> svm = svmInit(C, gamma);
    svmTrain(svm, trainData, trainLabels);
    svm->save("../models/pedestrian.yml");
  }

  // ================================ Test Model =============================================
  if (testModel == 1) {
    // Load model from saved file
    Ptr<SVM> svm = ml::SVM::load("../models/pedestrian.yml");

    // We will test our model on positive and negative images separately
    // Read images from Pos and Neg directories
    string testPosDir = testDir + "posPatches/";
    string testNegDir = testDir + "negPatches/";

    vector<Mat> testPosImages, testNegImages;
    vector<int> testPosLabels, testNegLabels;

    // Label 1 for positive images and -1 for negative images
    getDataset(testPosDir, 1, testPosImages, testPosLabels);
    getDataset(testNegDir, -1, testNegImages, testNegLabels);

    // Print total number of positive and negative examples
    cout << "positive - " << testPosImages.size() << " , " << testPosLabels.size() << endl;
    cout << "negative - " << testNegImages.size() << " , " << testNegLabels.size() << endl;

    // =========== Test on Positive Images ===============
    // Compute HOG features for images
    vector<vector<float> > hogPosTest;
    computeHOG(hogPosTest, testPosImages);

    // Convert hog features into data format recognized by SVM
    int descriptorSize = hogPosTest[0].size();
    cout << "Descriptor Size : " << descriptorSize << endl;
    Mat testPosData(hogPosTest.size(), descriptorSize, CV_32FC1);
    prepareData(hogPosTest, testPosData);
    cout << testPosData.rows << " " << testPosData.cols << endl;

    // Run classification on test images
    Mat testPosPredict;
    svmPredict(svm, testPosData, testPosPredict);
    int posCorrect = 0;
    float posError = 0;
    svmEvaluate(testPosPredict, testPosLabels, posCorrect, posError);

    // Calculate True and False Positives
    int tp = posCorrect;
    int fp = testPosLabels.size() - posCorrect;
    cout << "TP: " << tp << " FP: " << fp << " total: " << testPosLabels.size() << " error: " << posError << endl;

    // =========== Test on Negative Images ===============
    // Compute HOG features for images
    vector<vector<float> > hogNegTest;
    computeHOG(hogNegTest, testNegImages);

    // Convert hog features into data format recognized by SVM
    cout << "Descriptor Size : " << descriptorSize << endl;
    Mat testNegData(hogNegTest.size(), descriptorSize, CV_32FC1);
    prepareData(hogNegTest, testNegData);

    // Run classification on test images
    Mat testNegPredict;
    svmPredict(svm, testNegData, testNegPredict);
    int negCorrect = 0;
    float negError = 0;
    svmEvaluate(testNegPredict, testNegLabels, negCorrect, negError);

    // Calculate True and False Negatives
    int tn = negCorrect;
    int fn = testNegLabels.size() - negCorrect;
    cout << "TN: " << tn << " FN: " << fn << " total: " << testNegLabels.size() << " error: " << negError << endl;

    // Calculate Precision and Recall
    float precision = tp * 100.0 / (tp + fp);
    float recall = tp * 100.0 / (tp + fn);
    cout << "Precision: " << precision << " Recall: " << recall << endl;
  }
  return 0;
}
