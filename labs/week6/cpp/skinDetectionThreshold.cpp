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

int k = 7;

static int lefteye[] = {36, 37, 38, 39, 40, 41};
std::vector<int> lefteye_index (lefteye, lefteye + sizeof(lefteye) / sizeof(lefteye[0]) );
static int righteye[] = {42, 43, 44, 45, 46, 47};
std::vector<int> righteye_index (righteye, righteye + sizeof(righteye) / sizeof(righteye[0]) );
static int mouth[] = {48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59 };
std::vector<int> mouth_index (mouth, mouth + sizeof(mouth) / sizeof(mouth[0]) );
static int leftBrows[] = {17, 18, 19, 20, 21};
std::vector<int> leftBrows_index (leftBrows, leftBrows+ sizeof(leftBrows) / sizeof(leftBrows[0]) );
static int rightBrows[] = {22, 23, 24, 25, 26};
std::vector<int> rightBrows_index (rightBrows, rightBrows + sizeof(rightBrows) / sizeof(rightBrows[0]) );

Mat applyMask(Mat skinImage,std::vector<Point2f> points)
{
    Mat tempmask = Mat::ones(skinImage.rows, skinImage.cols, skinImage.depth());

    std::vector<Point> hullLeftEye;
    for(int i = 0; i < lefteye_index.size(); i++)
    {
        //Take the points just inside of the convex hull
        Point pt( points[lefteye_index[i]].x , points[lefteye_index[i]].y );
        hullLeftEye.push_back(pt);
    }
    fillConvexPoly(tempmask,&hullLeftEye[0], hullLeftEye.size(), Scalar(0,0,0), cv::LINE_AA);

    std::vector<Point> hullRightEye;
    for(int i = 0; i < righteye_index.size(); i++)
    {
        //Take the points just inside of the convex hull
        Point pt( points[righteye_index[i]].x , points[righteye_index[i]].y );
        hullRightEye.push_back(pt);
    }
    fillConvexPoly(tempmask,&hullRightEye[0], hullRightEye.size(), Scalar(0,0,0), cv::LINE_AA);

    std::vector<Point> hullMouth;
    for(int i = 0; i < mouth_index.size(); i++)
    {
        //Take the points just inside of the convex hull
        Point pt( points[mouth_index[i]].x , points[mouth_index[i]].y );
        hullMouth.push_back(pt);
    }
    fillConvexPoly(tempmask,&hullMouth[0], hullMouth.size(), Scalar(0,0,0), cv::LINE_AA);

    std::vector<Point> hullLeftBrow;
    for(int i = 0; i < leftBrows_index.size(); i++)
    {
        //Take the points just inside of the convex hull
        Point pt( points[leftBrows_index[i]].x , points[leftBrows_index[i]].y );
        hullLeftBrow.push_back(pt);
    }
    fillConvexPoly(tempmask,&hullLeftBrow[0], hullLeftBrow.size(), Scalar(0,0,0), cv::LINE_AA);

    std::vector<Point> hullRightBrow;
    for(int i = 0; i < rightBrows_index.size(); i++)
    {
        //Take the points just inside of the convex hull
        Point pt( points[rightBrows_index[i]].x , points[rightBrows_index[i]].y );
        hullRightBrow.push_back(pt);
    }
    fillConvexPoly(tempmask,&hullRightBrow[0], hullRightBrow.size(), Scalar(0,0,0), cv::LINE_AA);

    Mat result;

    cv::bitwise_and(skinImage,skinImage,result,tempmask);
    return result;
}

Mat findSkinHSV(Mat3b meanimg, Mat image)
{
    Mat3b hsv;
    Mat workHSV, skinRegionhsv, skinhsv;

    // Specify the offset around the mean value
    int hsvHueOffset = 10;
    int hsvSatOffset = 50;
    int hsvValOffset = 150;

    // Convert to the HSV color space
    cv::cvtColor(meanimg, hsv, COLOR_BGR2HSV);
    cv::cvtColor(image, workHSV, COLOR_BGR2HSV);

    // Find the range of pixel values to be taken as skin region
    Vec3b hsvPixel(hsv.at<Vec3b>(0,0));
    Scalar minHSV (hsvPixel.val[0] - hsvHueOffset, hsvPixel.val[1] - hsvSatOffset , hsvPixel[2] - hsvValOffset);
    Scalar maxHSV (hsvPixel.val[0] + hsvHueOffset, hsvPixel.val[1] + hsvSatOffset , hsvPixel[2] + hsvValOffset);

    // Apply the range function to find the pixel values in the specific range
    cv::inRange(workHSV, minHSV, maxHSV, skinRegionhsv);

    // Apply Gaussian blur to remove noise
    cv::GaussianBlur(skinRegionhsv, skinRegionhsv, Size(5, 5), 0);

    // Get the kernel for performing morphological opening operation
    Mat kernel=cv::getStructuringElement(cv::MORPH_ELLIPSE, Size(k, k));
    morphologyEx(skinRegionhsv, skinRegionhsv, MORPH_OPEN, kernel, Point(-1, -1), 3);

    // Apply the mask to the image
    cv::bitwise_and(image, image, skinhsv, skinRegionhsv);
    return skinhsv;
}

Mat findSkinYCB(Mat meanimg, Mat image)
{
    Mat3b ycb;
    Mat workYCB, skinRegionycb, skinycb;

    // Specify the offset around the mean value
    int CrOffset = 15;
    int CbOffset = 15;
    int YValOffset = 100;

    // Convert to the YCrCb color space
    cv::cvtColor(meanimg, ycb, COLOR_BGR2YCrCb);
    cv::cvtColor(image, workYCB, COLOR_BGR2YCrCb);

    // Find the range of pixel values to be taken as skin region
    Vec3b ycbPixel(ycb.at<Vec3b>(0,0));
    Scalar minYCB (ycbPixel.val[0] - YValOffset, ycbPixel.val[1] - CrOffset , ycbPixel[2] - CbOffset);
    Scalar maxYCB (ycbPixel.val[0] + YValOffset, ycbPixel.val[1] + CrOffset , ycbPixel[2] + CbOffset);

    // Apply the range function to find the pixel values in the specific range
    cv::inRange(workYCB, minYCB, maxYCB, skinRegionycb);

    // Apply Gaussian blur to remove noise
    cv::GaussianBlur(skinRegionycb, skinRegionycb, Size(5, 5), 0);

    // Get the kernel for performing morphological opening operation
    Mat kernel=cv::getStructuringElement(cv::MORPH_ELLIPSE, Size(k, k));
    morphologyEx(skinRegionycb, skinRegionycb, MORPH_OPEN, kernel, Point(-1, -1), 3);
    // Apply the mask to the image
    cv::bitwise_and(image, image, skinycb, skinRegionycb);
    return skinycb;
}

int main(int argc, char **argv)
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
    Mat image = imread(filename);

    // Find landmarks.
    std::vector<Point2f> landmarks;
    landmarks = getLandmarks(faceDetector, landmarkDetector, image, (float)FACE_DOWNSAMPLE_RATIO);

    // specify the points for taking a square patch
    int ix = landmarks[32].x;
    int fx = landmarks[34].x;
    int iy = landmarks[29].y;
    int fy = landmarks[30].y;

    // Take a patch on the nose
    Mat tempimg;
    cv::Rect selectedRegion = cv::Rect( ix, iy, fx-ix, fy-iy );
    tempimg = image(selectedRegion);
    Mat3b meanimg(1,1,CV_8UC3);

    // Compute the mean image from the patch
    meanimg.at<Vec3b>(0,0)[0] = cv::mean(tempimg)[0];
    meanimg.at<Vec3b>(0,0)[1] = cv::mean(tempimg)[1];
    meanimg.at<Vec3b>(0,0)[2] = cv::mean(tempimg)[2];

    // Find skin using HSV color space
    Mat skinhsv = findSkinHSV(meanimg, image);
    Mat maskedskinhsv = applyMask(skinhsv, landmarks);
    cv::putText(maskedskinhsv, "HSV", Point(50, 50), cv::FONT_HERSHEY_COMPLEX, .9, Scalar(255,255,255), 1, cv::LINE_AA);

    // Find skin using YCrCb color space
    Mat skinycb = findSkinYCB(meanimg, image);
    Mat maskedskinycb = applyMask(skinycb, landmarks);
    cv::putText(maskedskinycb, "YCrCb", Point(50, 50), cv::FONT_HERSHEY_COMPLEX, .9, Scalar(255,255,255), 1, cv::LINE_AA);

    // Display the results and save
    Mat combined;

    cv::hconcat(maskedskinhsv,maskedskinycb,combined);
    cv::imshow("OUTPUT",combined);
    int k = cv::waitKey(0) & 0xFF ;

    cv::imwrite("skinDetectThreshold.jpg",combined);
    return 0;
}
