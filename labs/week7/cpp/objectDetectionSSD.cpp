#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <stdlib.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/dnn.hpp>

using namespace std;

const size_t inWidth = 300;
const size_t inHeight = 300;
const double inScaleFactor = 1.0/127.5;
const float confidenceThreshold = 0.7;
const cv::Scalar meanVal(127.5, 127.5, 127.5);

const std::string configFile = "../models/ssd_mobilenet_v2_coco_2018_03_29.pbtxt";
const std::string weightFile = "../models/ssd_mobilenet_v2_frozen_inference_graph.pb";
std::vector<std::string> classes;

int main(int argc, char ** argv)
{
    const std::string classFile = "../models/coco_class_labels.txt";
    ifstream ifs(classFile.c_str());
    string line;
    while (getline(ifs, line))
    {
        classes.push_back(line);
    }

    cv::VideoCapture source;
    if (argc == 1)
        source.open(0);
    else
        source.open(argv[1]);

    cv::dnn::Net net = cv::dnn::readNetFromTensorflow(weightFile, configFile);

    cv::Mat frame;
    cv::Mat result;
    while(1)
    {
        source >> frame;
        if(frame.empty())
            break;

        cv::Mat inputBlob = cv::dnn::blobFromImage(frame, inScaleFactor, cv::Size(inWidth, inHeight), meanVal, true, false);

        net.setInput(inputBlob);
        cv::Mat detection = net.forward("detection_out");

        cv::Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

        frame.copyTo(result);

        for(int i = 0; i < detectionMat.rows; i++)
        {
            float confidence = detectionMat.at<float>(i, 2);
            int classId = detectionMat.at<float>(i, 1);

            if(confidence > confidenceThreshold)
            {
                int x1 = static_cast<int>(detectionMat.at<float>(i, 3) * result.cols);
                int y1 = static_cast<int>(detectionMat.at<float>(i, 4) * result.rows);
                int x2 = static_cast<int>(detectionMat.at<float>(i, 5) * result.cols);
                int y2 = static_cast<int>(detectionMat.at<float>(i, 6) * result.rows);

                std::string label = cv::format("Object : %s, confidence : %.3f", classes[classId].c_str(), confidence);
                cv::rectangle(result, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(0, 255, 0), 2, 4);
                cv::putText(result, label, cv::Point(x1, y1-10), cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0,0,255), 2, cv::LINE_AA);
            }
        }

        cv::imshow("OpenCV Tensorflow Object Detection Demo", result);

        char c = (char)cv::waitKey(30);
        if (c == 27)
            break;
    }

    return 0;
}
