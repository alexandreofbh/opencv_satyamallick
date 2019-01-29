#include <fstream>
#include <sstream>
#include <iostream>

#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>

using namespace cv;
using namespace dnn;
using namespace std;

std::vector<std::string> classes;

int main(int argc, char** argv)
{

    string protoFile = "../models/bvlc_googlenet.prototxt";
    string weightFile = "../models/bvlc_googlenet.caffemodel";

    Mat frame = imread("../data/images/panda.jpg");

    string classFile = "../models/classification_classes_ILSVRC2012.txt";
    ifstream ifs(classFile.c_str());
    string line;
    while (getline(ifs, line))
    {
        classes.push_back(line);
    }

    float scale = 1.0;
    int inHeight = 224;
    int inWidth = 224;
    bool swapRB = false;
    Scalar mean = Scalar(104, 117, 123);

    //! [Read and initialize network]
    Net net = readNetFromCaffe(protoFile, weightFile);

    // Process frames.
    Mat blob;
    //! [Create a 4D blob from a frame]
    blobFromImage(frame, blob, scale, Size(inWidth, inHeight), mean, swapRB, false);

    //! [Set input blob]
    net.setInput(blob);

    //! [Make forward pass]
    Mat prob = net.forward();

    //! [Get a class with a highest score]
    Point classIdPoint;
    double confidence;
    minMaxLoc(prob.reshape(1, 1), 0, &confidence, 0, &classIdPoint);
    int classId = classIdPoint.x;

    // Print predicted class.
    string label = format("Predicted Class : %s, confidence : %.3f", (classes[classId].c_str()), confidence);
    cout << label << endl;
    putText(frame, label, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2, LINE_AA);

    imshow("Classification Output", frame);
    waitKey(0);
    destroyAllWindows();
    return 0;
}
