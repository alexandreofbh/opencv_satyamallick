/*
Copyright 2017 BIG VISION LLC ALL RIGHTS RESERVED

This code is made available to the students of
the online course titled "Computer Vision for Faces"
by Satya Mallick for personal non-commercial use.

Sharing this code is strictly prohibited without written
permission from Big Vision LLC.

For licensing and other inquiries, please email
spmallick@bigvisionllc.com
*/


#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>

#include <dlib/opencv.h>
#include <dlib/image_processing.h>
#include <dlib/image_processing/frontal_face_detector.h>

using namespace dlib;

int main(int argc, char** argv)
{  
  std::cout << "Converts Color Image to Grapy Image." << std::endl;
  if (argc != 3) 
  {
	  std::cout << "Number of arguments are not correct." << std::endl;
    return EXIT_FAILURE;
  }
  
  std::string input_file_name = std::string(argv[1]);
  std::string output_file_name = std::string(argv[2]);
  
  std::cout << "Input File Name = " << input_file_name << std::endl;
  std::cout << "Output File Name = " << output_file_name << std::endl;
  
  
  // Open the Input image and check availability
  cv::Mat color_image = cv::imread(input_file_name, cv::IMREAD_COLOR);

	if (color_image.empty()) // Check for invalid input
	{
		std::cout <<  "Could not open the input image." << std::endl ;
		return EXIT_FAILURE;;
	}
  //Image Properties
  
  std::cout << "Image Properties: Width = " << color_image.size().width <<
      "; Height = " << color_image.size().height << "; No. of Channels = " << color_image.channels() << std::endl;
      
  cv::Mat gray_image;
  // convert color image to gray
	cv::cvtColor(color_image, gray_image, cv::COLOR_BGR2GRAY);  
  
  //Write the Grapy Image in to the file.
  cv::imwrite(output_file_name, gray_image); 
  
  std::cout << "Converted the Color Image to Grapy Image." << std::endl;
      
  return EXIT_SUCCESS;
}

