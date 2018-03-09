//
//  main_header.h
//  c++_practice
//
//  Created by NICK NORDEN on 2/28/18.
//  Copyright © 2018 NICK NORDEN. All rights reserved.
//

#ifndef main_header_h
#define main_header_h
#endif /* main_header_h */

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/plot.hpp>


#include <iostream>
#include <fstream>
#include <queue>
#include <stdio.h>
#include <math.h>
#include <time.h>

/** Function Headers */
void detectAndDisplay( cv::Mat frame );
void findEyes(cv::Mat, cv::Rect);
bool rectInImage(cv::Rect rect, cv::Mat image);
bool inMat(cv::Point p,int rows,int cols);
cv::Mat matrixMagnitude(const cv::Mat &matX, const cv::Mat &matY);
double computeDynamicThreshold(const cv::Mat &mat, double stdDevFactor);
cv::Point findEyeCenter(cv::Mat face, cv::Rect eye);
void pupilPrint(std::vector<cv::Point3d> pp);
// Algorithm Parameters
const int kFastEyeWidth = 50;
const int kWeightBlurSize = 5;
const float kWeightDivisor = 1.0;
const double kGradientThreshold = 50.0;
const int kEyePercentTop = 25;
const int kEyePercentSide = 13;
const int kEyePercentHeight = 30;
const int kEyePercentWidth = 35;

// Postprocessing
const bool kEnablePostProcess = true;
const float kPostProcessThreshold = 0.90;

/** Global variables */
//-- Note, either copy these two files from opencv/data/haarscascades to your current folder, or change these locations
const cv::String face_cascade_name = "/Users/nicknorden/Downloads/eyeLike-master/res/haarcascade_frontalface_alt.xml";
cv::CascadeClassifier face_cascade;
const std::string face_window_name = "leftPupil";
const cv::RNG rng(12345);
cv::Mat debugImage;
std::vector<cv::Point3d> pointList;
std::chrono::steady_clock::time_point time1;
