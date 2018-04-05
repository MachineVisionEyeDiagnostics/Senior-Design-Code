#ifndef FUNCTIONS_H
#define FUNCTIONS_H

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
#include <chrono>

/** Function Headers */
void captureVideo();
void detectAndDisplay( cv::Mat );
void findEyes( cv::Mat, cv::Rect);
cv::Mat floodKillEdges( cv::Mat &);
bool rectInImage( cv::Rect, cv::Mat );
bool inMat( cv::Point , int, int);
cv::Mat matrixMagnitude( const cv::Mat &, const cv::Mat &);
double computeDynamicThreshold( const cv::Mat &, double );
cv::Point findEyeCenter( cv::Mat , cv::Rect );
void pupilPrint( std::vector<cv::Point3d> );


// Algorithm Parameters
const int kFastEyeWidth = 50;
const int kWeightBlurSize = 5;
const float kWeightDivisor = 1.0;
const double kGradientThreshold = 50.0;
const int kEyePercentTop = 25;
const int kEyePercentSide = 13;
const int kEyePercentHeight = 30;
const int kEyePercentWidth = 35;

/** Global variables */
//-- Note, either copy these two files from opencv/data/haarscascades to your current folder, or change these locations

extern cv::CascadeClassifier face_cascade;
const std::string face_window_name = "leftPupil";
//extern const cv::RNG rng(12345);
extern cv::Mat debugImage;
extern std::vector<cv::Point3d> pointList;

#endif
