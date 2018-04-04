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

#define ENABLE_DFT true
#define DISABLE_DFT false
#define ENABLE_IMRECOGNIZER true
#define DISABLE_IMRECOGNIZER false
#define MIN_X_VAL 5
#define MIN_Y_VAL 5

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

class DecisionMaker {
public:
    std::string paths;
    std::ofstream file;
    void ImRecognizer(void);
    void WritePupilDft(std::string dft_file_name);
    void WritePupilData(std::string pupil_data_file_name);
    void InitData(std::vector<cv::Point3d> pointList, std::string path, std::string pupil_data_file_name, bool enable_dft, std::string dft_file_name, bool enable_imrec);
private:
    cv::Mat dtf_mat;
    cv::Mat complexI;
    std::vector<cv::Point3d> pupil_points;
    std::vector<cv::Point> x_axis_pupil_points;
    std::vector<cv::Point> pupil_data_xy_image;
    bool ENABLE_X_AXIS_PUPIL_POINTS;
    double xEQ;
    double yEQ;
};

DecisionMaker Mved;
std::chrono::steady_clock::time_point time1;


/*
 
 void pupilPrint(std::vector<cv::Point3d> pp){
 std::ofstream myFile;
 myFile.open("/Users/nicknorden/Desktop/pupil_data.txt");
 int n = 0;
 auto xEQ = (pp.begin())->x;
 auto yEQ = (pp.begin())->y;
 for (auto i = pp.begin(); i != pp.end(); ++i){
 split_x.push_back(cv::Point((*i).x));
 double ptx = (((*i).x)-xEQ);
 double pty= (((*i).y)-yEQ);
 myFile<<ptx<<'\t'<<n++<<'\t'<<pty<<std::endl;
 }
 cv::Mat dtf_mat = cv::Mat((int)split_x.size(),1,CV_64F,split_x.data());
 
 //cv::Mat planes[] = {cv::Mat_<float>(dtf_mat), cv::Mat::zeros(dtf_mat.size(), CV_32F)};
 cv::Mat complexI;    //Complex plane to contain the DFT coefficients {[0]-Real,[1]-Img}
 //cv::merge(planes, 2, complexI);
 cv::dft(dtf_mat, complexI, cv::DFT_REAL_OUTPUT);
 std::cout<<complexI<<std::endl;
 myFile.close();
 
 }
*/
