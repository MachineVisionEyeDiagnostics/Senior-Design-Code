#ifndef DECISION_MAKER_H
#define DECISION_MAKER_H

#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/plot.hpp>


#include <iostream>
#include <fstream>
#include <queue>
#include <stack>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <chrono>
#include <cmath>
#include <fftw3.h>

#define ENABLE_DFT true
#define DISABLE_DFT false
#define ENABLE_IMRECOGNIZER true
#define DISABLE_IMRECOGNIZER false
#define MIN_X_VAL -60
#define MIN_Y_VAL -60
#define ORGN_VAL  1
#define MAX_X_VAL 60
#define MAX_Y_VAL 60
#define MAX_X_JMP 25
#define MAX_Y_JMP 25
#define Real 0
#define Imag 1

class DecisionMaker {
public:
    std::stack<int> x_stack;
    std::stack<int> y_stack;
    void ImRecognizer(void);
    void PupilDft(void);
    void WritePupilData(std::string pupil_data_file_name);
    void InitData(std::vector<cv::Point3d> pointList, std::string path, std::string pupil_data_file_name, std::string dft_file_name);
private:
    cv::Mat erosion_dst;
	std::string paths;
    std::ofstream file;
    cv::Mat dtf_mat;
    cv::Mat complexI;
    std::vector<cv::Point3d> pupil_points;
    std::vector<int> x_axis_pupil_points;
    std::vector<cv::Point> pupil_data_xy_image;
    bool ENABLE_X_AXIS_PUPIL_POINTS;
    double xEQ;
    double yEQ;
};

#endif
