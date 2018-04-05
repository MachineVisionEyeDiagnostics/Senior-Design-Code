#ifndef DECISION_MAKER_H
#define DECISION_MAKER_H

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

#define ENABLE_DFT true
#define DISABLE_DFT false
#define ENABLE_IMRECOGNIZER true
#define DISABLE_IMRECOGNIZER false
#define MIN_X_VAL 5
#define MIN_Y_VAL 5

class DecisionMaker {
public:
    void ImRecognizer(void);
    void WritePupilDft(std::string dft_file_name);
    void WritePupilData(std::string pupil_data_file_name);
    void InitData(std::vector<cv::Point3d> pointList, std::string path, std::string pupil_data_file_name, std::string dft_file_name);
private:
	std::string paths;
    std::ofstream file;
    cv::Mat dtf_mat;
    cv::Mat complexI;
    std::vector<cv::Point3d> pupil_points;
    std::vector<cv::Point> x_axis_pupil_points;
    std::vector<cv::Point> pupil_data_xy_image;
    bool ENABLE_X_AXIS_PUPIL_POINTS;
    double xEQ;
    double yEQ;
};

#endif
