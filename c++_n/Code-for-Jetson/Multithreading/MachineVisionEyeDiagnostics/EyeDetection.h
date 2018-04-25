#ifndef EYEDETECTION_H
#define EYEDETECTION_H

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
#include <thread>

class EyeDetection {

public:
    std::vector<cv::Point3d> detected_centers;
    cv::Mat eyeROI;
    cv::Mat gradientX;
    cv::Mat gradientY;
    double dotproduct;
    cv::Mat mags;
/** Function Headers */
    double computeStepSize(cv::Point& , cv::Mat&, cv::Mat&  ,cv::Point&);
    bool bordersReached(cv::Point c,cv::Mat& );
    double computeObjective(cv::Point c,cv::Mat &, cv::Mat &);
    cv::Point computeGradient(cv::Point c, cv::Mat &, cv::Mat &);
    cv::Point getIntitialCenter(int , int);
    cv::Point centerLoc(int max_trials,int iterations_max,cv::Mat &, cv::Mat &, cv::Mat&);
	void captureVideo();
	void detectAndDisplay( cv::Mat );
	void findEyes( cv::Mat, cv::Rect);
	const std::vector<cv::Point3d> getPointList() const;
	bool getFace_Cascade();
	cv::Point unscalePoint(cv::Point, cv::Rect);
	void scaleToFastSize(const cv::Mat &,cv::Mat &);
	cv::Mat computeMatXGradient(const cv::Mat &);
	void testPossibleCentersFormula(int, int, const cv::Mat &,double, double, cv::Mat &);
	cv::Point findEyeCenter( cv::Mat , cv::Rect );
	bool floodShouldPushPoint(const cv::Point &, const cv::Mat &);
	cv::Mat floodKillEdges( cv::Mat &);
	bool rectInImage( cv::Rect, cv::Mat );
	bool inMat( cv::Point , int, int);
	cv::Mat matrixMagnitude( const cv::Mat &, const cv::Mat &);
	double computeDynamicThreshold( const cv::Mat &, double );

//void pupilPrint( std::vector<cv::Point3d> );

private:
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

	cv::CascadeClassifier face_cascade;
	const std::string face_window_name = "leftPupil";
//extern const cv::RNG rng(12345);
//cv::Mat debugImage;
	std::vector<cv::Point3d> pointList;
};

#endif
