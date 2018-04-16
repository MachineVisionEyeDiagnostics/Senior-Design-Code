#include "DecisionMaker.h"
#include "EyeDetection.h"
#include <thread>
#include <fftw3.h>


int main( int argc, const char** argv ) {
    //cv::Mat frame;

    DecisionMaker mved;
    EyeDetection pupil;
    bool face_cascade = pupil.getFace_Cascade();
    //const cv::String face_cascade_name = "/home/nvidia/src/opencv-3.4.0/data/haarcascades/haarcascade_frontalface_alt.xml";
    
    // Load the cascades
    if(!face_cascade){
        printf("--(!)Error loading face cascade, please change face_cascade_name in source code.\n");
        return -1;
    }
    
    pupil.captureVideo();
    
    mved.InitData(pupil.getPointList(), "/Users/nicknorde/Desktop", "pupil_data.txt", "pupil_dft.txt");
    return 0;
}




