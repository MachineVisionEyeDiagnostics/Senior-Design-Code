#include "DecisionMaker.h"
#include "EyeDetection.h"
#include <thread>

int main( int argc, const char** argv ) {
    //cv::Mat frame;
    DecisionMaker Mved;
    EyeDetection pupil;
    bool face_cascade = pupil.getFace_Cascade();
    //const cv::String face_cascade_name = "/home/nvidia/src/opencv-3.4.0/data/haarcascades/haarcascade_frontalface_alt.xml";
    
    // Load the cascades
    if(!face_cascade){
        printf("--(!)Error loading face cascade, please change face_cascade_name in source code.\n");
        return -1;
    }
    //time1= std::chrono::steady_clock::now();
    // For webcam use 1
    // For Jetson Tx2 use:
    // "nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480,format=(string)I420, framerate=(fraction)60/1 ! nvvidconv flip-method=4 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"
    // For Video feed, use "/home/nvidia/projects/MachineVisionEyeDiagnostics/eye_roll_front_cam3.MOV"
    /*cv::VideoCapture capture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480,format=(string)I420, framerate=(fraction)60/1 ! nvvidconv flip-method=4 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink");
    if(capture.isOpened()){
        while(1){
            //int64 start = cv::getTickCount();
            capture.read(frame);
            // mirror the frame
            //cv::flip(frame, frame, 1);
            //frame.copyTo(debugImage);
            // Apply the classifier to the frame
            if(!frame.empty()){
                detectAndDisplay( frame );
            }
            else{
                printf(" --(!) No captured frame -- Break!");
                break;
            }
            //double fps = cv::getTickFrequency()/(cv::getTickCount()-start);
            //std::cout<<"FPS"<<fps<<std::endl;
            int c = cv::waitKey(10);
            if((char)c == ' '){
                break;
            }
        }
    }*/
    
    pupil.captureVideo();
    //std::thread videothread(&EyeDetection::captureVideo, pupil);
    
    //videothread.join();
    
    
    Mved.InitData(pupil.getPointList(), "/home/nvidia/Desktop/", "pupil_data.txt", "pupil_dft.txt");

    return 0;
}



