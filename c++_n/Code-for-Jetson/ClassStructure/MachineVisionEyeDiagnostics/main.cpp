#include "DecisionMaker.h"
#include "functions.h"

int main( int argc, const char** argv ) {
    cv::Mat frame;
    DecisionMaker Mved;
    const cv::String face_cascade_name = "/home/nvidia/projects/MachineVisionEyeDiagnostics/haarcascade_frontalface_alt.xml";
    
    // Load the cascades
    if(!face_cascade.load(face_cascade_name)){
        printf("--(!)Error loading face cascade, please change face_cascade_name in source code.\n");
        return -1;
    }
    //time1= std::chrono::steady_clock::now();
    cv::VideoCapture capture("/home/nvidia/projects/MachineVisionEyeDiagnostics/eye_roll_front_cam3.MOV");
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
    }
    
    
    Mved.InitData(pointList, "/home/nvidia/Desktop/", "pupil_data.txt", "pupil_dft.txt");

    return 0;
}



