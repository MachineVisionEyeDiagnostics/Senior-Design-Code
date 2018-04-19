#include "DecisionMaker.h"
#include "EyeDetection.h"
#include "jetsonGPIO.h"
#include <thread>
#include <fftw3.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <termios.h>
#include <time.h>
#include <sys/time.h>
#include <iostream>
#include <string>
#include <unistd.h>

int getkey() {
    int character;
    struct termios orig_term_attr;
    struct termios new_term_attr;

    /* set the terminal to raw mode */
    tcgetattr(fileno(stdin), &orig_term_attr);
    memcpy(&new_term_attr, &orig_term_attr, sizeof(struct termios));
    new_term_attr.c_lflag &= ~(ECHO|ICANON);
    new_term_attr.c_cc[VTIME] = 0;
    new_term_attr.c_cc[VMIN] = 0;
    tcsetattr(fileno(stdin), TCSANOW, &new_term_attr);

    /* read a character from the stdin stream without blocking */
    /*   returns EOF (-1) if no character is available */
    character = fgetc(stdin);

    /* restore the original terminal attributes */
    tcsetattr(fileno(stdin), TCSANOW, &orig_term_attr);

    return character;
}

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

    
    jetsonTX2GPIONumber hgntest = gpio254; // Output
    jetsonTX2GPIONumber eyerolltest = gpio255; // Output
    // Make them available in user space
    gpioExport(hgntest);
    gpioExport(eyerolltest);
    gpioSetDirection(hgntest, outputPin);
    gpioSetDirection(hgntest, outputPin);
    gpioSetValue(hgntest, on);
    gpioSetValue(eyerolltest, on);
    gpioUnexport(hgntest);
    gpioUnexport(eyerolltest);
    
    pupil.captureVideo();
    
    mved.InitData(pupil.getPointList(), "/home/nvidia/Desktop/", "pupil_data.txt", "pupil_dft.txt");
    
    return 0;
}



