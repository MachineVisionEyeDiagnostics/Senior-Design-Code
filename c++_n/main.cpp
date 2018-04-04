#include "main_header.h"

int main( int argc, const char** argv ) {
    cv::Mat frame;
    // Load the cascades
    if(!face_cascade.load(face_cascade_name)){
        printf("--(!)Error loading face cascade, please change face_cascade_name in source code.\n");
        return -1;
    }
    time1= std::chrono::steady_clock::now();
    cv::VideoCapture capture(0);
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
    
    Mved.InitData(pointList, "/Users/nicknorden/Desktop/", "pupil_data_no_filter.txt", ENABLE_DFT, "pupil_dft.txt", ENABLE_IMRECOGNIZER);
    src = cv::imread("/Users/nicknorden/Desktop/filtered_pupil_data.png");
    
    cv::namedWindow( "Erosion", CV_WINDOW_AUTOSIZE );

    Erosion(0,0);
    return 0;
}


void Erosion( int, void* )
{
    cv::Mat element = (cv::Mat_<uchar>(3,3) <<  0,0,1,0,0,
                                                0,1,1,1,0,
                                                1,1,1,1,1,
                                                0,1,1,1,0,
                                                0,0,1,0,0);
    /// Apply the erosion operation
    erode( src, erosion_dst, element );
    cv::imshow( "Erosion", erosion_dst );
    cv::waitKey(0);
}

void DecisionMaker::ImRecognizer()
{
    cv::Mat plot(cv::Size(300,300), CV_8U, 255);
    for (int i = 0; i < pupil_data_xy_image.size(); i++)
        plot.at<int>(pupil_data_xy_image[i]) = 0;
    /*
     * need to use erosion filter to enlarge the points and create a better image
     */
    
    cv::imwrite("/Users/nicknorden/Desktop/filtered_pupil_data.png", plot);
    cv::waitKey(50);
    cv::destroyAllWindows();
}


void DecisionMaker::WritePupilDft(std::string dft_file_name)
{
    
    Mved.file.open(paths+dft_file_name);
    Mved.dtf_mat = cv::Mat((int)Mved.x_axis_pupil_points.size(),1,CV_64F,Mved.x_axis_pupil_points.data());
    cv::dft(Mved.dtf_mat, Mved.complexI, cv::DFT_REAL_OUTPUT);
    Mved.file<<Mved.complexI<<std::endl;
    Mved.file.close();
}
 
void DecisionMaker::WritePupilData(std::string pupil_data_file_name)
{
    int n =  0;
    x_stack.push(999);
    y_stack.push(999);
    std::ofstream filef;
    filef.open("/Users/nicknorden/Desktop/pupil_data_filter.txt");
    Mved.file.open(paths+pupil_data_file_name);
    for (auto i = pupil_points.begin(); i != pupil_points.end(); ++i)
    {
        int ptx = (((*i).x)-xEQ);
        int pty = (((*i).y)-yEQ);
        if(ENABLE_X_AXIS_PUPIL_POINTS)
        {
            x_axis_pupil_points.push_back(cv::Point((*i).x));
        }
        if(ENABLE_IMRECOGNIZER)
        {
            if( (abs(ptx)-MIN_X_VAL > 0) &&
                 ((abs(pty)-MIN_Y_VAL) > 0) &&
                 (abs((abs(ptx)-abs(x_stack.top()))) < MAX_X_JMP) &&
                 (abs((abs(pty)-abs(y_stack.top()))) < MAX_Y_JMP) )
            
            {
                std::cout<<ptx<<','<<x_stack.top()<<'\t'
                         <<pty<<','<<y_stack.top()<<std::endl;
                
                pupil_data_xy_image.push_back(cv::Point(ptx+50,pty+50));
                filef<<ptx<<'\t'<<n++<<'\t'<<pty<<std::endl;
             }
            else
            {
                x_stack.pop();
            }
        }
        Mved.file<<ptx<<'\t'<<n++<<'\t'<<pty<<'\t'<<std::endl;
        y_stack.push(pty);
        x_stack.push(ptx);
        
    }
    Mved.file.close();
    filef.close();
}


void DecisionMaker::InitData(std::vector<cv::Point3d> pointList, std::string path,
                             std::string pupil_data_file_name, bool dft,
                             std::string dft_file_name,
                             bool imrec)
{
    paths = path;
    pupil_points = pointList;
    ENABLE_X_AXIS_PUPIL_POINTS = ENABLE_DFT;
    xEQ = (pupil_points.begin())->x;
    yEQ = (pupil_points.begin())->y;
    WritePupilData(pupil_data_file_name);
    if(ENABLE_X_AXIS_PUPIL_POINTS)
    {
        WritePupilDft(dft_file_name);
    }
    if(ENABLE_IMRECOGNIZER)
    {
        ImRecognizer();
    }
    
}

void detectAndDisplay( cv::Mat frame ) {
    std::vector<cv::Rect> faces;
    std::vector<cv::Mat> rgbChannels(3);
    cv::split(frame, rgbChannels);
    cv::Mat frame_gray = rgbChannels[2];
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(150, 150) );
    for( int i = 0; i < faces.size(); i++ )
        rectangle(debugImage, faces[i], 1234);
            //-- Show what you got
    if (faces.size() > 0)
        findEyes(frame_gray, faces[0]);
}

void findEyes(cv::Mat frame_gray, cv::Rect face) {
    cv::Mat faceROI = frame_gray(face);
    cv::Mat debugFace = faceROI;
    //-- Find general eye region and draw cirlce over left pupil
    int eye_region_width = face.width * (kEyePercentWidth/100.0);
    int eye_region_height = face.width * (kEyePercentHeight/100.0);
    int eye_region_top = face.height * (kEyePercentTop/100.0);
    cv::Rect leftEyeRegion(face.width*(kEyePercentSide/100.0),eye_region_top,eye_region_width,eye_region_height);
    cv::Point leftPupil = findEyeCenter(faceROI,leftEyeRegion);
    leftPupil.x += leftEyeRegion.x;
    leftPupil.y += leftEyeRegion.y;
    #ifdef USE_RIGHT_EYE
    std::cout<<"using right eye"<<std::endl;
    cv::Rect rightEyeRegion(face.width - eye_region_width -face.width *(kEyePercentSide/100.0),
                         eye_region_top,eye_region_width,eye_region_height);
    cv::Point rightPupil = findEyeCenter(faceROI,rightEyeRegion);
    rightPupil.x += rightEyeRegion.x;
    rightPupil.y += rightEyeRegion.y;
    cv::Point cyclops = {(rightPupil.x+leftPupil.x)/2,(rightPupil.y+leftPupil.y)/2};
    auto time2= std::chrono::steady_clock::now();
    std::chrono::duration<double, std::centi> dt = time2-time1;
    pointList.push_back(cv::Point3d(cyclops.x,cyclops.y,dt.count())); //dt.count()
    circle(debugFace, cyclops, 3, cvScalar(255));
    #endif
    pointList.push_back(cv::Point3d(leftPupil.x,leftPupil.y,1));
    circle(debugFace, leftPupil, 3, cvScalar(255));
    //std::cout<<"movement of eye: "<<   <<std::endl;
    imshow(face_window_name, debugFace);
    //time1 = std::chrono::steady_clock::now();
}


cv::Mat floodKillEdges(cv::Mat &mat);

cv::Point unscalePoint(cv::Point p, cv::Rect origSize) {
    float ratio = (((float)kFastEyeWidth)/origSize.width);
    int x = round(p.x / ratio);
    int y = round(p.y / ratio);
    return cv::Point(x,y);
}

void scaleToFastSize(const cv::Mat &src,cv::Mat &dst) {
    cv::resize(src, dst, cv::Size(kFastEyeWidth,(((float)kFastEyeWidth)/src.cols) * src.rows),cv::INTER_LINEAR );
}

cv::Mat computeMatXGradient(const cv::Mat &mat) {
    cv::Mat out(mat.rows,mat.cols,CV_64F);
    
    for (int y = 0; y < mat.rows; ++y) {
        const uchar *Mr = mat.ptr<uchar>(y);
        double *Or = out.ptr<double>(y);
        
        Or[0] = Mr[1] - Mr[0];
        for (int x = 1; x < mat.cols - 1; ++x) {
            Or[x] = (Mr[x+1] - Mr[x-1])/2.0;
        }
        Or[mat.cols-1] = Mr[mat.cols-1] - Mr[mat.cols-2];
    }
    
    return out;
}


void testPossibleCentersFormula(int x, int y, const cv::Mat &weight,double gx, double gy, cv::Mat &out) {
    // for all possible centers
    for (int cy = 0; cy < out.rows; ++cy) {
        double *Or = out.ptr<double>(cy);
        const unsigned char *Wr = weight.ptr<unsigned char>(cy);
        for (int cx = 0; cx < out.cols; ++cx) {
            if (x == cx && y == cy) {
                continue;
            }
            // create a vector from the possible center to the gradient origin
            double dx = x - cx;
            double dy = y - cy;
            // normalize d
            double magnitude = (sqrt((dx * dx) + (dy * dy)));
            dx = dx/magnitude;
            dy = dy/magnitude;
            double dotProduct = abs(dx*gx + dy*gy);
            dotProduct = std::max(0.0,dotProduct);
            // square and multiply by the weight
            Or[cx] += 2*dotProduct * (Wr[cx]/kWeightDivisor);
        }
    }
}

cv::Point findEyeCenter(cv::Mat face, cv::Rect eye) {
    cv::Mat eyeROIUnscaled = face(eye);
    cv::Mat eyeROI;
    
     scaleToFastSize(eyeROIUnscaled, eyeROI);
    // draw eye region
    //rectangle(face,eye,1234);
    //-- Find the gradient
    cv::Mat gradientX = computeMatXGradient(eyeROI);
    cv::Mat gradientY = computeMatXGradient(eyeROI.t()).t();
    //-- Normalize and threshold the gradient
    // compute all the magnitudes
    
    cv::Mat mags = matrixMagnitude(gradientX, gradientY);
    //compute the threshold
    
    double gradientThresh = computeDynamicThreshold(mags, kGradientThreshold);
    //double gradientThresh = kGradientThreshold;
    //double gradientThresh = 0;
    //normalize
    for (int y = 0; y < eyeROI.rows; ++y) {
        double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
        const double *Mr = mags.ptr<double>(y);
        for (int x = 0; x < eyeROI.cols; ++x) {
            double gX = Xr[x], gY = Yr[x];
            double magnitude = Mr[x];
            if (magnitude > gradientThresh) {
                Xr[x] = gX/magnitude;
                Yr[x] = gY/magnitude;
            } else {
                Xr[x] = 0.0;
                Yr[x] = 0.0;
            }
        }
    }
    //cv::String debugWindow = "debugwindow";
    //imshow(debugWindow,gradientX);
    //-- Create a blurred and inverted image for weighting
    cv::Mat weight;
    
    GaussianBlur( eyeROI, weight, cv::Size( kWeightBlurSize, kWeightBlurSize ), 0, 0 );
    for (int y = 0; y < weight.rows; ++y) {
        unsigned char *row = weight.ptr<unsigned char>(y);
        for (int x = 0; x < weight.cols; ++x) {
            row[x] = (255 - row[x]);
        }
    }
    //imshow( "Display window", weight);
    //-- Run the algorithm!
    cv::Mat outSum = cv::Mat::zeros(eyeROI.rows,eyeROI.cols,CV_64F);
    // for each possible gradient location
    // Note: these loops are reversed from the way the paper does them
    // it evaluates every possible center for each gradient location instead of
    // every possible gradient location for every center.
    //printf("Eye Size: %ix%i\n",outSum.cols,outSum.rows);
    for (int y = 0; y < weight.rows; ++y) {
        const double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
       
        for (int x = 0; x < weight.cols; ++x) {
            double gX = Xr[x], gY = Yr[x];
            if (gX == 0.0 && gY == 0.0) {
                continue;
            }
            
            testPossibleCentersFormula(x, y, weight, gX, gY, outSum);
        }
    }
    // scale all the values down, basically averaging them
    double numGradients = (weight.rows*weight.cols);
    cv::Mat out;
    outSum.convertTo(out, CV_32F,1.0/numGradients);
    //imshow(debugWindow,out);
    //-- Find the maximum point
    cv::Point maxP;
    double maxVal;
    
    cv::minMaxLoc(out, NULL,&maxVal,NULL,&maxP);
    //-- Flood fill the edges
    if(kEnablePostProcess) {
        cv::Mat floodClone;
        //double floodThresh = computeDynamicThreshold(out, 1.5);
        double floodThresh = maxVal * kPostProcessThreshold;
        cv::threshold(out, floodClone, floodThresh, 0.0f, cv::THRESH_TOZERO);
        cv::Mat mask = floodKillEdges(floodClone);
        // redo max
        cv::minMaxLoc(out, NULL,&maxVal,NULL,&maxP,mask);
    }
    return unscalePoint(maxP,eye);
}

#pragma mark Postprocessing

bool floodShouldPushPoint(const cv::Point &np, const cv::Mat &mat) {
    return inMat(np, mat.rows, mat.cols);
}

// returns a mask
cv::Mat floodKillEdges(cv::Mat &mat) {
    rectangle(mat,cv::Rect(0,0,mat.cols,mat.rows),255);
    
    cv::Mat mask(mat.rows, mat.cols, CV_8U, 255);
    std::queue<cv::Point> toDo;
    toDo.push(cv::Point(0,0));
    while (!toDo.empty()) {
        cv::Point p = toDo.front();
        toDo.pop();
        if (mat.at<float>(p) == 0.0f) {
            continue;
        }
        // add in every direction
        cv::Point np(p.x + 1, p.y); // right
        if (floodShouldPushPoint(np, mat)) toDo.push(np);
        np.x = p.x - 1; np.y = p.y; // left
        if (floodShouldPushPoint(np, mat)) toDo.push(np);
        np.x = p.x; np.y = p.y + 1; // down
        if (floodShouldPushPoint(np, mat)) toDo.push(np);
        np.x = p.x; np.y = p.y - 1; // up
        if (floodShouldPushPoint(np, mat)) toDo.push(np);
        // kill it
        mat.at<float>(p) = 0.0f;
        mask.at<uchar>(p) = 0;
    }
    return mask;
}


bool rectInImage(cv::Rect rect, cv::Mat image) {
    return rect.x > 0 && rect.y > 0 && rect.x+rect.width < image.cols &&
    rect.y+rect.height < image.rows;
}

bool inMat(cv::Point p,int rows,int cols) {
    return p.x >= 0 && p.x < cols && p.y >= 0 && p.y < rows;
}

cv::Mat matrixMagnitude(const cv::Mat &matX, const cv::Mat &matY) {
    cv::Mat mags(matX.rows,matX.cols,CV_64F);
    for (int y = 0; y < matX.rows; ++y) {
        const double *Xr = matX.ptr<double>(y), *Yr = matY.ptr<double>(y);
        double *Mr = mags.ptr<double>(y);
        for (int x = 0; x < matX.cols; ++x) {
            double gX = Xr[x], gY = Yr[x];
            double magnitude = sqrt((gX * gX) + (gY * gY));
            Mr[x] = magnitude;
        }
    }
    return mags;
}

double computeDynamicThreshold(const cv::Mat &mat, double stdDevFactor) {
    cv::Scalar stdMagnGrad, meanMagnGrad;
    cv::meanStdDev(mat, meanMagnGrad, stdMagnGrad);
    double stdDev = stdMagnGrad[0] / sqrt(mat.rows*mat.cols);
    return stdDevFactor * stdDev + meanMagnGrad[0];
}


