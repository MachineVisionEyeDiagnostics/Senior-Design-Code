#include "EyeDetection.h"

/*For webcam use 1
 *For Jetson Tx2 use:
 *"nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480,format=(string)I420, framerate=(fraction)60/1 ! nvvidconv flip-method=4 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink"
 *For Video feed, use "/home/nvidia/projects/MachineVisionEyeDiagnostics/eye_roll_front_cam3.MOV"
 *cv::VideoCapture capture("nvcamerasrc ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480,format=(string)I420, framerate=(fraction)60/1 ! nvvidconv flip-method=4 ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink");
*/

const bool kEnablePostProcess = true;
const float kPostProcessThreshold = 0.90;

void EyeDetection::captureVideo() {
	cv::Mat frame;
	cv::VideoCapture capture(0);
    if(capture.isOpened()){
        capture.read(frame);
        while(1){
            double begin = (double)cv::getTickCount();
            std::thread test(&cv::VideoCapture::read, capture, frame);
            //capture.read(frame);
           
            if(!frame.empty()){
                detectAndDisplay( frame );
            }
            else{
                printf(" --(!) No captured frame -- Break!");
                break;
            }
            test.join();
            double fps = cv::getTickFrequency()/((double)(cv::getTickCount()-begin));
            std::cout<<"AVERGAGE FPS "<<fps<<std::endl;
            int c = cv::waitKey(10);
            if((char)c == ' '){
                break;
            }
        }
    }
}

void EyeDetection::detectAndDisplay( cv::Mat frame ) {
	cv::Mat debugImage;
    std::vector<cv::Rect> faces;
    std::vector<cv::Mat> rgbChannels(3);
    cv::split(frame, rgbChannels);
    cv::Mat frame_gray = rgbChannels[2];
    face_cascade.detectMultiScale( frame_gray, faces, 1.1, 2, 0|CV_HAAR_SCALE_IMAGE|CV_HAAR_FIND_BIGGEST_OBJECT, cv::Size(150, 150) );
    for( unsigned int i = 0; i < faces.size(); i++ )
        rectangle(debugImage, faces[i], 1234);
            //-- Show what you got
    if (faces.size() > 0)
        findEyes(frame_gray, faces[0]);
}

void EyeDetection::findEyes(cv::Mat frame_gray, cv::Rect face) {
	//std::vector<cv::Point3d> pointList;
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
    /*
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
     */
    pointList.push_back(cv::Point3d(leftPupil.x,leftPupil.y,1));
    circle(debugFace, leftPupil, 3, cvScalar(255));
    //std::cout<<"movement of eye: "<<   <<std::endl;
    imshow(face_window_name, debugFace);
    //time1 = std::chrono::steady_clock::now();
}


const std::vector<cv::Point3d> EyeDetection::getPointList() const {
	return pointList;
}


bool EyeDetection::getFace_Cascade() {
	bool exist = face_cascade.load("/Users/nicknorden/Downloads/eyeLike-master/res/haarcascade_frontalface_alt.xml");
	return exist;
}


cv::Point EyeDetection::unscalePoint(cv::Point p, cv::Rect origSize) {
    float ratio = (((float)kFastEyeWidth)/origSize.width);
    int x = round(p.x / ratio);
    int y = round(p.y / ratio);
    return cv::Point(x,y);
}

void EyeDetection::scaleToFastSize(const cv::Mat &src,cv::Mat &dst) {
    cv::resize(src, dst, cv::Size(kFastEyeWidth,(((float)kFastEyeWidth)/src.cols) * src.rows),cv::INTER_LINEAR );
}

cv::Mat EyeDetection::computeMatXGradient(const cv::Mat &mat) {
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


void EyeDetection::testPossibleCentersFormula(int x, int y, const cv::Mat &weight,double gx, double gy, cv::Mat &out) {
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

cv::Point EyeDetection::findEyeCenter(cv::Mat face, cv::Rect eye) {
    cv::Mat eyeROIUnscaled = face(eye);
    cv::Mat eyeROI;
    
     scaleToFastSize(eyeROIUnscaled, eyeROI);
    // draw eye region
    //rectangle(face,eye,1234);
    //-- Find the gradient
    
    /*
    cv::Mat gradientX = computeMatXGradient(eyeROI);
    cv::Mat gradientY = computeMatXGradient(eyeROI.t()).t();
    //-- Normalize and threshold the gradient
    // compute all the magnitudes
    
    cv::Mat mags = matrixMagnitude(gradientX, gradientY);
    */
    
    cv::Mat gradientX;
    cv::Sobel(eyeROI, gradientX,CV_64F, 1, 0, 3);
    cv::Mat gradientY;
    cv::Sobel(eyeROI, gradientY, CV_64F, 0, 1, 3);
    //-- Normalize and threshold the gradient
    // compute all the magnitudes
    cv::Mat mags(gradientX.size(),gradientX.depth());
    cv::magnitude(gradientX, gradientY, mags);
    cv::sort(mags, mags, CV_SORT_EVERY_ROW|CV_SORT_DESCENDING);

    //compute the threshold
    
   // double gradientThresh = computeDynamicThreshold(mags, kGradientThreshold);
     double gradientThresh = cv::mean(mags)[0];
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
    //cv::namedWindow(debugWindow);
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
    /*
    
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
     */
    computeSetSize(mags);
    
    
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


double EyeDetection::computeSetSize(cv::Mat &mags){
    const double threshold = 0.0001;
    std::vector<cv::Point3d> centers;
    const double sigma = 0.1;
    const double beta = 0.5;
    double a = 1;
    //const int   max_iter = 30;
    for(int i = 0; i < mags.rows; i++){
        for( int j = 0; j < mags.cols; j++){
            int cx = i;
            int cy = j;
            double obj = objectiveFunction(cx,cy);
            cv::Point g = gradientFunction(cx, cy);
            double g1 = g.x;
            double g2 = g.y;
            //int k = 0;
            while( sqrt(g1*g1+g2*g2) > threshold ){
                double newobj = objectiveFunction(cx-a*g1,cy-a*g1);
                while(((newobj-obj)/a) > (sigma*-1*g1*g1+g2+g2)){
                    a = a*beta;
                    newobj = objectiveFunction(cx-a*g1,cx-a*g2);
                }
            }
        }
        centers.push_back(cv::Point3d(newobj,cx,cy));
    }

}
double EyeDetection::objectiveFunction(int cx, int cy){
   // cv::Mat out;
    int n=0;
    for (int y = 0; y < mags.rows; ++y) {
        const double *Xq = gradientX.ptr<double>(y), *Yq = gradientY.ptr<double>(y);
        //const double *Or = out.ptr<double>(y);
        for (int x = 0; x < mags.cols; ++x) {
            double gx = Xq[x], gy = Yq[x];
            
            if (gx == 0.0 && gy == 0.0 && x == cx && y == cy) {
                continue;
            }
            double dx = x - cx;
            double dy = y - cy;
            // normalize d
            double magnitude = (sqrt((dx * dx) + (dy * dy)));
            dx = dx/magnitude;
            dy = dy/magnitude;
            double dottProduct = abs(dx*gx + dy*gy);
            dottProduct = std::max(0.0,dottProduct);
            dottProduct += 2*dottProduct; //(Wr[cx]/kWeightDivisor);
            n++;
        }
    }
    dottproduct = dottProduct/n;
    return (dottProduct);
}

cv::Point EyeDetection::gradientFunction(int cx, int cy){
    for (int y = 0; y < mags.rows; ++y) {
        const double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
        for (int x = 0; x < mags.cols; ++x) {
            double gx = Xr[x], gy = Yr[x];
            if (gx == 0.0 && gy == 0.0 && x == cx && y == cy) {
                continue;
            }
            double dx = x - cx;
            double dy = y - cy;
            double n = (sqrt((dx * dx) + (dy * dy)));
            cv::Mat g = cv::Mat::zeros(2,1);
            g(0,0) = gx;
            g(1,0) = gy;
            cv::Mat c = cv::Mat::zeros(2,1);
            c(0,0) = cx;
            c(1,0) = cy;
            double e = (x-cx)*g;
            objdx += ((x-cx)*e*e -g.at<double>(0,0)*e*n*n)/(n*n*n*n);
            objdy += ((x-cx)*e*e -g.at<double>(0,0)*e*n*n)/(n*n*n*n);
        }
    }
    return cv::Point(objdx*2/(y+1), objdy*2/(y+1));
}


/*
 % Armijo stepsize rule parameters
 sigma = .1;
 beta = .5;
 obj=func(x);
 g=grad(x);
 k=0;                                  % k = # iterations
 nf=1;                    % nf = # function eval.
 
 % Begin method
 while  norm(g) > 1e-6
 d = -g;                   % steepest descent direction
 a = 1;
 newobj = func(x + a*d);
 nf = nf+1;
 while (newobj-obj)/a > sigma*g'*d
 a = a*beta;
 newobj = func(x + a*d);
 nf = nf+1;
 end
 if (mod(k,100)==1) fprintf('%5.0f %5.0f %12.5e \n',k,nf,obj); end
 x = x + a*d;
 obj=newobj;
 g=grad(x);
 k = k + 1;
 end
 
 % Output x and k
 x, k
 
 */


//#pragma mark Postprocessing

bool EyeDetection::floodShouldPushPoint(const cv::Point &np, const cv::Mat &mat) {
    return inMat(np, mat.rows, mat.cols);
}


// returns a mask
cv::Mat EyeDetection::floodKillEdges(cv::Mat &mat) {
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


bool EyeDetection::rectInImage(cv::Rect rect, cv::Mat image) {
    return rect.x > 0 && rect.y > 0 && rect.x+rect.width < image.cols &&
    rect.y+rect.height < image.rows;
}

bool EyeDetection::inMat(cv::Point p,int rows,int cols) {
    return p.x >= 0 && p.x < cols && p.y >= 0 && p.y < rows;
}

cv::Mat EyeDetection::matrixMagnitude(const cv::Mat &matX, const cv::Mat &matY) {
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

double EyeDetection::computeDynamicThreshold(const cv::Mat &mat, double stdDevFactor) {
    cv::Scalar stdMagnGrad, meanMagnGrad;
    cv::meanStdDev(mat, meanMagnGrad, stdMagnGrad);
    //double stdDev = stdMagnGrad[0] / sqrt(mat.rows*mat.cols);
    //return stdDevFactor * stdDev + meanMagnGrad[0];
    return meanMagnGrad[0];
}
