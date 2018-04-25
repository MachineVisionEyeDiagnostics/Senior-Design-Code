#include "DecisionMaker.h"
#define PI 3.14159265

void::DecisionMaker::ImRecognizer()

{
    cv::Mat keypoints;
    std::vector<cv::Point> kpvec;
    cv::Mat circlePoints;
    cv::RotatedRect square = cv::minAreaRect(pupil_data_xy_image);
    cv::boxPoints(square, keypoints);
    for(int i=0;i<4;i++){
        kpvec.push_back(cv::Point(keypoints.at<float>(i,0),keypoints.at<float>(i,1)));
    }
    std::vector<cv::Point> midpoints;
    
    cv::Point TopMidPoint = cv::Point((kpvec[0].x+kpvec[1].x)/2,(kpvec[0].y+kpvec[1].y)/2);
    cv::Point RightMidPoint = cv::Point((kpvec[1].x+kpvec[2].x)/2,(kpvec[1].y+kpvec[2].y)/2);
    cv::Point BottomMidPoint = cv::Point((kpvec[2].x+kpvec[3].x)/2,(kpvec[2].y+kpvec[3].y)/2);
    cv::Point LeftMidPoint = cv::Point((kpvec[3].x+kpvec[0].x)/2,(kpvec[3].y+kpvec[0].y)/2);
    midpoints.push_back(TopMidPoint);
    midpoints.push_back(RightMidPoint);
    midpoints.push_back(BottomMidPoint);
    midpoints.push_back(LeftMidPoint);
    midpoints.push_back(LeftMidPoint);
    
    float a = sqrt(((TopMidPoint.x-square.center.x)*(TopMidPoint.x-square.center.x) +
               (TopMidPoint.y-square.center.y)*(TopMidPoint.y-square.center.y)));
    float b = sqrt(((RightMidPoint.x-square.center.x)*(RightMidPoint.x-square.center.x) +
               (RightMidPoint.y-square.center.y)*(RightMidPoint.y-square.center.y)));
    float A,B,C,D,E,F;
    float sin_theta = sin(square.angle*PI/180);
    float cos_theta = cos(square.angle*PI/180);
    //a = .8*a;
    b = .8*b;
    if(a>b){
        A = a*a*sin_theta*sin_theta+b*b*cos_theta*cos_theta;
        B = 2*(b*b-a*a)*sin_theta*cos_theta;
        C = a*a*cos_theta*cos_theta+b*b*sin_theta*sin_theta;
        D = -2*A*square.center.x-B*square.center.y;
        E = -B*square.center.x-2*C*square.center.y;
        F =A*square.center.x*square.center.x+B*square.center.x*square.center.y+C*square.center.y*square.center.y-a*a*b*b;
    }
    else{
        cv::swap(a,b);
        A = a*a*sin_theta*sin_theta+b*b*cos_theta*cos_theta;
        B = 2*(b*b-a*a)*sin_theta*cos_theta;
        C = a*a*cos_theta*cos_theta+b*b*sin_theta*sin_theta;
        D = -2*A*square.center.x-B*square.center.y;
        E = -B*square.center.x-2*C*square.center.y;
        F =A*square.center.x*square.center.x+B*square.center.x*square.center.y+C*square.center.y*square.center.y-a*a*b*b;
        
    }
    double inside = 0.0;
    int outside = 0;

    for(auto i:pupil_data_xy_image){
        float point = A*i.x*i.x+B*i.x*i.y+C*i.y*i.y+D*i.x+E*i.y+F;
            if(point<=0)
                inside++;
            else
                outside++;
        }
    
    for(auto i: midpoints)
        std::cout<<i<<"::";
    std::cout<<"::"<<A<<','<<B<<','<<C<<','<<D<<','<<E<<','<<F<<"::";
    double result = inside/pupil_data_xy_image.size();
    if(result > 0.6)
        std::cout<<'\n'<<"eye roll passed: "<<result<<'\n';
    else
        std::cout<<'\n'<<"eye roll failed: "<<result<<'\n';

    
}
void DecisionMaker::PupilDft(void){

    int len = (int)x_axis_pupil_points.size();
    //Input arrays
    fftw_complex x[len];
    //Output array
    fftw_complex y[len];
    //Fill the first array with some data
    for (int i=0; i<len; i++)
    {
        x[i][Real]=x_axis_pupil_points[i];
        x[i][Imag]=0;
    }
    //Plant the FFT and execute it
    fftw_plan plan= fftw_plan_dft_1d(len, x, y, FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(plan);
    //Do some cleaning
    fftw_destroy_plan(plan);
    fftw_cleanup();
    //Display the results
    std::cout << "FFT = " <<std::endl;
    double mag[len/2+1];
    double sum_of_harmonics = 0;
    for (int i=1; i<len/2; i++){
        mag[i] = sqrt(y[i][0]*y[i][0]+y[i][1]*y[i][1]);
        sum_of_harmonics += mag[i+1];
    }
    std::cout<<'\n'<<"harmonic percentage: "<<sum_of_harmonics/mag[1]<<'\n';
}


 
void DecisionMaker::WritePupilData(std::string pupil_data_file_name)
{
    int n =  0;
    x_stack.push(999);
    y_stack.push(999);
    std::ofstream filef;
    filef.open("/Users/nicknorden/Desktop/snrProj/output/pupil_data_filter.txt");
    file.open(paths+pupil_data_file_name);
    for (auto i = pupil_points.begin(); i != pupil_points.end(); ++i)
    {
        int ptx = (((*i).x)-xEQ);
        int pty = (((*i).y)-yEQ);
        if(ENABLE_X_AXIS_PUPIL_POINTS)
        {
            x_axis_pupil_points.push_back((*i).x);
        }
        if(ENABLE_IMRECOGNIZER)
        {
            if(  ((abs(ptx) - ORGN_VAL) > 0) &&
               ((abs(pty) - ORGN_VAL) > 0) &&
               (abs((abs(ptx)-abs(x_stack.top()))) < MAX_X_JMP) &&
               (abs((abs(pty)-abs(y_stack.top()))) < MAX_Y_JMP) &&
               (abs(ptx) < MAX_X_VAL) &&
               (abs(pty) < MAX_Y_VAL) &&
               (abs(ptx) > MIN_X_VAL) &&
               (abs(pty) > MIN_Y_VAL)  )
                
            {
                pupil_data_xy_image.push_back(cv::Point(ptx+100,pty+100));
                filef<<ptx+100<<'\t'<<n++<<'\t'<<pty+100<<std::endl;
            }
            else
            {
                x_stack.pop();
            }
        }
        file<<ptx<<'\t'<<n++<<'\t'<<pty<<std::endl;
        y_stack.push(pty);
        x_stack.push(ptx);
        
    }
    file.close();
    filef.close();
}


void DecisionMaker::InitData(std::vector<cv::Point3d> pointList, std::string path, std::string pupil_data_file_name, std::string dft_file_name)
{
    paths = path;
    pupil_points = pointList;
    ENABLE_X_AXIS_PUPIL_POINTS = ENABLE_DFT;
    xEQ = (pupil_points.begin())->x;
    yEQ = (pupil_points.begin())->y;
    WritePupilData(pupil_data_file_name);
    if(ENABLE_X_AXIS_PUPIL_POINTS)
    {
        PupilDft();
    }
    if(ENABLE_IMRECOGNIZER)
    {
        ImRecognizer();
    }
    
}

