#include "DecisionMaker.h"

void DecisionMaker::ImRecognizer()
{
    int j = (int)pupil_data_xy_image.size();
    cv::Mat plot(cv::Size(j+200,j+100), CV_8U, 255);
    for (int i = 0; i < j; i++)
        plot.at<int>(pupil_data_xy_image[i]) = 0;
    
    cv::Mat element = (cv::Mat_<uchar>(5,5) <<  0,0,1,1,0,0,
                       0,1,1,1,1,0,
                       1,1,1,1,1,1,
                       0,1,1,1,1,0,
                       0,0,1,1,0,0);
    /// Apply the erosion operation
    cv::erode( plot, erosion_dst, element );
    cv::erode( erosion_dst, plot, element);
    cv::erode( plot, erosion_dst, element );
    cv::erode( erosion_dst, plot, element );
    cv::erode( plot, erosion_dst, element );
    
    cv::imwrite("/Users/nicknorden/Desktop/filtered_pupil_data.png", erosion_dst);
    cv::waitKey(50);
    cv::destroyAllWindows();
}


void DecisionMaker::PupilDft(void)
{
    // Need to log scale all data

        //Define the length of the complex arrays
    int len = x_axis_pupil_points.size();
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
    for (int i=0; i<len; i++){
        if(y[i][Imag]<0)
            std::cout << y[i][Real]<< " - " << abs(y[i][Imag]) << "i" << std::endl;
        else
            std::cout << y[i][Real]<< " + " << y[i][Imag] << "i" << std::endl;
    }
}
 
void DecisionMaker::WritePupilData(std::string pupil_data_file_name)
{
    int n =  0;
    x_stack.push(999);
    y_stack.push(999);
    std::ofstream filef;
    filef.open("/Users/nicknorden/Desktop/pupil_data_filter.txt");
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
                /*
                 *std::cout<<ptx<<','<<x_stack.top()<<'\t'
                 *         <<pty<<','<<y_stack.top()<<std::endl;
                 */
                pupil_data_xy_image.push_back(cv::Point(ptx+40,pty+40));
                filef<<ptx<<'\t'<<n++<<'\t'<<pty<<std::endl;
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
