#include "DecisionMaker.h"

void DecisionMaker::ImRecognizer()
{
    cv::Mat plot(cv::Size(250,250), CV_8U, 255);
    for (unsigned int i = 0; i < pupil_data_xy_image.size(); i++)
        plot.at<int>(pupil_data_xy_image[i]) = 0;
    /*
     * need to use erosion filter to enlarge the points and create a better image
     */
    cv::imshow("pupil data", plot);
    cv::waitKey(10000);
    cv::destroyAllWindows();
}


void DecisionMaker::WritePupilDft(std::string dft_file_name)
{
    // Need to log scale all data
    file.open(paths+dft_file_name);
    dtf_mat = cv::Mat((int)x_axis_pupil_points.size(),1,CV_64F,x_axis_pupil_points.data());
    cv::dft(dtf_mat, complexI, cv::DFT_REAL_OUTPUT);
    file<<complexI<<std::endl;
    file.close();
}
 
void DecisionMaker::WritePupilData(std::string pupil_data_file_name)
{
    int n =  0;
    file.open(paths+pupil_data_file_name);
    for (auto i = pupil_points.begin(); i != pupil_points.end(); ++i)
    {
        double ptx = (((*i).x)-xEQ);
        double pty = (((*i).y)-yEQ);
        if(ENABLE_X_AXIS_PUPIL_POINTS)
        {
            x_axis_pupil_points.push_back(cv::Point((*i).x));
        }
        if(ENABLE_IMRECOGNIZER)
        {
            if( ((abs(ptx)-MIN_X_VAL) > 0) && ((abs(pty)-MIN_Y_VAL) > 0) )
            {
            pupil_data_xy_image.push_back(cv::Point(ptx+50,pty+50));
            }
        }
        /*
         * create some test case where it looks at the last value of ptx and pty with the current value.
         * If ( ((ptx_previous - ptx_current)>SOME_VAL_X) && ((pty_previous - pty_current)>SOME_VAL_Y) )
         * then we will consider this a value where the pupil location failed and do not want to consider it
         * this will only be true for eye rolling where people typically obscure their eyes with their eyelid
         */
        file<<ptx<<'\t'<<n++<<'\t'<<pty<<'\t'<<(*i).z<<std::endl;
    }
    file.close();
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
        WritePupilDft(dft_file_name);
    }
    if(ENABLE_IMRECOGNIZER)
    {
        ImRecognizer();
    }
    
}
