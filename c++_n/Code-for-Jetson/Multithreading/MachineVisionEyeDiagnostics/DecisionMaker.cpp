#include "DecisionMaker.h"
#include "circleFit.h"

/*
struct {
    bool operator() (cv::Point pt1, cv::Point pt2){
        return (pt1.x < pt2.x);
    }
} PointSorterX;

struct {
    bool operator() (cv::Point pt1, cv::Point pt2){
        return (pt1.y < pt2.y);
    }
} PointSorterY;

struct {
    bool operator() (cv::Point pt1, cv::Point pt2){
        return ((pt1.y < pt2.y)&&(pt1.x<0)&&(pt2.x<0));
    }
} LowerPoint;

struct {
    bool operator() (cv::Point pt1, cv::Point pt2){
        return (pt1.x > pt2.x);
    }
} UpperPointXm;

struct {
    bool operator() (cv::Point pt1, cv::Point pt2){
        return (pt1.x < pt2.x);
    }
} UpperPointX;


struct {
    bool operator() (cv::Point pt1, cv::Point pt2){
        return (pt1.x > pt2.x);
    }
} PointSorterXmx;

struct {
    bool operator() (cv::Point pt1, cv::Point pt2){
        return (pt1.y > pt2.y);
    }
} PointSorterYmx;


struct {
    cv::Point top_left;
    cv::Point top_right;
    cv::Point bottom_left;
    cv::Point bottom_right;
}rectCorners;
*/
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
    
    cv::RotatedRect Ellip = cv::fitEllipse(midpoints);
    cout<<Ellip.size<<Ellip.center<<"";

    
    

    
    
    
}/*
    std::vector<cv::Point> Ydec = pupil_data_xy_image;
    std::vector<cv::Point> Xdec = pupil_data_xy_image;
    
    std::sort(pupil_data_xy_image.begin(),pupil_data_xy_image.end(),PointSorterX);
    
    std::vector<cv::Point>::iterator result = std::min_element(pupil_data_xy_image.begin(),pupil_data_xy_image.end(),LowerPoint);
    for(auto i = pupil_data_xy_image.begin(); i != pupil_data_xy_image.end(); i++){
        if((*result).y == (*i).y){
            rectCorners.bottom_left = (*i);
            break;
        }
        else
            rectCorners.bottom_left = *result;
    }
    result = std::max_element(pupil_data_xy_image.begin(),pupil_data_xy_image.end(),LowerPoint);
    for(auto j = pupil_data_xy_image.begin(); j != pupil_data_xy_image.end(); j++){
        if((*result).y == (*j).y){
            rectCorners.top_left = (*j);
            break;
        }
        else
            rectCorners.top_left = *result;
    }
    
   
    
    std::sort(Ydec.begin(),Ydec.end(),PointSorterYmx);
    std::sort(Xdec.begin(),Xdec.end(),PointSorterXmx);
    for(( std::vector<cv::Point> y,x = Ydec.begin(), Xdec.begin()); y != Ydec.end(); y++){
        if((*y).x) == (*x).x);
    }
        
    std::vector<cv::Point>::iterator result1;
  
    
    for(auto i:pupil_data_xy_image){
        std::cout<< i <<std::endl;
    }
        cout<<rectCorners.bottom_left<<" : ";
        cout<<rectCorners.top_left<<" : ";
        cout<<rectCorners.top_right<<" : ";
        cout<<rectCorners.bottom_right<<" : ";
*/
/*
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

    cv::imwrite("/Users/nicknorden/Desktop/snrProj/output/filtered_pupil_data.png", plot);
    cv::destroyAllWindows();
}
*/

void DecisionMaker::PupilDft(void)
{
    // Need to log scale all data

        //Define the length of the complex arrays
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
                /*
                 *std::cout<<ptx<<','<<x_stack.top()<<'\t'
                 *         <<pty<<','<<y_stack.top()<<std::endl;
                 */
                pupil_data_xy_image.push_back(cv::Point(ptx+100,pty+100));
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

Circle CircleFitByTaubin (Data& data)
/*
 Circle fit to a given set of data points (in 2D)
 
 This is an algebraic fit, due to Taubin, based on the journal article
 
 G. Taubin, "Estimation Of Planar Curves, Surfaces And Nonplanar
 Space Curves Defined By Implicit Equations, With
 Applications To Edge And Range Image Segmentation",
 IEEE Trans. PAMI, Vol. 13, pages 1115-1138, (1991)
 
 Input:  data     - the class of data (contains the given points):
 
 data.n   - the number of data points
 data.X[] - the array of X-coordinates
 data.Y[] - the array of Y-coordinates
 
 Output:
 circle - parameters of the fitting circle:
 
 circle.a - the X-coordinate of the center of the fitting circle
 circle.b - the Y-coordinate of the center of the fitting circle
 circle.r - the radius of the fitting circle
 circle.s - the root mean square error (the estimate of sigma)
 circle.j - the total number of iterations
 
 The method is based on the minimization of the function
 
 sum [(x-a)^2 + (y-b)^2 - R^2]^2
 F = -------------------------------
 sum [(x-a)^2 + (y-b)^2]
 
 This method is more balanced than the simple Kasa fit.
 
 It works well whether data points are sampled along an entire circle or
 along a small arc.
 
 It still has a small bias and its statistical accuracy is slightly
 lower than that of the geometric fit (minimizing geometric distances),
 but slightly higher than that of the very similar Pratt fit.
 Besides, the Taubin fit is slightly simpler than the Pratt fit
 
 It provides a very good initial guess for a subsequent geometric fit.
 
 Nikolai Chernov  (September 2012)
 
 */
{
    int i,iter,IterMAX=99;
    
    reals Xi,Yi,Zi;
    reals Mz,Mxy,Mxx,Myy,Mxz,Myz,Mzz,Cov_xy,Var_z;
    reals A0,A1,A2,A22,A3,A33;
    reals Dy,xnew,x,ynew,y;
    reals DET,Xcenter,Ycenter;
    
    Circle circle;
    
    data.means();   // Compute x- and y- sample means (via a function in the class "data")
    
    //     computing moments
    
    Mxx=Myy=Mxy=Mxz=Myz=Mzz=0.;
    
    for (i=0; i<data.n; i++)
    {
        Xi = data.X[i] - data.meanX;   //  centered x-coordinates
        Yi = data.Y[i] - data.meanY;   //  centered y-coordinates
        Zi = Xi*Xi + Yi*Yi;
        
        Mxy += Xi*Yi;
        Mxx += Xi*Xi;
        Myy += Yi*Yi;
        Mxz += Xi*Zi;
        Myz += Yi*Zi;
        Mzz += Zi*Zi;
    }
    Mxx /= data.n;
    Myy /= data.n;
    Mxy /= data.n;
    Mxz /= data.n;
    Myz /= data.n;
    Mzz /= data.n;
    
    //      computing coefficients of the characteristic polynomial
    
    Mz = Mxx + Myy;
    Cov_xy = Mxx*Myy - Mxy*Mxy;
    Var_z = Mzz - Mz*Mz;
    A3 = Four*Mz;
    A2 = -Three*Mz*Mz - Mzz;
    A1 = Var_z*Mz + Four*Cov_xy*Mz - Mxz*Mxz - Myz*Myz;
    A0 = Mxz*(Mxz*Myy - Myz*Mxy) + Myz*(Myz*Mxx - Mxz*Mxy) - Var_z*Cov_xy;
    A22 = A2 + A2;
    A33 = A3 + A3 + A3;
    
    //    finding the root of the characteristic polynomial
    //    using Newton's method starting at x=0
    //     (it is guaranteed to converge to the right root)
    
    for (x=0.,y=A0,iter=0; iter<IterMAX; iter++)  // usually, 4-6 iterations are enough
    {
        Dy = A1 + x*(A22 + A33*x);
        xnew = x - y/Dy;
        if ((xnew == x)||(!isfinite(xnew))) break;
        ynew = A0 + xnew*(A1 + xnew*(A2 + xnew*A3));
        if (abs(ynew)>=abs(y))  break;
        x = xnew;  y = ynew;
    }
    
    //       computing paramters of the fitting circle
    
    DET = x*x - x*Mz + Cov_xy;
    Xcenter = (Mxz*(Myy - x) - Myz*Mxy)/DET/Two;
    Ycenter = (Myz*(Mxx - x) - Mxz*Mxy)/DET/Two;
    
    //       assembling the output
    
    circle.a = Xcenter + data.meanX;
    circle.b = Ycenter + data.meanY;
    circle.r = sqrt(Xcenter*Xcenter + Ycenter*Ycenter + Mz);
    circle.s = Sigma(data,circle);
    circle.i = 0;
    circle.j = iter;  //  return the number of iterations, too
    
    return circle;
}

reals Sigma (Data& data, Circle& circle)
{
    reals sum=0.,dx,dy;
    
    for (int i=0; i<data.n; i++)
    {
        dx = data.X[i] - circle.a;
        dy = data.Y[i] - circle.b;
        sum += SQR(sqrt(dx*dx+dy*dy) - circle.r);
    }
    return sqrt(sum/data.n);
}

