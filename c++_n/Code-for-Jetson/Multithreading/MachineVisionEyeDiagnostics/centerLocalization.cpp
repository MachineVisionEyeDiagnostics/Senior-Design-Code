#include "EyeDetection.h"


bool comp(const cv::Point3d& a, const cv::Point3d& b)
{
    return (a.z < b.z);
}


cv::Point centerLoc(G,X,max_trials,iterations_max,cv::Mat& eyeROI)
{
    double convergence_threshold = 10e-2;
    std::vector<cv::Point3d> detected_centers;
    for(int i = 0; i < max_trials; i++)
    {
        cv::Point c = getInitialCenter();
        for(int j = 0; j< iterations_max; j++)
        {
            cv::Point c_old = c;
            cv::Point g = computeGradient(c, G, X);
            double s = computeStepSize();
            c.x = c.x-s*g.x;
            c.y = c.y-s*g.y;
            if(!bordersReached(c,eyeROI))
                break;
            if(sqrt((c.x-c_old.x)*(c.x-c_old.x)-(c.y-c_old.y)*(c.y-c_old.y)) <= convergence_threshold)
            {
                double j = computeObjective(c,G,X);
                detected_centers.push_back(cv::Point3d(c.x,c.y,j));
            }
        }
    }
    std::sort(detected_centers.begin(),detected_centers.end(),comp);
    return(cv::Point(detected_centers[1].x,detected_centers[1].y));
}

cv::Point getIntitialCenter(int W,int H)
{
    cv::Point c;
    //use W, H, to set up bounds for rand
    c.x = rand();
    c.y = rand();
    return c;
}

cv::Point computeGradient(cv::Point c, cv::Mat&gradientX, cv::Mat&gradientY)
{
    double partial_x, partial_y, e_i, n, dx, dy = 0.0;
    cv::Point ctr = c;
    for(int y = 0; y < gradientX.rows; ++y)
    {
        const double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
        for (int x = 0; x < gradientX.cols; ++x)
        {
            double gx = Xr[x], gy = Yr[x];
            if (gx == 0.0 && gy == 0.0 && x == ctr.x && y == ctr.y)
            {
                continue;
            }
            dx = x - ctr.x;
            dy = y - ctr.y;
            n = (sqrt((dx * dx) + (dy * dy)));
            cv::Point g = cv::Point(gx,gy);
            e_i = (x-ctr.x)*g.x+(y-ctr.y)*g.y;
            partial_x += ( (x-ctr.x)*e_i*e_i -g.x*e_i*n*n )/(n*n*n*n);
            partial_y += ( (x-ctr.y)*e_i*e_i -g.y*e_i*n*n )/(n*n*n*n);
        }
    }
    return cv::Point(partial_x,partial_y);
}

double computeStepSize(void)
{
    double a = 0.5;
    static int i = 0;
    return a/pow(a,i);
}


bool bordersReached(cv::Point c,cv::Mat& eyeROI )
{
    return (c.x > 0 && c.y > 0 && c.x < eyeROI.cols &&
    c.y < eyeROI.rows);
}

double computeObjective(cv::Point c,cv::Mat&gradientX, cv::Mat&gradientY)
{
    double dotproduct = 0.0;
    cv::Point ctr = c;
    int n=0;
    for (int y = 0; y < gradientX.rows; ++y) {
        const double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
        for (int x = 0; x < gradientX.cols; ++x) {
            double gx = Xr[x], gy = Yr[x];
            
            if (gx == 0.0 && gy == 0.0 && x == ctr.x && y == ctr.y) {
                continue;
            }
            double dx = x - ctr.x;
            double dy = y - ctr.y;
            double magnitude = (sqrt((dx * dx) + (dy * dy)));
            dx = dx/magnitude;
            dy = dy/magnitude;
            dotproduct = abs(dx*gx + dy*gy);
            dotproduct = std::max(0.0,dotproduct);
            dotproduct += 2*dotproduct;
            n++;
        }
    }
    dotproduct = dotproduct/n;
    return (dotproduct);
}








