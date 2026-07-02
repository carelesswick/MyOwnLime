#ifndef GRADE_H
#define GRADE_H

#include <string>
#include <opencv2/core.hpp>



cv::Mat ComputeGradientX(const cv::Mat& img);
cv::Mat ComputeGradientY(const cv::Mat& img);


#endif // GRADE_H