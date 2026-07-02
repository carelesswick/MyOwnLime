#ifndef IMAGE_IO
#define IMAGE_IO
#include <opencv2/core.hpp>
#include <string>



cv::Mat GetSrc(const std::string& strs);
std::vector<cv::Mat> SplitSrc(const cv::Mat& src);
cv::Mat GetInitialMaxLight(const std::vector<cv::Mat>& src);




#endif