#include <opencv2/opencv.hpp>
#include <iostream>
#include "imageIO.h"
#include "grade.h"

int main()
{
    std::string test = "../images/3ps.jpg";
    cv::Mat floatimg =  GetSrc(test);
    std::vector<cv::Mat> SplitChannel = SplitSrc(floatimg);
    cv::Mat InitialMaxLight = GetInitialMaxLight(SplitChannel);

    cv::Mat MaxLight3Channel,GauseBlurMaxLight;
    //通道数不对怎么除法？借助广播
    cv::merge(std::vector<cv::Mat>{InitialMaxLight, InitialMaxLight, InitialMaxLight}, MaxLight3Channel);
    cv::GaussianBlur(MaxLight3Channel,GauseBlurMaxLight,cv::Size(3,3),0,0);
    cv::Mat StraightLight;
    cv::divide(floatimg, (GauseBlurMaxLight + 0.001), StraightLight);

    auto gradx = cv::abs(ComputeGradientX(InitialMaxLight));
    auto grady = cv::abs(ComputeGradientY(InitialMaxLight));
    
    cv::Mat norm_gradx, norm_grady;
    cv::normalize(gradx, norm_gradx, 0.0, 1.0, cv::NORM_MINMAX);
    cv::normalize(grady, norm_grady, 0.0, 1.0, cv::NORM_MINMAX);





    
    cv::imshow("归一化",floatimg);
    cv::imshow("初始最大光照",StraightLight);
    cv::imshow("光照图水平梯度",gradx);
    cv::imshow("光照图竖直梯度",grady);

    cv::waitKey(0);

    return 0;
}