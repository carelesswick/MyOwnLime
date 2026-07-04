#include "imageIO.h"
#include <opencv2/opencv.hpp>

//读取原始图片,归一化，并转为浮点型
cv::Mat GetSrc(const std::string& strs){ //AI说这里有问题，如果后续传字面量字符串，可能会报错，建议改成const std::string& strs
    cv::Mat FloatForm;
    cv::Mat Original = cv::imread(strs);
    // cv::resize(Original,Original,cv::Size(1024,768));
    if (Original.empty()) {
        std::cerr << "Error: Could not open or find the image: " << strs << std::endl;
        return cv::Mat(); // 返回空Mat
    }
    Original.convertTo(FloatForm,CV_32FC3,1.0/255.0);
    return FloatForm;

}
//传入归一化和转为浮点型的原始图像，分离通道
std::vector<cv::Mat> SplitSrc(const cv::Mat& src){
    if(src.empty() || src.type() != CV_32FC3){
        std::cerr << "Error: Invalid input image." << std::endl;
        return std::vector<cv::Mat> {};
    }
    std::vector<cv::Mat> channels;
    cv::split(src,channels);
    return channels;
}

//传入分离通道后的数组，获取初始最大光照图Mat
cv::Mat GetInitialMaxLight(const std::vector<cv::Mat>& src){
    cv::Mat MaxLight(src[0].rows,src[0].cols,CV_32FC1,cv::Scalar::all(0));//注意最大光照图是单通道
    for(int i = 0;i < src[0].rows;++i){
        for(int j = 0;j < src[0].cols;++j){
            float B = src[0].at<float>(i,j);
            float G = src[1].at<float>(i,j);
            float R = src[2].at<float>(i,j);
            MaxLight.at<float>(i,j) = std::max({B, G, R}); 
        }
    }
    return MaxLight;
}

////但是直接这样通过最大值求取光照图，效果很差，比如噪声啥的都会被放大，画面会严重失真，所以需要进行光照图优化

