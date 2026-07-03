#include <opencv2/opencv.hpp>
#include <opencv2/core/utility.hpp> //计时专用
#include <iostream>
#include "imageIO.h"
#include "grade.h"


int main()
{
    std::string test = "../images/10.jpg";
    cv::Mat floatimg =  GetSrc(test);
    std::vector<cv::Mat> SplitChannel = SplitSrc(floatimg);

    //初始光照图InitialMaxLight
    cv::Mat InitialMaxLight = GetInitialMaxLight(SplitChannel);

    cv::Mat MaxLight3Channel,GauseBlurMaxLight;
    //通道数不对怎么除法?借助广播
    cv::merge(std::vector<cv::Mat>{InitialMaxLight, InitialMaxLight, InitialMaxLight}, MaxLight3Channel);
    cv::GaussianBlur(MaxLight3Channel,GauseBlurMaxLight,cv::Size(3,3),0,0);
    cv::Mat StraightLight;
    cv::divide(floatimg, (GauseBlurMaxLight + 0.001), StraightLight);


    //初始光照图的水平和竖直梯度
    auto grad_x = ComputeGradientX(InitialMaxLight);
    auto grad_y = ComputeGradientY(InitialMaxLight);
    
    // cv::Mat norm_gradx, norm_grady;
    // cv::normalize(gradx, norm_gradx, 0.0, 1.0, cv::NORM_MINMAX);
    // cv::normalize(grady, norm_grady, 0.0, 1.0, cv::NORM_MINMAX);

    //权重矩阵weight
    cv::Mat weight = ComputeWeight(grad_x,grad_y);

    // ===================== ADMM 变量初始化 =====================
    // 1. 光照图 T:初始化为初始最大光照图
    cv::Mat T = InitialMaxLight.clone();//深拷贝,防止修改初始光照图(如果直接赋值,那么默认是浅拷贝,会对原图进行修改)

    // 2. 辅助变量 G:初始化为 T 的梯度
    cv::Mat G_x = grad_x.clone();
    cv::Mat G_y = grad_y.clone();

    // 3. 对偶变量 Lambda:初始化为全0
    cv::Mat lambda_x = cv::Mat::zeros(T.size(), CV_32FC1);
    cv::Mat lambda_y = cv::Mat::zeros(T.size(), CV_32FC1);

    float alpha = 0.15f;   // L1正则强度,越大光照图越平滑
    float rho = 1.0f;      // ADMM惩罚系数,影响收敛速度
    int max_iter = 20;     // 迭代次数,一般20次就足够收敛

    // 预生成频域拉普拉斯核(尺寸和rho不变,只生成一次,不用每轮都算)
    cv::Mat freq_kernel = GenerateLaplacianFreqKernel(T.size(), rho);


    cv::TickMeter tm;
    tm.start(); // 开始计时
    std::cout << "开始 ADMM 迭代优化光照图,共 " << max_iter << " 轮..." << std::endl;
    for (int iter = 0; iter < max_iter; iter++)
    {
        // 执行单轮迭代:T → G → Λ 依次更新
        ADMM_Step(InitialMaxLight, weight, alpha, rho, T, G_x, G_y, lambda_x, lambda_y);
        
        // 可选:打印当前迭代的残差,判断收敛情况
        std::cout << "第 " << iter + 1 << " 轮迭代完成" << std::endl;
    }
    tm.stop(); // 停止计时
    std::cout << "光照图优化完成!用时：" <<tm.getTimeSec() << std::endl;

    // ===================== 第六阶段:生成最终增强图像 =====================
    // 1. 把单通道光照图复制成三通道,和原图维度匹配
    cv::Mat T_3ch;
    cv::merge(std::vector<cv::Mat>{T, T, T}, T_3ch);

    // 2. Retinex 公式:增强图 = 原图 / 光照图,加极小值防止除零
    cv::Mat enhanced_img;
    cv::divide(floatimg, T_3ch + 0.001f, enhanced_img);

    // 3. 截断数值到[0,1],避免数值溢出导致显示异常
    cv::threshold(enhanced_img, enhanced_img, 1.0f, 1.0f, cv::THRESH_TRUNC);

    // ===================== 第七阶段:结果显示与对比 =====================
    cv::imshow("1. 原始低照度图", floatimg);
    cv::imshow("2. 初始最大光照图", MaxLight3Channel);
    cv::imshow("4. ADMM优化后光照图", T);
    cv::imshow("5. LIME最终增强结果", enhanced_img);

    cv::waitKey(0);
    
    // cv::imshow("归一化",floatimg);
    // cv::imshow("光照图水平梯度",gradx);
    // cv::imshow("光照图竖直梯度",grady);
    // cv::waitKey(0);

    return 0;
}