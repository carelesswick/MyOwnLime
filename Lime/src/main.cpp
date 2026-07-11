#include <opencv2/opencv.hpp>
#include <opencv2/photo.hpp>       // fastNlMeansDenoisingColored
#include <opencv2/core/utility.hpp> //计时专用
#include <filesystem> // C++17 文件系统库
#include <iostream>
#include "imageIO.h"
#include "grade.h"


int main()
{
    // std::string test = "../images/dark.png";
    std::string test = "../images/10.jpg";
    // std::string input_dir = "../images/";
    std::string output_dir = "../output/";
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


    // ===== 对初始光照图轻度去噪，防止噪声被误判为边缘 =====
    cv::Mat T_hat_denoised;
    cv::GaussianBlur(InitialMaxLight, T_hat_denoised, cv::Size(3, 3), 1.0);

    // 用去噪后的光照图计算权重，梯度更干净
    auto grad_x = ComputeGradientX(T_hat_denoised);
    auto grad_y = ComputeGradientY(T_hat_denoised);
    
    // cv::Mat norm_gradx, norm_grady;
    // cv::normalize(gradx, norm_gradx, 0.0, 1.0, cv::NORM_MINMAX);
    // cv::normalize(grady, norm_grady, 0.0, 1.0, cv::NORM_MINMAX);

    //权重矩阵weight
    cv::Mat weight = ComputeWeight(grad_x,grad_y);

    // ===================== ADMM 变量初始化 =====================
    // 1. 光照图 T:初始化为初始最大光照图
    cv::Mat T = InitialMaxLight.clone();//深拷贝,防止修改初始光照图(如果直接赋值,那么默认是浅拷贝,会对原图进行修改)

    // 2. 辅助变量 G：初始化为全0
    //    注意：不能初始化为 ∇T_hat！否则第一轮 T 子问题会退化回 T=T_hat，完全无平滑
    cv::Mat G_x = cv::Mat::zeros(T.size(), CV_32FC1);
    cv::Mat G_y = cv::Mat::zeros(T.size(), CV_32FC1);

    // 3. 对偶变量 Lambda:初始化为全0
    cv::Mat lambda_x = cv::Mat::zeros(T.size(), CV_32FC1);
    cv::Mat lambda_y = cv::Mat::zeros(T.size(), CV_32FC1);

    // 注：权重已在 ComputeWeight 中归一化到 [0,1]，阈值上限 = α/ρ
    //     ρ 越大 → T 子问题拉普拉斯平滑越强 → 噪声越少
    //     α 越大 → 平坦区域梯度越被压缩 → 越平滑
    // W=1/(|∇T|+1) 使 W∈[0.5,1], α=1 配合 ρ 倍增形成 continuation
    float alpha     = 1.0f;    // L1 正则强度（与参考实现对齐）
    float base_rho  = 1.0f;    // ADMM 惩罚系数初始值
    float rho       = base_rho;
    float rho_max   = 1000.0f; // ρ 上限
    float rho_scale = 2.0f;    // 每轮倍增
    int max_iter    = 10;      // ρ 倍增下 10 轮即可收敛

    std::cout << "开始 ADMM 迭代优化光照图,共 " << max_iter << " 轮..." << std::endl;
    cv::TickMeter tm;
    tm.start();
    for (int iter = 0; iter < max_iter; iter++)
    {
        cv::TickMeter tm_iter;
        tm_iter.start();

        // 执行单轮迭代：T → G → Λ 依次更新
        ADMM_Step(InitialMaxLight, weight, alpha, rho, T, G_x, G_y, lambda_x, lambda_y);

        tm_iter.stop();

        // 递增 rho 加速收敛（论文标准做法）
        rho = std::min(rho * rho_scale, rho_max);

        std::cout << "第 " << iter + 1 << " 轮, rho=" << rho
                  << ", 耗时: " << tm_iter.getTimeMilli() << " ms" << std::endl;
    }
    tm.stop(); // 停止计时
    std::cout << "ADMM 总耗时: " << tm.getTimeSec() << " s" << std::endl;

    cv::TickMeter tm2;
    tm2.start(); //计时

    // ===================== 第六阶段：生成最终增强图像 =====================
    // 1. 光照图 Gamma 校正（LIME 论文公式 10）：T ← T^γ
    //    γ<1 使 T 增大 → 暗区除法放大倍数降低 → 噪声被大幅抑制
    //    例: T=0.05, γ=0.7 → T^0.7=0.123, 放大倍数从 20x 降到 8x
    constexpr float GAMMA = 0.7f;
    cv::Mat T_processed;
    cv::pow(T, GAMMA, T_processed);
    cv::max(T_processed, 0.0001f, T_processed);  // 防除零

    // 2. 扩展到三通道并做 Retinex 增强
    cv::Mat T_3ch;
    cv::merge(std::vector<cv::Mat>{T_processed, T_processed, T_processed}, T_3ch);
    cv::Mat enhanced_img;
    cv::divide(floatimg, T_3ch, enhanced_img);

    // 3. 裁剪到 [0, 1]
    cv::max(enhanced_img, 0.0f, enhanced_img);
    cv::min(enhanced_img, 1.0f, enhanced_img);

    // 4. 非局部均值去噪（BM3D 近似），比双边滤波强得多
    //    转 8-bit → 去噪 → 转回 float
    cv::Mat enhanced_8u;
    enhanced_img.convertTo(enhanced_8u, CV_8UC3, 255.0);
    cv::fastNlMeansDenoisingColored(enhanced_8u, enhanced_8u, 3.0f, 3.0f, 7, 21);
    enhanced_8u.convertTo(enhanced_img, CV_32FC3, 1.0 / 255.0);
    tm2.stop(); // 停止计时

    // ===================== 第七阶段:结果显示与对比 =====================
    // cv::imshow("1. 原始低照度图", floatimg);
    // cv::imshow("2. 初始最大光照图", MaxLight3Channel);
    // cv::imshow("4. ADMM优化后光照图", T);
    // cv::imshow("5. LIME最终增强结果", enhanced_img);
    std::cout<<"去噪用时："<<tm2.getTimeSec()<<std::endl;

    cv::imwrite(output_dir + "lime.jpg", enhanced_8u);

    cv::waitKey(0);
    
    // cv::imshow("归一化",floatimg);
    // cv::imshow("光照图水平梯度",gradx);
    // cv::imshow("光照图竖直梯度",grady);
    // cv::waitKey(0);

    return 0;
}