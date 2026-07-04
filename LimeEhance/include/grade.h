#ifndef GRADE_H
#define GRADE_H

#include <string>
#include <opencv2/core.hpp>


//对初始最大光照图进行梯度的求解
//求梯度分了方向 后续进行封装
cv::Mat ComputeGradientX(const cv::Mat& img);


//一般OpenCV图像是按行连续存储的，外层行、内层列的遍历顺序缓存命中率更高，也是行业通用写法：
cv::Mat ComputeGradientY(const cv::Mat& img);


cv::Mat ComputeWeight(const cv::Mat& gradientx, const cv::Mat& gradienty);



//软阈值函数,但是呢此时的都是for循环，后续可以改成SIMD的优化
cv::Mat SoftThreshold(const cv::Mat& value,const cv::Mat& threshold);


cv::Mat SolveG(
    const cv::Mat& T,
    const cv::Mat& Lambda,
    const cv::Mat& Weight,
    const cv::Mat& GradientT,
    float rho,
    float alpha);


//计算散度
cv::Mat ComputeDivergence(
    const cv::Mat& gx,
    const cv::Mat& gy);


/**
 * @brief 生成离散五点拉普拉斯算子的频域核（对应前向梯度+后向散度的离散格式）
 * @param size  输入图像尺寸
 * @param rho   ADMM惩罚参数
 * @return      频域实值核，CV_32FC1，尺寸与输入一致
 */
cv::Mat GenerateLaplacianFreqKernel(cv::Size size, float rho);

/**
 * @brief ADMM 第一步：求解 T 子问题（FFT 频域求解亥姆霍兹方程）
 * @param T_hat     初始光照图（输入，CV_32FC1）
 * @param G_x       辅助变量G的水平分量（输入，CV_32FC1）
 * @param G_y       辅助变量G的垂直分量（输入，CV_32FC1）
 * @param lambda_x  对偶变量Λ的水平分量（输入，CV_32FC1）
 * @param lambda_y  对偶变量Λ的垂直分量（输入，CV_32FC1）
 * @param rho       ADMM惩罚参数
 * @return          更新后的最优光照图 T（CV_32FC1）
 */
cv::Mat SolveT(
    const cv::Mat& T_hat,
    const cv::Mat& G_x,
    const cv::Mat& G_y,
    const cv::Mat& lambda_x,
    const cv::Mat& lambda_y,
    float rho);


/**
 * @brief 执行单轮 ADMM 迭代（T → G → Λ 依次更新）
 * @param T_hat     初始光照图（固定不变）
 * @param weight    结构感知权重矩阵（固定不变）
 * @param alpha     L1正则系数
 * @param rho       ADMM惩罚参数
 * @param T         输入/输出：光照图，迭代更新
 * @param G_x       输入/输出：G水平分量
 * @param G_y       输入/输出：G垂直分量
 * @param lambda_x  输入/输出：对偶变量水平分量
 * @param lambda_y  输入/输出：对偶变量垂直分量
 */
void ADMM_Step(
    const cv::Mat& T_hat,
    const cv::Mat& weight,
    float alpha,
    float rho,
    cv::Mat& T,
    cv::Mat& G_x,
    cv::Mat& G_y,
    cv::Mat& lambda_x,
    cv::Mat& lambda_y);


#endif // GRADE_H