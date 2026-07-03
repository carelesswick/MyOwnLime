#include "grade.h"
#include <opencv2/opencv.hpp>


//对初始最大光照图进行梯度的求解
//求梯度分了方向 后续进行封装
cv::Mat ComputeGradientX(const cv::Mat& img){
    
    cv::Mat GradientX(img.size(), CV_32FC1, cv::Scalar(0));
    if (img.empty())
    {
        std::cerr << "Error: Input image is empty!" << std::endl;
        return {};
    }

    if (img.type() != CV_32FC1)
    {
        std::cerr << "Error: GradientX only supports CV_32FC1." << std::endl;
        return {};
    }

    for (int i = 0;i < img.rows;++i){
        for (int j = 0; j < img.cols - 1;++j){
            GradientX.at<float>(i,j) = img.at<float>(i,j+1) - img.at<float>(i,j);
        }//at带有性能开销 后续需要改进
    }
    return GradientX;

}

//一般OpenCV图像是按行连续存储的，外层行、内层列的遍历顺序缓存命中率更高，也是行业通用写法：
cv::Mat ComputeGradientY(const cv::Mat& img){
    
    cv::Mat GradientY(img.size(), CV_32FC1, cv::Scalar(0));
    if (img.empty() || img.type() != CV_32FC1)
    {
        std::cerr << "Error: Input image is empty or not CV_32FC1" << std::endl;
        return {};
    }
    for( int i = 0;i <img.rows - 1;++i){
        for(int j = 0;j < img.cols;++j){
            GradientY.at<float>(i,j) = img.at<float>(i+1,j) - img.at<float>(i,j);
        }
    }
    return GradientY;
}

cv::Mat ComputeWeight(const cv::Mat& gradientx, const cv::Mat& gradienty)
{
    // 与参考 LIME 实现一致：denominator = |gradient| + 1（不是 +0.001）
    // 这样 W ∈ [0.5, 1]，阈值 αW/ρ 变化平缓，配合 ρ 倍增形成 continuation
    cv::Mat absGradx = cv::abs(gradientx);
    cv::Mat absGrady = cv::abs(gradienty);

    cv::Mat Wx, Wy;
    cv::divide(1.0f, absGradx + 1.0f, Wx);
    cv::divide(1.0f, absGrady + 1.0f, Wy);

    // 不加高斯模糊（与参考实现一致）
    cv::Mat weight;
    cv::vconcat(Wx, Wy, weight);

    return weight;
}


//软阈值函数,但是呢此时的都是for循环，后续可以改成SIMD的优化
cv::Mat SoftThreshold(const cv::Mat& value,const cv::Mat& threshold)
{
    CV_Assert(value.type() == CV_32FC1);
    CV_Assert(threshold.type() == CV_32FC1);
    CV_Assert(value.size() == threshold.size());

    cv::Mat output(value.size(), CV_32FC1);

    // 遍历每个像素
    for(int i = 0; i < value.rows; ++i)
    {
        for(int j = 0; j < value.cols; ++j)
        {
            float d = value.at<float>(i,j);
            float lambda = threshold.at<float>(i,j);

            float sign = 0.f;
            if(d > 0)
                sign = 1.f;
            else if(d < 0)
                sign = -1.f;

            output.at<float>(i,j) =
                sign * std::max(std::abs(d) - lambda, 0.0f);
        }
    }

    return output;
}

cv::Mat SolveG(
    const cv::Mat& T,
    const cv::Mat& Lambda,
    const cv::Mat& Weight,
    float rho,
    float alpha)
{
    // 计算 ∇T,因为之前的梯度是分开算的，因此在这里呢需要把水平方向和竖直方向的梯度拼接起来
    cv::Mat GradientX = ComputeGradientX(T);
    cv::Mat GradientY = ComputeGradientY(T);
    cv::Mat GradientT;
    cv::vconcat(GradientX, GradientY, GradientT);

    // d = ∇T - Λ/ρ
    cv::Mat d = GradientT - Lambda / rho;

    // λ = αW/ρ
    cv::Mat threshold = alpha * Weight / rho;

    // Soft Threshold
    return SoftThreshold(d, threshold);
}


//计算散度
cv::Mat ComputeDivergence(
    const cv::Mat& gx,
    const cv::Mat& gy)
{
    CV_Assert(gx.type()==CV_32FC1);
    CV_Assert(gy.type()==CV_32FC1);
    CV_Assert(gx.size()==gy.size());

    cv::Mat div(gx.size(),CV_32FC1,cv::Scalar(0));

    for(int i=0;i<gx.rows;i++)
    {
        for(int j=0;j<gx.cols;j++)
        {
            float dx=gx.at<float>(i,j);
            float dy=gy.at<float>(i,j);

            if(j>0)
                dx-=gx.at<float>(i,j-1);

            if(i>0)
                dy-=gy.at<float>(i-1,j);

            div.at<float>(i,j)=dx+dy;
        }
    }

    return div;
}

/**
 * @brief 生成离散五点拉普拉斯算子的频域核（对应前向梯度+后向散度的离散格式）
 * @param size  输入图像尺寸
 * @param rho   ADMM惩罚参数
 * @return      频域实值核，CV_32FC1，尺寸与输入一致
 */
cv::Mat GenerateLaplacianFreqKernel(cv::Size size, float rho)
{
    int H = size.height;
    int W = size.width;
    cv::Mat kernel(H, W, CV_32FC1);

    for (int v = 0; v < H; v++)    // v: 行方向频率索引
    {
        for (int u = 0; u < W; u++) // u: 列方向频率索引
        {
            // 离散五点拉普拉斯的标准频域响应公式
            float cos_u = cosf(2.0f * (float)M_PI * u / W);
            float cos_v = cosf(2.0f * (float)M_PI * v / H);
            float laplacian_freq = 2.0f * cos_u + 2.0f * cos_v - 4.0f;

            // 对应算子 (2I - ρ·∇²) 的频域响应
            kernel.at<float>(v, u) = 2.0f - rho * laplacian_freq;
        }
    }
    return kernel;
}

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
    float rho)
{
    // ========== 输入合法性检查 ==========
    CV_Assert(!T_hat.empty() && T_hat.type() == CV_32FC1);
    CV_Assert(G_x.size() == T_hat.size() && G_x.type() == CV_32FC1);
    CV_Assert(G_y.size() == T_hat.size() && G_y.type() == CV_32FC1);
    CV_Assert(lambda_x.size() == T_hat.size() && lambda_x.type() == CV_32FC1);
    CV_Assert(lambda_y.size() == T_hat.size() && lambda_y.type() == CV_32FC1);

    // ========== 步骤1：计算 b = G + Λ/ρ ==========
    cv::Mat b_x = G_x + lambda_x / rho;
    cv::Mat b_y = G_y + lambda_y / rho;

    // ========== 步骤2：计算 b 的散度 div(b) ==========
    cv::Mat div_b = ComputeDivergence(b_x, b_y);

    // ========== 步骤3：构造右端项 rhs ==========
    cv::Mat rhs = 2.0f * T_hat - rho * div_b;

    // ========== 步骤4：生成频域实值核（纯实数，不需要转复数） ==========
    cv::Mat freq_kernel = GenerateLaplacianFreqKernel(T_hat.size(), rho);

    // ========== 步骤5：rhs 做正 FFT，转到频域 ==========
    cv::Mat rhs_fft;
    cv::dft(rhs, rhs_fft, cv::DFT_COMPLEX_OUTPUT); // 双通道复数：通道0实部，通道1虚部

    // ========== 步骤6：频域除法 ==========
    // 拆分复数矩阵为实部、虚部两个单通道
    cv::Mat channels[2];
    cv::split(rhs_fft, channels);

    // 分母是纯实数，实部、虚部分别除以频域核（等价于复数除法）
    // 注意：必须直接赋值给 channels[i]，而不能赋值给中间变量后忘记写回
    channels[0] = channels[0] / freq_kernel;
    channels[1] = channels[1] / freq_kernel;

    // 合并回双通道复数矩阵
    cv::Mat T_fft;
    cv::merge(channels, 2, T_fft);

    // ========== 步骤7：逆 FFT 转回空域，得到最优 T ==========
    cv::Mat T;
    cv::idft(T_fft, T, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    T.convertTo(T, CV_32FC1);

    // ========== 步骤8：裁剪 T 到合理范围 ==========
    // FFT 求解可能产生负值或极小值，导致增强阶段除法放大噪声
    cv::max(T, 0.001f, T);
    cv::min(T, 1.0f, T);

    return T;
}


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
    cv::Mat& lambda_y)
{
    // ---------- 步骤1：固定 G、Λ，更新 T ----------
    T = SolveT(T_hat, G_x, G_y, lambda_x, lambda_y, rho);

    // ---------- 步骤2：计算新 T 的梯度 ∇T ----------
    cv::Mat grad_T_x = ComputeGradientX(T);
    cv::Mat grad_T_y = ComputeGradientY(T);

    // ---------- 步骤3：固定 T、Λ，更新 G ----------
    // 适配你原来的拼接式 SolveG 接口：把分量拼成2倍高度矩阵
    cv::Mat grad_T_concat, lambda_concat, G_concat;
    cv::vconcat(grad_T_x, grad_T_y, grad_T_concat);
    cv::vconcat(lambda_x, lambda_y, lambda_concat);

    // 小优化：你原来的 SolveG 内部又算了一次梯度，这里可以直接传梯度进去，避免重复计算
    // 如果你想改，可以把 SolveG 的入参改成直接传梯度，性能更好
    G_concat = SolveG(T, lambda_concat, weight, rho, alpha);

    // 拆分回 x、y 分量
    int H = T.rows;
    G_x = G_concat.rowRange(0, H).clone();
    G_y = G_concat.rowRange(H, 2 * H).clone();

    // ---------- 步骤4：更新对偶变量 Λ ----------
    // 对应公式：Λ = Λ + ρ * (G - ∇T)
    lambda_x = lambda_x + rho * (G_x - grad_T_x);
    lambda_y = lambda_y + rho * (G_y - grad_T_y);
}