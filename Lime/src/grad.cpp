#include "grade.h"
#include <opencv2/opencv.hpp>
#include <arm_neon.h>


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
        const float* img_row_ptr = img.ptr<float>(i);//读取第i行的首指针
        float* gx_row_ptr = GradientX.ptr<float>(i);//输出的第i行的首指针

        int j = 0;

        for(;j <= img.cols-5;j+=4){
            float32x4_t previous = vld1q_f32(img_row_ptr+j);
            float32x4_t next = vld1q_f32(img_row_ptr+1+j);//注意最后一位是j+4不是j+5

            float32x4_t sub = vsubq_f32(next,previous);
            vst1q_f32(gx_row_ptr+j,sub);
        }
        for(;j < img.cols-1;++j){
            gx_row_ptr[j] = img_row_ptr[j + 1] - img_row_ptr[j];
        }
        // for (int j = 0; j < img.cols - 1;++j){
        //     //at在每次调用时都会检查越界的情况，存在性能开销，带有性能开销 后续需要改进
        //     // GradientX.at<float>(i,j) = img.at<float>(i,j+1) - img.at<float>(i,j);
        //     // 等价于指针偏移：*(gx_row_ptr + j) = *(img_row_ptr + j + 1) - *(img_row_ptr + j);
        //     gx_row_ptr[j] = img_row_ptr[j + 1] - img_row_ptr[j];
        // }
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
        const float* curr_row = img.ptr<float>(i);
        const float* next_row = img.ptr<float>(i + 1);
        float* gy_row_ptr = GradientY.ptr<float>(i);

        int j = 0;
        for(;j <=img.cols - 4;j+=4){
            float32x4_t previous = vld1q_f32(curr_row+j);
            float32x4_t next = vld1q_f32(next_row+j);
            float32x4_t sub = vsubq_f32(next,previous);
            vst1q_f32(gy_row_ptr+j,sub);
        }
        for(;j<img.cols;++j){
            gy_row_ptr[j] = next_row[j] - curr_row[j];

        }

        // for(int j = 0;j < img.cols;++j){
        //     // GradientY.at<float>(i,j) = img.at<float>(i+1,j) - img.at<float>(i,j);
        //     gy_row_ptr[j] = next_row[j] - curr_row[j];
        // }
    }
    return GradientY;
}


//权重是跟梯度有关系，目的是梯度越大的地方 说明是处于物体的边缘，那么权重就越小，不能平滑，反之，梯度越小，那么权重就越大，可以平滑
cv::Mat ComputeWeight(const cv::Mat& gradientx, const cv::Mat& gradienty)
{
    // 与参考 LIME 实现一致：denominator = |gradient| + 1（不是 +0.001）
    // 这样 W ∈ [0.5, 1]，阈值 αW/ρ 变化平缓，配合 ρ 倍增形成 continuation
    cv::Mat absGradx = cv::abs(gradientx);
    cv::Mat absGrady = cv::abs(gradienty);

    cv::Mat Wx, Wy;
    cv::divide(1.0f, absGradx + 1.0f, Wx);
    cv::divide(1.0f, absGrady + 1.0f, Wy);

    cv::Mat weight;
    cv::vconcat(Wx, Wy, weight);
    cv::GaussianBlur(weight, weight, cv::Size(3, 3), 0, 0);

    return weight;
}

//软阈值函数改成SIMD的优化
cv::Mat SoftThreshold(const cv::Mat& value, const cv::Mat& threshold)
{
    CV_Assert(value.type() == CV_32FC1 && threshold.type() == CV_32FC1);
    CV_Assert(value.size() == threshold.size());

    cv::Mat output(value.size(), CV_32FC1);
    int cols = value.cols;

    // 预定义常量向量
    const float32x4_t zero_vec = vdupq_n_f32(0.0f);
    const float32x4_t one_vec = vdupq_n_f32(1.0f);
    const float32x4_t neg_one_vec = vdupq_n_f32(-1.0f);

    for (int i = 0; i < value.rows; ++i)
    {
        const float* d_ptr = value.ptr<float>(i);
        const float* th_ptr = threshold.ptr<float>(i);
        float* out_ptr = output.ptr<float>(i);

        int j = 0;
        // 批量4像素NEON循环
        for (; j <= cols - 4; j += 4)
        {
            float32x4_t d = vld1q_f32(d_ptr + j);
            float32x4_t th = vld1q_f32(th_ptr + j);

            float32x4_t abs_d = vabsq_f32(d);
            float32x4_t diff = vsubq_f32(abs_d, th);
            diff = vmaxq_f32(diff, zero_vec);

            // 计算符号向量（修复核心报错部分）
            uint32x4_t mask_pos = vcgtq_f32(d, zero_vec);
            uint32x4_t mask_neg = vcltq_f32(d, zero_vec);
            float32x4_t sign_pos = vbslq_f32(mask_pos, one_vec, zero_vec);
            float32x4_t sign_neg = vbslq_f32(mask_neg, neg_one_vec, zero_vec);
            float32x4_t sign = vaddq_f32(sign_pos, sign_neg);

            // 软阈值结果
            float32x4_t res = vmulq_f32(sign, diff);
            vst1q_f32(out_ptr + j, res);
        }

        // 尾部不足4个像素串行兜底
        for (; j < cols; ++j)
        {
            float d = d_ptr[j];
            float lambda = th_ptr[j];
            float abs_d = fabsf(d);
            out_ptr[j] = abs_d > lambda ? copysignf(abs_d - lambda, d) : 0.0f;
        }
    }
    return output;
}


cv::Mat SolveG(
    const cv::Mat& T,
    const cv::Mat& Lambda,
    const cv::Mat& Weight,
    const cv::Mat& GradientT,
    float rho,
    float alpha)
{
    // 计算 ∇T,因为之前的梯度是分开算的，因此在这里呢需要把水平方向和竖直方向的梯度拼接起来
    // cv::Mat GradientX = ComputeGradientX(T);
    // cv::Mat GradientY = ComputeGradientY(T);
    // cv::Mat GradientT;
    // cv::vconcat(GradientX, GradientY, GradientT);

    // d = ∇T - Λ/ρ
    cv::Mat d = GradientT - Lambda / rho;

    // λ = αW/ρ
    cv::Mat threshold = alpha * Weight / rho;//权重是固定的，alpha也是固定的，只有ρ是变化的

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
        const float* gx_row = gx.ptr<float>(i);
        const float* gy_row = gy.ptr<float>(i);
        
        float* div_row = div.ptr<float>(i);

        const float* gy_previous_row = nullptr;
        if(i > 0){
            gy_previous_row = gy.ptr<float>(i-1);
        }

        for(int j=0;j<gx.cols;j++)
        {
            float dx=gx_row[j];
            float dy=gy_row[j];

            if(j>0)
                dx-=gx_row[j-1];

            if(i>0)
                dy-=gy_previous_row[j];

            div_row[j]=dx+dy;
        }
    }

    return div;
}

//不要把不变的数学计算放在循环里重复算
/**
 * @brief 生成离散五点拉普拉斯算子的频域核（对应前向梯度+后向散度的离散格式）
 * @param size  输入图像尺寸
 * @param rho   ADMM惩罚参数
 * @return      频域实值核，CV_32FC1，尺寸与输入一致
 */
cv::Mat GenerateLaplacianFreqKernel(cv::Size size, float rho)
{
    static cv::Mat g_laplacian_freq_cache;
    static cv::Size g_cached_size(0, 0);

    if (g_cached_size != size)
    {
        int H = size.height;
        int W = size.width;
        g_laplacian_freq_cache.create(H, W, CV_32FC1);

        for (int v = 0; v < H; v++)
        {
            for (int u = 0; u < W; u++)
            {
                float cos_u = cosf(2.0f * (float)M_PI * u / W);
                float cos_v = cosf(2.0f * (float)M_PI * v / H);
                g_laplacian_freq_cache.at<float>(v, u) = 2.0f * cos_u + 2.0f * cos_v - 4.0f;
            }
        }
        g_cached_size = size;
    }

    // 🔥 直接用 OpenCV 的矩阵运算替代 for 循环
    cv::Mat kernel = 2.0 - rho * g_laplacian_freq_cache;
    
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
    CV_Assert(!T_hat.empty() && T_hat.type() == CV_32FC1);
    CV_Assert(G_x.size() == T_hat.size() && G_x.type() == CV_32FC1);
    CV_Assert(G_y.size() == T_hat.size() && G_y.type() == CV_32FC1);
    CV_Assert(lambda_x.size() == T_hat.size() && lambda_x.type() == CV_32FC1);
    CV_Assert(lambda_y.size() == T_hat.size() && lambda_y.type() == CV_32FC1);

    const cv::Size sz = T_hat.size();
    const int H = sz.height;
    const int W = sz.width;

    // ====== FFT 最优尺寸：质数尺寸（如 1039×789）比 2 的幂（如 1024×768）慢 20 倍 ======
    const int optH = cv::getOptimalDFTSize(H);
    const int optW = cv::getOptimalDFTSize(W);
    const bool need_pad = (optH != H || optW != W);

    // ------ Buffer 复用：padded 尺寸的 static 缓冲区 ------
    static cv::Size  s_opt_sz(0, 0);
    static cv::Mat   s_fft;          // padded 尺寸的频域复数矩阵
    static cv::Mat   s_inv_kernel;   // padded 尺寸的频域核倒数

    const cv::Size pad_sz(optW, optH);
    if (pad_sz != s_opt_sz)
    {
        s_inv_kernel = cv::Mat(optH, optW, CV_32FC1);
        s_opt_sz     = pad_sz;
    }
    cv::TickMeter tm;
    double t_kernel = 0, t_div = 0, t_fft_fwd = 0, t_freq_mul = 0, t_fft_inv = 0;

    // ====== 步骤1：b = G + Λ/ρ ======
    const float inv_rho = 1.0f / rho;
    cv::Mat b_x, b_y;
    cv::scaleAdd(lambda_x, inv_rho, G_x, b_x);
    cv::scaleAdd(lambda_y, inv_rho, G_y, b_y);

    // ====== 步骤2：div(b) ======
    tm.reset();tm.start();
    cv::Mat div_b = ComputeDivergence(b_x, b_y);
    tm.stop();t_div = tm.getTimeMilli();

    // ====== 步骤3：rhs = 2·T_hat - ρ·div_b ======
    cv::Mat rhs, rhs_padded;
    cv::scaleAdd(div_b, -rho, 2.0f * T_hat, rhs);

    // ====== 步骤4：生成频域核 + 倒数 ======
    tm.reset(); tm.start();
    cv::Mat kernel = GenerateLaplacianFreqKernel(
        need_pad ? pad_sz : sz, rho);
    cv::divide(1.0f, kernel, s_inv_kernel);   // s_inv_kernel = 1/kernel
    tm.stop(); t_kernel = tm.getTimeMilli();

    // ====== 步骤5：rhs padding 到最优 FFT 尺寸 + 正 FFT ======
    tm.reset(); tm.start();
    if (need_pad)
    {
        cv::copyMakeBorder(rhs, rhs_padded,
            0, optH - H, 0, optW - W, cv::BORDER_REFLECT);
        cv::dft(rhs_padded, s_fft, cv::DFT_COMPLEX_OUTPUT);
    }
    else
    {
        cv::dft(rhs, s_fft, cv::DFT_COMPLEX_OUTPUT);
    }
    tm.stop(); t_fft_fwd = tm.getTimeMilli();

    // ====== 步骤6：频域乘法（s_fft × inv_kernel，就地修改）======
    tm.reset(); tm.start();
    {
        const int   total = optH * optW;
        float*      ptr   = s_fft.ptr<float>(0);
        const float* invk = s_inv_kernel.ptr<float>(0);
        for (int i = 0; i < total; i++)
        {
            ptr[0] *= invk[i];
            ptr[1] *= invk[i];
            ptr += 2;
        }
    }
    tm.stop(); t_freq_mul = tm.getTimeMilli();

    // ====== 步骤7：逆 FFT + 裁剪回原尺寸 ======
    tm.reset(); tm.start();
    cv::Mat T_padded, T;
    cv::idft(s_fft, T_padded, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);

    if (need_pad)
    {
        T = T_padded(cv::Rect(0, 0, W, H)).clone();
    }
    else
    {
        T = T_padded;
    }
    T.convertTo(T, CV_32FC1);
    tm.stop(); t_fft_inv = tm.getTimeMilli();

    static int solveT_call_count = 0;
    solveT_call_count++;
    if (solveT_call_count == 1 || solveT_call_count == 10)
    {
          std::cout << "    SolveT 内部 (第" << solveT_call_count << "次):" << std::endl;
          std::cout << "      Divergence:      " << t_div      << " ms" << std::endl;
          std::cout << "      FreqKernel+inv:  " << t_kernel    << " ms" << std::endl;
          std::cout << "      FFT forward:     " << t_fft_fwd  << " ms" << std::endl;
          std::cout << "      Freq multiply:   " << t_freq_mul << " ms" << std::endl;
          std::cout << "      FFT inverse:     " << t_fft_inv  << " ms" << std::endl;
          std::cout << "      SolveT 合计:     "
                    << (t_div + t_kernel + t_fft_fwd + t_freq_mul + t_fft_inv)
                    << " ms" << std::endl;
    }


    // ====== 步骤8：裁剪 T 到合理范围 ======
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
static int s_admm_call_count = 0;  // 全局计数器，控制打印频率

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
    cv::TickMeter tm;
    double t_solveT = 0, t_grad = 0, t_vconcat = 0;
    double t_solveG = 0, t_split = 0, t_lambda = 0;

    // ---------- 步骤1：固定 G、Λ，更新 T ----------
    tm.start();
    T = SolveT(T_hat, G_x, G_y, lambda_x, lambda_y, rho);
    tm.stop();
    t_solveT = tm.getTimeMilli();

    // ---------- 步骤2：计算新 T 的梯度 ∇T ----------
    tm.reset(); tm.start();
    cv::Mat grad_T_x = ComputeGradientX(T);
    cv::Mat grad_T_y = ComputeGradientY(T);
    tm.stop();
    t_grad = tm.getTimeMilli();

    // ---------- 步骤3：固定 T、Λ，更新 G ----------
    // 适配你原来的拼接式 SolveG 接口：把分量拼成2倍高度矩阵
    tm.reset(); tm.start();
    cv::Mat grad_T_concat, lambda_concat, G_concat;
    cv::vconcat(grad_T_x, grad_T_y, grad_T_concat);
    cv::vconcat(lambda_x, lambda_y, lambda_concat);
    tm.stop();
    t_vconcat = tm.getTimeMilli();

    tm.reset(); tm.start();
    G_concat = SolveG(T, lambda_concat, weight,grad_T_concat, rho, alpha);
    tm.stop();
    t_solveG = tm.getTimeMilli();

    // 拆分回 x、y 分量
    tm.reset(); tm.start();
    int H = T.rows;
    G_x = G_concat.rowRange(0, H).clone();
    G_y = G_concat.rowRange(H, 2 * H).clone();
    tm.stop();
    t_split = tm.getTimeMilli();

    // ---------- 步骤4：更新对偶变量 Λ ----------
    // 对应公式：Λ = Λ + ρ * (G - ∇T)
    tm.reset(); tm.start();
    lambda_x = lambda_x + rho * (G_x - grad_T_x);
    lambda_y = lambda_y + rho * (G_y - grad_T_y);
    tm.stop();
    t_lambda = tm.getTimeMilli();

    // --- 打印：只在第 1 轮和最后一轮打印 ---
    s_admm_call_count++;
    if (s_admm_call_count == 1 || s_admm_call_count == 10)
    {
        std::cout << "  ADMM_Step 内部 (第" << s_admm_call_count << "次, rho=" << rho << "):" << std::endl;
        std::cout << "    [1] SolveT:       " << t_solveT   << " ms" << std::endl;
        std::cout << "    [2] Gradient:     " << t_grad     << " ms" << std::endl;
        std::cout << "    [3] vconcat:      " << t_vconcat  << " ms" << std::endl;
        std::cout << "    [4] SolveG:       " << t_solveG   << " ms" << std::endl;
        std::cout << "    [5] split G:      " << t_split    << " ms" << std::endl;
        std::cout << "    [6] Lambda:       " << t_lambda   << " ms" << std::endl;
        std::cout << "    ADMM_Step 合计:   "
                  << (t_solveT + t_grad + t_vconcat + t_solveG + t_split + t_lambda)
                  << " ms" << std::endl;
    }
}