#include "grade.h"
#include <opencv2/opencv.hpp>
#include <arm_neon.h>
#include <omp.h>  //多线程
#include <fftw3.h>
#include <fftw3.h>

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

#pragma omp parallel for
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
#pragma omp parallel for
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


// 一趟融合：d = GradientT - Lambda/rho, thresh = inter/rho, soft_threshold → G_out
// 三个逐元素操作合并为一个循环，内存流量从 ~42MB 降到 ~22MB
void SolveG(
    const cv::Mat& Lambda,
    const cv::Mat& GradientT,
    const cv::Mat& inter,
    float rho,
    cv::Mat& G_out)
{
    G_out.create(GradientT.size(), CV_32FC1);

    const float inv_rho = 1.0f / rho;
    const int cols = GradientT.cols;

    const float32x4_t zero     = vdupq_n_f32(0.0f);
    const float32x4_t one      = vdupq_n_f32(1.0f);
    const float32x4_t neg_one  = vdupq_n_f32(-1.0f);
    const float32x4_t v_inv    = vdupq_n_f32(inv_rho);

    #pragma omp parallel for
    for (int i = 0; i < GradientT.rows; ++i)
    {
        const float* lam_ptr   = Lambda.ptr<float>(i);
        const float* grad_ptr  = GradientT.ptr<float>(i);
        const float* inter_ptr = inter.ptr<float>(i);
        float*       out_ptr   = G_out.ptr<float>(i);

        int j = 0;
        for (; j <= cols - 4; j += 4)
        {
            // 1. 加载三个输入
            float32x4_t lam   = vld1q_f32(lam_ptr + j);
            float32x4_t grad  = vld1q_f32(grad_ptr + j);
            float32x4_t inter = vld1q_f32(inter_ptr + j);

            // 2. d = GradientT - Lambda / rho（vmlsq_f32: a - b*c，一条指令完成乘减）
            float32x4_t d = vmlsq_f32(grad, lam, v_inv);

            // 3. thresh = inter / rho
            float32x4_t thresh = vmulq_f32(inter, v_inv);

            // 4. soft_threshold: sign(d) * max(|d| - thresh, 0)
            float32x4_t abs_d = vabsq_f32(d);
            float32x4_t diff  = vmaxq_f32(vsubq_f32(abs_d, thresh), zero);

            uint32x4_t mask_pos = vcgtq_f32(d, zero);
            uint32x4_t mask_neg = vcltq_f32(d, zero);
            float32x4_t sign = vaddq_f32(
                vbslq_f32(mask_pos, one, zero),
                vbslq_f32(mask_neg, neg_one, zero));

            vst1q_f32(out_ptr + j, vmulq_f32(sign, diff));
        }

        // 尾部标量兜底
        for (; j < cols; ++j)
        {
            float d = grad_ptr[j] - lam_ptr[j] * inv_rho;
            float thresh = inter_ptr[j] * inv_rho;
            float abs_d = fabsf(d);
            out_ptr[j] = abs_d > thresh ? copysignf(abs_d - thresh, d) : 0.0f;
        }
    }
    // std::cout << "OpenCV threads: " << cv::getNumThreads() << std::endl;
}


//计算散度（后向差分对应着梯度的前向差分）
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

    // ------ FFTW3 Buffer 复用：plan + work buffer 跨迭代缓存 ------
    // FFTW r2c 输出尺寸为 optH × (optW/2+1)（半复数格式，利用共轭对称性省一半内存）
    static fftwf_plan     s_plan_fwd  = nullptr;  // 正变换 plan: real → half-complex
    static fftwf_plan     s_plan_inv  = nullptr;  // 逆变换 plan: half-complex → real
    static cv::Size       s_plan_sz(0, 0);
    static float*         s_fftw_buf  = nullptr;  // 实数 buffer（正向输入 / 逆向输出复用）
    static fftwf_complex* s_fftw_cplx = nullptr;  // 复数 buffer（正向输出 / 逆向输入复用）
    static cv::Mat        s_inv_kernel;            // 频域核倒数（半复数尺寸）

    const cv::Size pad_sz(optW, optH);
    if (pad_sz != s_plan_sz)
    {
        // 销毁旧 plan（必须在分配新 buffer 前，否则 FFTW_MEASURE 会写坏旧 buffer）
        if (s_plan_fwd) { fftwf_destroy_plan(s_plan_fwd); s_plan_fwd = nullptr; }
        if (s_plan_inv) { fftwf_destroy_plan(s_plan_inv); s_plan_inv = nullptr; }
        fftwf_free(s_fftw_buf);
        fftwf_free(s_fftw_cplx);

        const int N_pixels = optW * optH;
        const int N_cplx   = optH * (optW / 2 + 1);
        s_fftw_buf  = fftwf_alloc_real(N_pixels);
        s_fftw_cplx = fftwf_alloc_complex(N_cplx);

        // FFTW_ESTIMATE: 瞬时建 plan，不做自优化（ARM 上 MEASURE 太慢）
        s_plan_fwd = fftwf_plan_dft_r2c_2d(optH, optW,
                        s_fftw_buf, s_fftw_cplx, FFTW_ESTIMATE);
        s_plan_inv = fftwf_plan_dft_c2r_2d(optH, optW,
                        s_fftw_cplx, s_fftw_buf, FFTW_ESTIMATE);

        s_plan_sz = pad_sz;
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

    // ====== 步骤4：生成频域核倒数（FFTW 半复数格式：W → W/2+1）======
    tm.reset(); tm.start();
    {
        cv::Mat kernel_full = GenerateLaplacianFreqKernel(
            need_pad ? pad_sz : sz, rho);
        // FFTW r2c 只输出 optH × (optW/2+1) 个复数 → 核也只需前半列
        cv::Mat kernel_half = kernel_full(cv::Rect(0, 0, optW / 2 + 1, optH));
        cv::divide(1.0f, kernel_half, s_inv_kernel);
    }
    tm.stop(); t_kernel = tm.getTimeMilli();

    // ====== 步骤5：正 FFT（FFTW r2c：实数 → 半复数，写入 s_fftw_cplx）======
    tm.reset(); tm.start();
    {
        const int N_pixels = optW * optH;
        if (need_pad)
        {
            cv::copyMakeBorder(rhs, rhs_padded,
                0, optH - H, 0, optW - W, cv::BORDER_REFLECT);
            memcpy(s_fftw_buf, rhs_padded.ptr<float>(0), N_pixels * sizeof(float));
        }
        else
        {
            memcpy(s_fftw_buf, rhs.ptr<float>(0), N_pixels * sizeof(float));
        }
        fftwf_execute(s_plan_fwd);   // s_fftw_buf → s_fftw_cplx
    }
    tm.stop(); t_fft_fwd = tm.getTimeMilli();

    // ====== 步骤6：频域乘法（FFTW 半复数 × inv_kernel，就地修改 s_fftw_cplx）======
    tm.reset(); tm.start();
    {
        const int   N_cplx = optH * (optW / 2 + 1);
        const float* invk  = s_inv_kernel.ptr<float>(0);

        #pragma omp parallel for
        for (int i = 0; i < N_cplx; i++)
        {
            s_fftw_cplx[i][0] *= invk[i];   // 实部
            s_fftw_cplx[i][1] *= invk[i];   // 虚部
        }
    }
    tm.stop(); t_freq_mul = tm.getTimeMilli();

    // ====== 步骤7：逆 FFT + 手动归一化（FFTW 不做 DFT_SCALE，需手动 /N）======
    cv::Mat T;
    tm.reset(); tm.start();
    fftwf_execute(s_plan_inv);   // s_fftw_cplx → s_fftw_buf（c2r 会破坏输入复数）

    // FFTW 不做归一化，手动除以 N = optW * optH
    {
        const float inv_N   = 1.0f / (float)(optW * optH);
        const int   N_pixels = optW * optH;
        #pragma omp parallel for
        for (int i = 0; i < N_pixels; i++)
            s_fftw_buf[i] *= inv_N;
    }

    if (need_pad)
    {
        // 用 cv::Mat 包装 FFTW buffer（不拷贝数据），再 crop 回原尺寸
        cv::Mat T_padded(optH, optW, CV_32FC1, s_fftw_buf);
        T = T_padded(cv::Rect(0, 0, W, H)).clone();
    }
    else
    {
        T = cv::Mat(H, W, CV_32FC1, s_fftw_buf).clone();
    }
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
    const cv::Mat& inter,
    float rho,
    cv::Mat& T,
    cv::Mat& G_x,
    cv::Mat& G_y,
    cv::Mat& lambda_x,
    cv::Mat& lambda_y)
{
    static cv::Mat s_G_concat;  // 复用 buffer

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
    tm.reset(); tm.start();
    cv::Mat grad_T_concat, lambda_concat;
    cv::vconcat(grad_T_x, grad_T_y, grad_T_concat);
    cv::vconcat(lambda_x, lambda_y, lambda_concat);
    tm.stop();
    t_vconcat = tm.getTimeMilli();

    tm.reset(); tm.start();
    s_G_concat.create(grad_T_concat.size(), CV_32FC1);
    SolveG(lambda_concat, grad_T_concat, inter, rho, s_G_concat);
    tm.stop();
    t_solveG = tm.getTimeMilli();

    // 拆分回 x、y 分量
    tm.reset(); tm.start();
    int H = T.rows;
    G_x = s_G_concat.rowRange(0, H).clone();
    G_y = s_G_concat.rowRange(H, 2 * H).clone();
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