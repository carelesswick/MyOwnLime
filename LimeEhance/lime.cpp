// #include <opencv2/core/core.hpp> // OpenCV 核心模块，包含了基本的数据结构和算法
// #include <opencv2/highgui.hpp> // OpenCV GUI 模块，包含了图像和视频的 I/O 函数
// #include <opencv2/imgproc/imgproc.hpp> // OpenCV 图像处理模块，包含了图像处理函数
// #include <opencv2/imgproc/types_c.h>
#include <opencv2/opencv.hpp>
#include <opencv2/photo.hpp>       // fastNlMeansDenoisingColored 去噪
#include <opencv2/imgproc.hpp>     // CLAHE 对比度增强
#include <iostream>                // 输入输出流库
#include <cstring>
#include <math.h>
#include <complex.h>
#include <vector>
#include <cmath>

using namespace std;

namespace LIME //LIME的命名空间
{

    class lime
    {  //存储数据
        public:
        cv::Mat img_norm; //输入图像归一化
        cv::Mat R;        //反射率图（Retinex分解的R分量）
        cv::Mat out_lime; //增强后的图像
        cv::Mat dv;
        cv::Mat dh;
        cv::Mat T_hat;    //初始化的光照图
        cv::Mat W; 	      //权重矩阵
        cv::Mat veCDD;
        int channel; 	  //存储图像通道数
        int row; 		  //图像的行数
        int col; 		  //图像的列数
        float alpha=1;    //平滑项权重（控制光照图平滑程度）
        float rho =2;     //ADMM惩罚参数增长率
        float gamma = 0.7;//Gamma校正指数（<1避免过度增强暗区）
        float epsilon;    //迭代收敛系数
        float thd;        //迭代收敛时间阈值

        // 后处理控制参数
        bool  use_denoise = true;      //是否启用BM3D去噪
        float denoise_h  = 3.0f;       //去噪强度（越大去噪越强）
        bool  use_clahe  = false;      //是否启用CLAHE对比度增强
        float clahe_clip = 2.0f;       //CLAHE裁剪限幅
        bool  use_gamma_on_T = true;   //是否对光照图T做Gamma校正

        // 调试/中间结果输出
        bool  save_intermediate = false;//是否保存中间结果

    	public:
        lime(cv::Mat src);
        cv::Mat Mat2Vec(cv::Mat mat);                         //矩阵向量化
        cv::Mat reshape1D(cv::Mat mat);						  //按行压缩矩阵到一维
        cv::Mat getReal(cv::Mat mat);						  //获取矩阵实部
        cv::Mat derivative(cv::Mat matrix);					  //求解矩阵导数
        cv::Mat solveT(cv::Mat G, cv::Mat Z, float u);		  //求解子问题T
        cv::Mat solveG(cv::Mat T,cv::Mat Z,float u,cv::Mat W);//求解子问题G
        cv::Mat solveZ(cv::Mat T,cv::Mat G,cv::Mat Z,float u);//求解子问题Z
        cv::Mat Dev(int n, int k);							  //求解矩阵一阶导数
        cv::Mat getMax(const cv::Mat &bgr);					  //获取色彩通道最大值
        cv::Mat optIllumMap();								  //获取优化后光照图
        cv::Mat enhance(cv::Mat &src);						  //图像增强（含后处理）
        cv::Mat postProcess(cv::Mat &enhanced);               //后处理管道

        static inline float comp(float& a, float& b, float& c) // 声明一个静态内联函数，用于比较三个浮点数的大小，并返回最大值
        {
            return fmax(a, fmax(b, c));//fmax是标准C++数学库函数 返回俩个浮点数中的较大值属于cmath头文件
        }

        void weightStrategy();
        void _init_IllumMap(cv::Mat src);
        void Illum_filter(cv::Mat& img_in, cv::Mat& img_out); //滤波、GAMMA矫正
        void Illumination(cv::Mat& src, cv::Mat& out); 		  //求解每个像素的光照强度
        float solveU(float u);								  //求解子问题u
        float Frobenius(cv::Mat mat);
    };

    //构造函数
    lime::lime(cv::Mat src)
    {
        //获取输入图像的通道数
        channel = src.channels();
    }

    //光照图初始化
    void lime::_init_IllumMap(cv::Mat src){
        // 将输入图像转换为 float CV_32F类型，并进行归一化
        src.convertTo(img_norm, CV_32F, 1 / 255.0, 0); //第三个参数进行了缩放，也就是归一化。
        // 获取归一化图像的大小
        cv::Size sz(img_norm.size());
        row = img_norm.rows;//rows 对应图像的高度
        col = img_norm.cols;//cols 对应图像的宽度
        //构建初始照明图T
        T_hat = lime::getMax(img_norm);
        //求T_hat的f范数
        epsilon = Frobenius(T_hat)*0.001;
        dv = Dev(row, 1);//dv*图片
        dh = Dev(col, -1);//图片*dh
        //float u = dv.at<float>(0,0);
        //float u2 = dh.at<float>(0,0);
        veCDD = cv::Mat(1,row*col, CV_32F, cv::Scalar::all(0.0));
        //定义一维矩阵并初始化为0
        veCDD.at<float>(0,0) = 4;
        veCDD.at<float>(0,1) = -1;
        veCDD.at<float>(0,row) = -1;
        veCDD.at<float>(0,row*col-1) = -1;
        veCDD.at<float>(0,row*col-row) = -1;
    }

    //获取色彩通道最大值
    cv::Mat lime::getMax(const cv::Mat& bgr)
        {
            cv::Mat temp_mat(row, col, CV_32F, cv::Scalar::all(0.0));//定义一个全零矩阵
            std::vector<cv::Mat> img_norm_rgb;//多个矩阵组成的动态数组
            cv::Mat img_norm_b, img_norm_g, img_norm_r;//定义三个单通道矩阵
            cv::split(bgr, img_norm_rgb);//将bgr图像分割成三个单通道矩阵，分别存放在img_norm_rgb[0]、img_norm_rgb[1]、img_norm_rgb[2]中
            img_norm_g = img_norm_rgb.at(0);
            img_norm_b = img_norm_rgb.at(1);
            img_norm_r = img_norm_rgb.at(2);
            for(int i = 0; i < row; i++){
                for(int j = 0; j< col; j++){
                    temp_mat.at<float>(i,j) = MAX(MAX(img_norm_g.at<float>(i,j),img_norm_b.at<float>(i,j)), img_norm_r.at<float>(i,j));
                }
            }
            return temp_mat;
        }

    //求解矩阵范数：计算矩阵的 Frobenius 范数，即矩阵中每个元素平方和的平方根。这是用于衡量矩阵的整体"大小"或"能量"的一种方式
    float lime::Frobenius(cv::Mat mat)
    {
        int row = mat.rows;
        int col = mat.cols;
        float total = 0.0;
        for(int i = 0; i < row ; i++){
            for(int j =0; j< col ; j++){
                total = total + pow(mat.at<float>(i,j), 2);
            }
        }
        total = sqrt(total);
        return total;
    }

    //求解矩阵导数
    cv::Mat lime::derivative(cv::Mat matrix){
        cv::Mat v = dv * matrix;
        cv::Mat h = matrix * dh;
        cv::Mat matrix_C ;
        //矩阵垂直拼接
        cv::vconcat(v,h,matrix_C);
        return matrix_C;
    }

    //求解子问题T
    cv::Mat lime::solveT(cv::Mat G, cv::Mat Z, float u){
        cv::Mat X = G - (Z / u);   // X = G - Z/u, 对应ADMM中T子问题的中间变量
        int row_temp = X.rows;
        cv::Mat Xv = X.rowRange(0, row);  //左闭右开区间进行矩阵行的提取
        cv::Mat Xh = X.rowRange(row,row_temp);//要取 -1
        cv::Mat temp = dv*Xv+ Xh*dh;
        cv::Mat numerator;
        cv::Mat denominator;
        cv::Mat mat_temp1;
        mat_temp1 = Mat2Vec(2*T_hat + u*temp);
        //使用opencv自带的离散傅里叶变换函数执行傅里叶变换运算
        cv::dft(mat_temp1,numerator,cv::DFT_COMPLEX_OUTPUT);
        cv::Mat mat_temp2 = veCDD* u;
        cv::dft(mat_temp2, denominator,cv::DFT_COMPLEX_OUTPUT);
        denominator = denominator + 2;
        cv::Mat T_temp;
        temp = numerator / denominator;
        temp = getReal(temp);
        cv::dft(temp,T_temp,cv::DFT_COMPLEX_OUTPUT);
        T_temp = getReal(T_temp);
        T_temp = T_temp/(T_temp.cols);
        auto u5 = T_temp.at<float>(0,0);
        auto u6 = T_temp.at<float>(0,4);
        normalize(T_temp,T_temp,0.2,1,cv::NORM_MINMAX);
        cv::Mat T = reshape1D(T_temp);
        T.convertTo(T, CV_32F);
        return T;
    }
    cv::Mat lime::getReal(cv::Mat mat){ //获取矩阵的实部
        int col_temp = mat.cols;
        cv::Mat mat_return(1,col_temp, CV_32F, cv::Scalar::all(0.0));
        for(int i =0; i<col_temp; i++){
                mat_return.at<float>(0,i) = mat.at<float>(0,2*i);
            }
        return mat_return;

    }

    //将多维矩阵压缩成一维
    cv::Mat lime::Mat2Vec(cv::Mat mat){
        mat = mat.t(); //矩阵转置
        int row = mat.rows;
        int col = mat.cols;
        cv::Mat mat_one(1,row * col, CV_32F);
        int num_elements = row * col;

        for(int i = 0; i < row ; i++){
            for(int j =0; j< col ; j++){
                mat_one.at<float>(0,i*col+j) = mat.at<float>(i,j);
            }
        }

        return mat_one;
    }

    //将一维向量reshape回二维矩阵
    cv::Mat lime::reshape1D(cv::Mat mat){
        cv::Mat mat_temp(row,col, CV_32F);

        for(int i = 0; i < row ; i++){
        for(int j =0; j< col ; j++){

            mat_temp.at<float>(i,j) = mat.at<float>(0,i*col + j);
            }
        }
        return mat_temp;
    }

    //求解子问题G（软阈值操作）
    cv::Mat lime::solveG(cv::Mat T,cv::Mat Z,float u,cv::Mat W){
        //求出 T的一阶导数
        cv::Mat dT = derivative(T);
        cv::Mat epsilon = alpha * W / u;
        cv::Mat X = dT + Z / u;
        //获取一个图像矩阵的符号矩阵
        int row_temp = X.rows;
        int col_temp = X.cols;
        cv::Mat mat_temp(row_temp,col_temp,CV_32F);

       for(int i = 0; i < row_temp ; i++){
        for(int j =0; j< col_temp ; j++){
            if (X.at<float>(i,j) >0){
                mat_temp.at<float>(i,j) = 1;
            }
            else if(X.at<float>(i,j)<0){
                mat_temp.at<float>(i,j) =-1;
            }
            else
            mat_temp.at<float>(i,j) = 0;
        }
      }

        cv::Mat S_ce =cv::max(cv::abs(X) - epsilon, 0);
        cv::Mat O = mat_temp.mul(S_ce);
        return O;
    }

    //求解子问题Z（拉格朗日乘子更新）
    cv::Mat lime::solveZ(cv::Mat T,cv::Mat G,cv::Mat Z,float u){
        cv::Mat dT = derivative(T);
        return Z + u*(dT - G);
    }

    //求解子问题u（惩罚参数增长）
    float lime::solveU(float u){
        return u* rho;
    }

    //求解权重矩阵
    void lime::weightStrategy(){
        cv::Mat dTv = dv * T_hat;//差分矩阵获得垂直方向梯度
        cv::Mat dTh = T_hat* dh;//差分矩阵获得水平方向梯度
        cv::Mat Wv = 1/ (cv::abs(dTv) + 1);//abs() 的作用是对矩阵元素取绝对值，梯度幅值映射成权重
        cv::Mat Wh = 1/ (cv::abs(dTh) + 1);
        /**
         * 梯度大，说明这个位置变化剧烈，可能是边缘、纹理、噪声
         * 梯度小，说明这个位置比较平滑，属于更可信的区域
            所以取倒数后：
            梯度大时，权重变小
            梯度小时，权重变大
         * 这正好符合 LIME 里的思路：在优化光照图时，更相信平滑区域，弱化边缘和突变区域的影响。
         * 加 1 是为了避免分母为 0，同时让数值更稳定。
         */
        cv::vconcat(Wv, Wh, W);//将wv和wh两个矩阵垂直拼接
    }

    //求矩阵导数（构建差分矩阵）
    cv::Mat lime::Dev(int n, int k){
        cv::Mat mat_temp = cv::Mat::eye(n,n,CV_32F);  //eye()函数用于创建一个单位矩阵
        mat_temp = mat_temp *-1;//让矩阵的对角元素为-1
        //让矩阵k对角的元素为1
        if(k > 0){
        for(int y = 0;y <n - k; y++){
        mat_temp.at<float>(y,y + k) = 1;
        }
    }
    else{
        for(int y = -k;y <n ; y++){
        mat_temp.at<float>(y,y + k) = 1;
        }
    }
    return mat_temp;
    }

    //获取优化后的光照图（ADMM迭代求解）
    cv::Mat lime::optIllumMap(){
        //得到权重矩阵W
        weightStrategy();
        cv::Mat T(row,col, CV_32F, cv::Scalar::all(0.0));
        cv::Mat G(row*2,col, CV_32F, cv::Scalar::all(0.0));
        cv::Mat Z(row*2,col, CV_32F, cv::Scalar::all(0.0));
        int t = 0;
        float u = 1;
        const int MAX_ITER = 100;  // 安全上限，防止无穷循环

        while (true){
            T = solveT(G,Z,u);
            G = solveG(T,Z,u,W);
            Z = solveZ(T,G,Z,u);
            u = solveU(u);

            //加速收敛过程
            if(t == 0){
                float temp = Frobenius(derivative(T) - G);
                thd = ceil(2* log(temp / epsilon));
                // 安全钳：阈值异常时设上限
                if (thd <= 0 || std::isnan(thd) || std::isinf(thd)) {
                    thd = 10;
                }
            }
            t += 1;
            //达到收敛阈值或超过最大迭代次数就结束迭代
            if(t >= thd || t >= MAX_ITER){
                break;
            }
        }
        auto r1 = T.at<float>(0,0);
        auto r2 = T.at<float>(1,0);
        auto r3 = T.at<float>(2,0);

        if (save_intermediate) {
            cv::Mat T_vis;
            T.convertTo(T_vis, CV_8U, 255.0);
            cv::imwrite("intermediate_T_optimized.jpg", T_vis);
        }

        return T;
    }

    // ======================== 后处理管道 ========================
    // 对增强后的图像进行去噪和对比度增强
    cv::Mat lime::postProcess(cv::Mat &enhanced) {
        cv::Mat result = enhanced.clone();

        // ---- Step 1: BM3D去噪（LIME论文推荐的后处理） ----
        // 使用OpenCV内置的 fastNlMeansDenoisingColored 近似BM3D效果
        if (use_denoise && result.channels() == 3) {
            cv::Mat denoised;
            // 转换为8位进行去噪
            cv::Mat result_8u;
            if (result.depth() == CV_32F || result.depth() == CV_64F) {
                result.convertTo(result_8u, CV_8U, 255.0);
            } else {
                result_8u = result;
            }

            // fastNlMeansDenoisingColored: 非局部均值去噪（BM3D的近亲）
            // 参数: src, dst, h(亮度去噪强度), hColor(色度去噪强度),
            //       templateWindowSize, searchWindowSize
            cv::fastNlMeansDenoisingColored(
                result_8u, denoised,
                denoise_h,          // 亮度分量去噪强度
                denoise_h,          // 色度分量去噪强度
                7,                  // 模板窗口大小
                21                  // 搜索窗口大小
            );

            // 转回float用于后续处理
            denoised.convertTo(result, CV_32F, 1.0 / 255.0);

            if (save_intermediate) {
                cv::imwrite("intermediate_denoised.jpg", denoised);
            }
        }

        // ---- Step 2: CLAHE 对比度增强（可选） ----
        // 在亮度通道上做CLAHE，提升暗区细节同时压制过曝
        if (use_clahe && result.channels() == 3) {
            cv::Mat result_8u;
            result.convertTo(result_8u, CV_8U, 255.0);

            // 转到LAB色彩空间，只在L通道做CLAHE
            cv::Mat lab;
            cv::cvtColor(result_8u, lab, cv::COLOR_BGR2Lab);

            std::vector<cv::Mat> lab_channels;
            cv::split(lab, lab_channels);

            // 对L通道做CLAHE
            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clahe_clip);
            clahe->apply(lab_channels[0], lab_channels[0]);

            cv::merge(lab_channels, lab);
            cv::cvtColor(lab, result_8u, cv::COLOR_Lab2BGR);

            result_8u.convertTo(result, CV_32F, 1.0 / 255.0);

            if (save_intermediate) {
                cv::imwrite("intermediate_clahe.jpg", result_8u);
            }
        }

        return result;
    }

    //图像增强（含完整后处理管道）
    cv::Mat lime::enhance(cv::Mat &src){
        // ---- 阶段1: 初始化 ----
        _init_IllumMap(src);         //估算初始光照图 T_hat

        // ---- 阶段2: ADMM优化光照图 ----
        cv::Mat T = optIllumMap();   //通过ADMM迭代优化得到光照图T

        // ---- 阶段3: 光照图后处理（Gamma校正） ----
        // 根据LIME论文公式(10): R = I / (T^γ)
        // γ < 1 可避免暗区被过度增强
        cv::Mat T_processed;
        if (use_gamma_on_T) {
            // T ← T^γ, 同时防止除零
            cv::pow(T, gamma, T_processed);
            cv::threshold(T_processed, T_processed, 0.0001, 0.0001, cv::THRESH_TOZERO);
        } else {
            T_processed = T.clone();
            cv::threshold(T_processed, T_processed, 0.0001, 0.0001, cv::THRESH_TOZERO);
        }

        // ---- 阶段4: Retinex分解 ----
        // 逐通道增强: enhanced_c = I_c / T_processed
        cv::Size sz(img_norm.size());
        R = cv::Mat(sz, CV_32F, cv::Scalar::all(0.0));

        std::vector<cv::Mat> img_norm_rgb;
        cv::split(img_norm, img_norm_rgb);

        // BGR顺序: [0]=B, [1]=G, [2]=R
        for (int c = 0; c < 3; c++) {
            cv::Mat enhanced = img_norm_rgb[c] / T_processed;
            // 钳制负值和过大的值
            cv::threshold(enhanced, enhanced, 0.0, 0.0, cv::THRESH_TOZERO);
            cv::threshold(enhanced, enhanced, 1.0, 1.0, cv::THRESH_TRUNC);
            img_norm_rgb[c] = enhanced;
        }

        cv::Mat enhanced_float;
        cv::merge(img_norm_rgb, enhanced_float);

        // 保存反射率图R（Retinex理论中的物体固有属性）
        R = enhanced_float.clone();

        // ---- 阶段5: 后处理（去噪 + 对比度增强） ----
        cv::Mat processed = postProcess(enhanced_float);

        // ---- 阶段6: 转换回8位输出 ----
        processed.convertTo(out_lime, CV_8U, 255.0);

        return out_lime;
    }

    // 滤波和伽马校正（对光照图进行平滑+Gamma校正以改善视觉效果）
    void lime::Illum_filter(cv::Mat& img_in, cv::Mat& img_out)
    {
        // 使用高斯滤波替代均值滤波，更好保持边缘
        int ksize = 5;
        cv::GaussianBlur(img_in, img_out, cv::Size(ksize, ksize), 1.0);

        // 对滤波后的光照图进行伽马校正，使用类成员 gamma 参数（默认0.7）
        // 根据 LIME 论文：T ← T^γ，γ<1 可避免过度增强暗区
        int row = img_out.rows;
        int col = img_out.cols;
        float tem;

        for (int i = 0; i < row; i++)
            {
            for (int j = 0; j < col; j++)
                {
                    tem = pow(img_out.at<float>(i, j), gamma);
                    tem = tem <= 0.0001f ? 0.0001f : tem;
                    tem = tem > 1.0f ? 1.0f : tem;
                    img_out.at<float>(i, j) = tem;
                }
            }
    }

    //计算每个像素亮度（获取RGB三通道最大值作为初始光照估计）
    void lime::Illumination(cv::Mat& src, cv::Mat& out)
    {
        int row = src.rows, col = src.cols;

        for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < col; j++)
                {
                    // 调用 comp 函数计算亮度，并将结果存储在 out 矩阵中
                    out.at<float>(i, j) = lime::comp(src.at<cv::Vec3f>(i, j)[0],
                    src.at<cv::Vec3f>(i, j)[1],
                    src.at<cv::Vec3f>(i, j)[2]);
                }

            }
    }
}

int main(int argc, char* argv[])
{
    // ---- 解析输入参数 ----
    std::string input_path = "../data/1.bmp";
    std::string output_path = "output_enhanced.jpg";
    bool use_denoise = true;
    bool use_clahe  = false;

    if (argc >= 2) input_path  = argv[1];
    if (argc >= 3) output_path = argv[2];
    if (argc >= 4) use_denoise = (atoi(argv[3]) != 0);
    if (argc >= 5) use_clahe   = (atoi(argv[4]) != 0);

    std::cout << "========================================" << std::endl;
    std::cout << "  LIME - Low-light IMage Enhancement   " << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "输入图像: " << input_path << std::endl;
    std::cout << "输出图像: " << output_path << std::endl;
    std::cout << "去噪(BM3D近似): " << (use_denoise ? "开启" : "关闭") << std::endl;
    std::cout << "CLAHE对比度增强: " << (use_clahe  ? "开启" : "关闭") << std::endl;
    std::cout << "----------------------------------------" << std::endl;

    // ---- 读取图像 ----
    double t_total = cv::getTickCount();
    cv::Mat img_in = cv::imread(input_path);
    if (img_in.empty()) {
        std::cerr << "Error: 无法读取图像 " << input_path << std::endl;
        std::cerr << "用法: " << argv[0] << " <输入图像> [输出图像] [去噪:1/0] [CLAHE:1/0]" << std::endl;
        return -1;
    }
    std::cout << "图像尺寸: " << img_in.cols << "x" << img_in.rows
              << ", 通道数: " << img_in.channels() << std::endl;

    // ---- LIME增强处理 ----
    LIME::lime* l = new LIME::lime(img_in);

    // 配置后处理参数
    l->use_denoise = use_denoise;
    l->use_clahe   = use_clahe;
    l->gamma       = 0.7f;        // Gamma校正指数
    l->denoise_h   = 3.0f;        // 去噪强度
    l->clahe_clip  = 2.0f;        // CLAHE裁剪限幅
    l->save_intermediate = false;  // 设为true可保存中间结果调试

    cv::Mat img_out = l->enhance(img_in);

    // ---- 保存结果 ----
    cv::imwrite(output_path, img_out);

    t_total = (cv::getTickCount() - t_total) / cv::getTickFrequency();
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "增强完成! 总耗时: " << t_total << " s" << std::endl;
    std::cout << "结果已保存至: " << output_path << std::endl;
    std::cout << "========================================" << std::endl;

    delete l;
    return 0;
}
