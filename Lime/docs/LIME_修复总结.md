# LIME 光照增强算法 — Bug 修复与优化总结

## 概述

本文档记录了 LIME（Low-light IMage Enhancement）C++ 实现中从"结果完全错误（彩色噪声爆炸）"到"正常增强输出"的完整排查与修复过程。

---

## 一、修复清单（按优先级排序）

| # | 优先级 | 问题 | 位置 | 症状 | 修复 |
|---|--------|------|------|------|------|
| 1 | 🔴 致命 | **FFT 频域除法未生效** | `grad.cpp:229-241` SolveT | T 子问题完全无效，增强结果完全错误 | `channels[i] = channels[i] / kernel` 直接赋值回数组 |
| 2 | 🔴 致命 | **权重 EPS 用 0.001 导致软阈值失控** | `grad.cpp:48-69` ComputeWeight | W∈[2,1000]，阈值达 150，G 全部清零，正则化失效 | 改为 `+1.0`，W∈[0.5,1] |
| 3 | 🔴 致命 | **G 初始化为 ∇T_hat** | `main.cpp:41-42` | 首轮 T 子问题退化为 T=T_hat（数学恒等式），平滑不发生 | 初始化为全 0 |
| 4 | 🟠 严重 | **缺少 continuation（ρ 倍增）** | `main.cpp:57-63` | 阈值固定，前期不收敛后期不精细 | `ρ *= 2` 每轮倍增，从 1→1024 |
| 5 | 🟠 严重 | **缺少 Gamma on T** | `main.cpp:85-109` | 暗区 I/T 放大 20~100 倍，噪声爆炸 | `T ← T^0.7`，暗区放大倍数降低 2~5 倍 |
| 6 | 🟡 中等 | **权重未预去噪** | `main.cpp:26-28` | InitialMaxLight 噪声被 W 误判为边缘 | GaussianBlur(3×3) 后再算梯度 |
| 7 | 🟡 中等 | **T 未 clamp** | `grad.cpp:242-245` SolveT | FFT 可能产生负值/极小值 | clamp 到 [0.001, 1] |
| 8 | 🟢 增强 | **后处理去噪** | `main.cpp:104-109` | 传感器噪声在增强后残留 | fastNlMeansDenoisingColored (NLM) |

---

## 二、各修复详解

### 修复 1：FFT 频域除法 Bug（`grad.cpp` SolveT）

**原代码**：
```cpp
cv::Mat channels[2];
cv::split(rhs_fft, channels);
cv::Mat real_part = channels[0];
cv::Mat imag_part = channels[1];

real_part = real_part / freq_kernel;   // ❌ 创建新矩阵，channels[0] 未变！
imag_part = imag_part / freq_kernel;   // ❌ 同样

cv::merge(channels, 2, T_fft);        // 合并的是旧数据，除法白做了
```

**根因**：`cv::Mat::operator/` 返回新对象，原 `channels` 数组未被更新。`cv::merge` 时合并的是未除法的原始频域数据，导致亥姆霍兹算子 `(2I - ρ∇²)⁻¹` 完全没有作用。

**修复**：
```cpp
channels[0] = channels[0] / freq_kernel;  // 直接赋值回数组
channels[1] = channels[1] / freq_kernel;
cv::merge(channels, 2, T_fft);
```

---

### 修复 2+4：权重 EPS + Continuation 机制

**原代码**：
```cpp
// ComputeWeight: W = 1 / (|∇T| + 0.001)  →  W ∈ [2, 1000]
// main.cpp:     α = 0.05, ρ 固定
// 软阈值 = αW/ρ ≈ 0.05×1000/1 = 50（平坦区），所有梯度清零
```

**问题分析**：

| 像素区域 | \|∇T\| 典型值 | 原 W | 原阈值 (α=0.05,ρ=1) | 后果 |
|----------|---------------|------|---------------------|------|
| 平坦区 | 0.001 | ~500 | 25 | G=0 ✓ |
| 弱边缘 | 0.01 | ~91 | 4.55 | G=0 ✗ 本该保留 |
| 强边缘 | 0.1 | ~9 | 0.45 | G=0 ✗ 本该保留 |
| 强边缘 | 0.5 | ~2 | 0.10 | 勉强通过 |

结论：阈值远超梯度量级，**所有区域的 G 都被清零**，L1 正则化完全失效。

**修复方案（对齐参考实现）**：

```cpp
// ComputeWeight: W = 1 / (|∇T| + 1.0)  →  W ∈ [0.5, 1]
// main.cpp:     α = 1.0, ρ 每轮 ×2

// 软阈值 = αW/ρ ≈ 1/ρ（约）
// ρ: 1 → 2 → 4 → 8 → 16 → 32 → ... → 1024
// 阈值: 1 → 0.5 → 0.25 → 0.12 → 0.06 → 0.03 → ... → 0.001
```

这是 **continuation 策略**：
- 前期 ρ 小、阈值大 → 所有梯度被抑制 → T 被强力平滑
- 中期 ρ 增长 → 粗结构（强边缘）逐渐浮现
- 后期 ρ 大、阈值极小 → 细纹理被精细保留

---

### 修复 3：G 初始化为 0

**原代码**：`G = ∇T_hat`

当 G=∇T_hat 且 Λ=0 时，T 子问题方程退化为：
```
(2I - ρ∇²)T = 2T_hat - ρ·div(∇T_hat)
```
在频域中恰好满足 `T = T_hat`（与 reader 推导一致），第一轮迭代**完全无平滑**。

**修复**：`G = 0`，第一轮直接执行强力平滑。

---

### 修复 5：Gamma on T（LIME 论文公式 10）

**论文公式**：`R = I / T^γ`，其中 γ ∈ (0,1)

**效果**：

| T 值 | T^0.7 | I/T 放大倍数 | I/T^0.7 放大倍数 | 降噪效果 |
|------|-------|-------------|-----------------|---------|
| 0.01 | 0.040 | 100x | 25x | 4x |
| 0.05 | 0.123 | 20x | 8.1x | 2.5x |
| 0.10 | 0.200 | 10x | 5x | 2x |
| 0.30 | 0.431 | 3.3x | 2.3x | 1.4x |

暗区噪声放大被降低了 2~5 倍，这是 LIME 论文自带的噪声控制机制。

---

### 修复 6：权重预去噪

`InitialMaxLight` 的梯度用于计算 W。低光图像的传感器噪声会在 InitialMaxLight 中产生随机的梯度尖峰，被 W 误判为"边缘"从而降低该处的平滑力度。

**修复**：计算梯度前对 `InitialMaxLight` 做 `GaussianBlur(3×3, σ=1.0)`，让权重基于真实结构而非噪声。

---

### 修复 7：T 裁剪

FFT 求解过程中，数值误差或边界效应可能导致 T 出现负值或极小值。增强时 `I / T` 会在这些像素处产生极大值。

**修复**：SolveT 末尾 `cv::max(T, 0.001f, T)` + `cv::min(T, 1.0f, T)`。

---

### 修复 8：NLM 去噪

低光图像的传感器噪声是固有的，LIME 增强后噪声随亮度放大。双边滤波（bilateral filter）对传感器噪声效果有限。

**修复**：使用 OpenCV 的 `fastNlMeansDenoisingColored`（非局部均值去噪，BM3D 近似），对增强结果做后处理。

参数：`h=3.0, hColor=3.0, templateWindowSize=7, searchWindowSize=21`

---

## 三、最终参数配置

```cpp
// ADMM 参数（与参考 LIME 实现对齐）
float alpha     = 1.0f;       // L1 正则强度
float base_rho  = 1.0f;       // ADMM 惩罚系数初始值
float rho_scale = 2.0f;       // 每轮倍增
float rho_max   = 1000.0f;    // ρ 上限
int   max_iter  = 10;         // 迭代次数

// 权重
constexpr float WEIGHT_EPS = 1.0f;  // W = 1/(|∇T| + 1)

// 后处理
constexpr float GAMMA       = 0.7f;   // T^γ，γ<1 降暗区噪声
constexpr float NLM_H       = 3.0f;   // NLM 去噪强度
```

---

## 四、涉及文件

| 文件 | 变更内容 |
|------|---------|
| `src/grad.cpp` | ComputeWeight（EPS 1.0）、SolveT（FFT 除法 bug + T clamp）、SoftThreshold/ComputeDivergence/GenerateLaplacianFreqKernel（无变化，验证正确） |
| `src/main.cpp` | G=0 初始化、α & ρ 参数、ρ 倍增循环、T_hat 预去噪、Gamma on T、NLM 去噪 |
| `include/grade.h` | 接口声明（无变化） |
| `include/imageIO.h` | 无变化 |
| `src/imageIO.cpp` | 无变化 |

---

## 五、参考文献

- Guo X, Li Y, Ling H. **LIME: Low-Light Image Enhancement via Illumination Map Estimation**. IEEE TIP, 2017.
- 参考实现 1：`参考.cpp` — 基础 ADMM 框架
- 参考实现 2：`lime.cpp` — 含完整后处理管道（Gamma + NLM + CLAHE）
