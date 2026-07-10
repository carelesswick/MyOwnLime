# ============================================================================
#  LIME — Low-light IMage Enhancement
# ============================================================================
cmake_minimum_required(VERSION 3.10)
project(LimeDispose)

# ---------------------------------------------------------------------------
# 1. 全局 C++ 标准
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# 2. 第三方库
# ---------------------------------------------------------------------------
set(OpenCV_DIR "/usr/lib/x86_64-linux-gnu/cmake/opencv4")
find_package(OpenCV REQUIRED)

# ---------------------------------------------------------------------------
# 3. 构建目标
# ---------------------------------------------------------------------------
add_executable(${CMAKE_PROJECT_NAME}
    src/main.cpp
    src/imageIO.cpp
    src/grad.cpp
)

# ---------------------------------------------------------------------------
# 4. 头文件路径
# ---------------------------------------------------------------------------
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ---------------------------------------------------------------------------
# 5. 链接库
# ---------------------------------------------------------------------------
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
)

# ---------------------------------------------------------------------------
# 6. SIMD 指令集（ComputeGradientX/Y 使用了 AVX2 intrinsics）
#    -mavx2 : 启用 AVX2 指令集（256-bit 向量寄存器，一次处理 8 个 float）
#    -mfma  : 启用 FMA（融合乘加），a×b+c 一条指令完成，加速浮点运算
#
#    备选方案：
#    -march=native : 让编译器自动检测当前 CPU 支持的所有指令集（更省心）
#                    若使用此选项，下面两个可以删掉
# ---------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE -mavx2 -mfma)
elseif(MSVC)
    target_compile_options(${CMAKE_PROJECT_NAME} PRIVATE /arch:AVX2)
endif()
