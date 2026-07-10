# 1. 套话：指明最低版本（注意：cmake和minimum之间是下划线，不是空格）
cmake_minimum_required(VERSION 3.10)

# 2. 指定工程的名字（以后用 ${CMAKE_PROJECT_NAME} 就能代替 "LimeDispose"）
project(LimeDispose)

# 3. 全局设置 C++ 标准（可选：这是全局设置，通常写在最前面）
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 4. 寻找第三方库
find_package(OpenCV REQUIRED)

# -------------------------------------------------------------
# 【非常重要】在执行任何 target_ 命令之前，必须先生成目标 (Target)！
# -------------------------------------------------------------


# 5. 构建可执行文件（先造好这辆车，才能往车上装配件！）
add_executable(${CMAKE_PROJECT_NAME} 
                src/main.cpp
                src/imageIO.cpp
                src/grad.cpp) 
# -------------------------------------------------------------
# 下面开始给刚才生成的 ${CMAKE_PROJECT_NAME} 目标“安装配件”
# -------------------------------------------------------------


# 7. 包含头文件（让代码认识 OpenCV 的函数）
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE 
                            ${OpenCV_INCLUDE_DIRS}
                            ${CMAKE_CURRENT_SOURCE_DIR}/include)

# 8. 链接库文件（把 OpenCV 和 ImGui 的实际功能绑上来）
# 可以写成两句，也可以像下面这样合并成一句，用空格隔开即可
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE 
                        ${OpenCV_LIBS})