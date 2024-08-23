
cmake_minimum_required(VERSION 3.4.1)

project(UAV-VisionLoc-C)


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -march=native -flto")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O3 -march=native -flto")

set(CMAKE_BUILD_TYPE Release)
# set(CMAKE_BUILD_TYPE Debug)
# rknn api
set(RKNN_API_PATH ${CMAKE_SOURCE_DIR}/3rdparty/librknn_api)  #该变量定义了RKNN API库的路径
set(LIB_ARCH aarch64)
set(RKNN_RT_LIB ${RKNN_API_PATH}/${LIB_ARCH}/librknnrt.so)  # 该变量定义了RKNN运行时库的路径
include_directories(${RKNN_API_PATH}/include)  # 添加RKNN API的头文件

# 查找HDF5库
# find_package(HDF5 REQUIRED COMPONENTS C HL)
# include_directories(${HDF5_INCLUDE_DIRS})

set(HDF5_DIR ${CMAKE_SOURCE_DIR}/3rdparty/hdf5)
set(HDF5_LIBRARIES ${HDF5_DIR}/lib/libhdf5_serial.so.103)
include_directories(${HDF5_DIR}/include)

# 查找yaml-cpp库
find_package(yaml-cpp REQUIRED)
include_directories(${yaml-cpp_INCLUDE_DIRS})

# opencv
set(OpenCV_DIR ${CMAKE_SOURCE_DIR}/3rdparty/opencv/share/OpenCV)  # 指定了OpenCV库的路径。
find_package(OpenCV REQUIRED)  # 自动查找OpenCV库，并将相关信息保存在CMake内置变量中


# faiss
# set(FAISS_API_PATH /home/xujg/miniconda3/envs/vtl2)
set(FAISS_API_PATH ${CMAKE_SOURCE_DIR}/3rdparty/faiss)
set(FAISS_INCLUDE_DIR ${FAISS_API_PATH}/include)
set(FAISS_LIBRARY_DIR ${FAISS_API_PATH}/lib)
set(FAISS_LIB ${FAISS_LIBRARY_DIR}/libfaiss.so)

include_directories(${FAISS_INCLUDE_DIR})
link_directories(${FAISS_LIBRARY_DIR})

set(LIBUVC_LIB ${CMAKE_SOURCE_DIR}/3rdparty/libuvc/lib/libuvc.so)
include_directories(${CMAKE_SOURCE_DIR}/3rdparty/libuvc)

# 指定了程序运行时查找动态链接库的路径
set(CMAKE_INSTALL_RPATH "${HDF5_DIR}/lib;${CMAKE_SOURCE_DIR}/3rdparty/libuvc/lib;${FAISS_LIBRARY_DIR};lib")
# set(CMAKE_INSTALL_RPATH "${CMAKE_SOURCE_DIR}/require")

include_directories(${CMAKE_SOURCE_DIR}/include)

# 创建一个可执行文件
add_executable(${PROJECT_NAME}
    src/main.cc src/utils.cc src/preprocess.cc src/load_database.cc src/postprocess.cc 
)

# 指定工程需要链接的库文件
target_link_libraries(${PROJECT_NAME}
  ${RKNN_RT_LIB}
  ${OpenCV_LIBS}
  ${HDF5_LIBRARIES}
  ${FAISS_LIB}
  ${LIBUVC_LIB}
  yaml-cpp
)

# 指定了程序安装的路径
set(CMAKE_INSTALL_PREFIX ${CMAKE_SOURCE_DIR}/install/${PROJECT_NAME}_${CMAKE_SYSTEM_NAME})

# 拷贝可执行程序和需要用的库以及后面测试要用到的model测试文件
install(TARGETS ${PROJECT_NAME} DESTINATION ./)
# install(DIRECTORY ./model DESTINATION ./)
install(FILES ${CMAKE_SOURCE_DIR}/../model/uvl_v0807.rknn DESTINATION ./model)
install(FILES ${CMAKE_SOURCE_DIR}/../data/database/database_features.h5 DESTINATION ./model)
install(FILES ${CMAKE_SOURCE_DIR}/../data/queries/@437073@4220795@0.023907@0.027543@-0.013408@381.7430114746@35075@.jpg DESTINATION ./model)
install(PROGRAMS ${RKNN_RT_LIB} DESTINATION lib)
