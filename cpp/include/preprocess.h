#ifndef PREPROCESS_H
#define PREPROCESS_H

#ifdef __cplusplus
extern "C" {
#endif

// C兼容部分的函数声明


#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

#include <opencv2/opencv.hpp>
#include <string>

// C++专用部分

// C++函数声明
cv::Mat centerCrop(const cv::Mat& img, int targetWidth=512, int targetHeight=512);
cv::Mat eulerToRotationMatrix(float roll=0.0, float pitch=0.0, float yaw=0.0);
cv::Mat distort(const std::string& pathStr);
cv::Mat preProcess(const std::string& pathStr, float contrastFactor=3);
cv::Mat VideoPrerocess(const cv::Mat& img, float contrastFactor=3, float roll=0.0, float pitch=0.0, float yaw=0.0);

#endif // __cplusplus

#endif // PREPROCESS_H
