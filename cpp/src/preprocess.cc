#include "preprocess.h"

cv::Mat centerCrop(const cv::Mat& img, int targetWidth, int targetHeight) {
    int height = img.rows;
    int width = img.cols;
    int startX = width / 2 - targetWidth / 2;
    int startY = height / 2 - targetHeight / 2;
    return img(cv::Rect(startX, startY, targetWidth, targetHeight));
}

cv::Mat eulerToRotationMatrix(float roll, float pitch, float yaw) {
    cv::Mat R_x = (cv::Mat_<double>(3, 3) <<
        1, 0, 0,
        0, cos(roll), -sin(roll),
        0, sin(roll), cos(roll));

    cv::Mat R_y = (cv::Mat_<double>(3, 3) <<
        cos(pitch), 0, sin(pitch),
        0, 1, 0,
        -sin(pitch), 0, cos(pitch));

    cv::Mat R_z = (cv::Mat_<double>(3, 3) <<
        cos(yaw), -sin(yaw), 0,
        sin(yaw), cos(yaw), 0,
        0, 0, 1);
    cv::Mat R = R_z * R_y * R_x;
    return R;
}

cv::Mat distort(const std::string& pathStr) {

    // std::string pathStr(imgPath);
    cv::Mat img = cv::imread(pathStr);
    if (img.empty()) {
        printf("Could not open or find the image!\n");
        return cv::Mat();
    }

    int height = img.rows;
    int width = img.cols;
    printf("image height: %d, width: %d\n", height, width);

    // Extract roll, pitch, yaw from the filename
    // Assuming the filename format is something like "image@roll@pitch@yaw.jpg"
    size_t atPos = pathStr.find('@');
    atPos = pathStr.find('@', atPos + 1);
    atPos = pathStr.find('@', atPos + 1);

    float yaw = std::stof(pathStr.substr(atPos + 1));
    atPos = pathStr.find('@', atPos + 1);
    float pitch = std::stof(pathStr.substr(atPos + 1));
    atPos = pathStr.find('@', atPos + 1);
    float roll = std::stof(pathStr.substr(atPos + 1));

    printf("yaw: %f, pitch: %f, roll: %f\n", yaw, pitch, roll);
    

    cv::Mat R = eulerToRotationMatrix(roll, pitch, yaw);
    // Define the source points (four corners of the image)
    std::vector<cv::Point2f> srcPoints = {
        cv::Point2f(0, 0),
        cv::Point2f(width - 1, 0),
        cv::Point2f(width - 1, height - 1),
        cv::Point2f(0, height - 1)
    };
    // Apply the rotation to the source points
    std::vector<cv::Point2f> dstPoints(4);
    for (size_t i = 0; i < srcPoints.size(); ++i) {
        cv::Mat pt = (cv::Mat_<double>(2, 1) << srcPoints[i].x - width / 2.0, srcPoints[i].y - height / 2.0);
        cv::Mat dstPt = R(cv::Rect(0, 0, 2, 2)) * pt;
        dstPoints[i] = cv::Point2f(dstPt.at<double>(0) + width / 2.0, dstPt.at<double>(1) + height / 2.0);
    }
    // Compute the perspective transform matrix
    cv::Mat matrix = cv::getPerspectiveTransform(srcPoints, dstPoints);

    // Apply the perspective transformation
    cv::Mat adjustedImage;
    cv::warpPerspective(img, adjustedImage, matrix, img.size());
    return adjustedImage;
}

cv::Mat preProcess(const std::string& pathStr, float contrastFactor) {
    // Load and distort image
    cv::Mat img = distort(pathStr);


    // Center crop
    img = centerCrop(img);

    // Convert to grayscale
    cv::Mat gray;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);

    // Adjust contrast using contrastFactor
    gray.convertTo(gray, CV_32F, 1 / 255.0);
    float mean = cv::mean(gray)[0];
    gray = mean + contrastFactor * (gray - mean);

    // Clip values to [0, 1]
    cv::threshold(gray, gray, 1, 1, cv::THRESH_TRUNC);
    cv::threshold(gray, gray, 0, 0, cv::THRESH_TOZERO);

    // Convert grayscale image to 3-channel image
    // 将灰度图像复制到三个通道
    cv::Mat image;
    cv::merge(std::vector<cv::Mat>{gray, gray, gray}, image);

    // 转换图像为浮点类型
    image.convertTo(image, CV_32F);

    // 均值和标准差
    static const float meanVals[3] = {0.485, 0.456, 0.406};
    static const float stdVals[3] = {0.229, 0.224, 0.225};

    // 拆分图像为独立通道
    std::vector<cv::Mat> channels(3);
    cv::split(image, channels);

    // 归一化每个通道
    for (int i = 0; i < 3; ++i) {
        channels[i] = (channels[i] - meanVals[i]) / stdVals[i];
    }

    // 合并归一化后的通道
    cv::merge(channels, image);

    return image;
}


// int main() {
//     std::string imgPath = "/home/xujg/code/RK_demo/rknn_model_zoo-main/examples/uav_vision_loc/model/@437915@4221754@1.484649@-0.183243@0.004551@203.8990020752@39093@.jpg";
//     cv::Mat processedImg = preProcess(imgPath);
//     if (!processedImg.empty()) {
//         cv::imwrite("./demo_output.jpg", processedImg);
//     }
//     return 0;
// }