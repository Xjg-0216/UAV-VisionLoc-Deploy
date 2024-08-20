#include <libuvc/libuvc.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <unistd.h>
#include "rknn_api.h"
#include "preprocess.h"
#include "load_database.h"
#include "postprocess.h"
#include <chrono>
#include <cstdint>
#include "aair.h"

// 全局变量用于存储最新的摄像头图像和UDP数据
std::mutex mtx;
std::condition_variable data_condition;
cv::Mat latest_image;
AAIR latest_udp_data;
bool image_ready = false;
bool udp_data_ready = false;






int read_data_from_file(const char *path, char **out_data)
{
    FILE *fp = fopen(path, "rb");
    if(fp == NULL) {
        printf("fopen %s fail!\n", path);
        return -1;
    }
    fseek(fp, 0, SEEK_END);
    int file_size = ftell(fp);
    char *data = (char *)malloc(file_size+1);
    data[file_size] = 0;
    fseek(fp, 0, SEEK_SET);
    if(file_size != fread(data, 1, file_size, fp)) {
        printf("fread %s fail!\n", path);
        free(data);
        fclose(fp);
        return -1;
    }
    if(fp) {
        fclose(fp);
    }
    *out_data = data;
    return file_size;
}

void capture_camera(uvc_device_handle_t *devh, uvc_stream_ctrl_t ctrl) {
    uvc_stream_handle_t* stream_handle;
    uvc_error_t res;

    // 打开视频流
    res = uvc_stream_open_ctrl(devh, &stream_handle, &ctrl);
    if (res != UVC_SUCCESS) {
        uvc_perror(res, "uvc_stream_open_ctrl");
        return;
    }

    // 启动视频流
    res = uvc_stream_start(stream_handle, nullptr, nullptr, 0);
    if (res != UVC_SUCCESS) {
        uvc_perror(res, "uvc_stream_start");
        uvc_stream_close(stream_handle);
        return;
    }

    while (true) {
        uvc_frame_t *frame;
        res = uvc_stream_get_frame(stream_handle, &frame, 10000); // 10秒超时
        if (res != UVC_SUCCESS) {
            uvc_perror(res, "uvc_stream_get_frame");
            // 如果发生超时或其他错误，可以决定是否退出循环或继续
            continue;
        }

        // 确保在多线程环境中使用互斥锁保护共享资源
        std::lock_guard<std::mutex> lock(mtx);

        // 将帧转换为BGR格式
        uvc_frame_t *bgr = uvc_allocate_frame(frame->width * frame->height * 3);
        if (!bgr) {
            std::cerr << "Unable to allocate bgr frame!" << std::endl;
            continue;
        }

        res = uvc_any2bgr(frame, bgr);
        if (res != UVC_SUCCESS) {
            uvc_perror(res, "uvc_any2bgr");
            uvc_free_frame(bgr);
            continue;
        }

        // 使用OpenCV显示图像
        latest_image = cv::Mat(cv::Size(bgr->width, bgr->height), CV_8UC3, bgr->data).clone();
        image_ready = true;
        uvc_free_frame(bgr);

        // 通知数据已准备好
        data_condition.notify_all();
    }

    // 停止视频流
    uvc_stream_stop(stream_handle);

    // 关闭视频流
    uvc_stream_close(stream_handle);

    // 关闭设备句柄
    uvc_close(devh);
}

void receive_udp_data(int sock) {
    char buffer[1024]; // Adjust size as needed
    while (true) {
        AAIR data;
        ssize_t recv_len = recvfrom(sock, &data, sizeof(data), 0, NULL, NULL);
        if (recv_len < 0) {
            perror("recvfrom");
            continue;
        }

        // 确保接收到的数据大小与结构体大小匹配
        if (recv_len != sizeof(data)) {
            std::cerr << "Received data size mismatch" << std::endl;
            continue;
        }

        std::lock_guard<std::mutex> lock(mtx);
        latest_udp_data = data; // 你可以在这里处理或使用结构体数据
        udp_data_ready = true;
        data_condition.notify_all();
    }
}

int main(int argc, char** argv) {
    if (argc != 3) {
        printf("%s <model_path> <database_path>\n", argv[0]);
        return -1;
    }

    char* model_path = argv[1];
    char* database_path = argv[2];

    rknn_context context;
    char *model;
    int model_len = 0;
    int ret;

    // 1.加载本地数据库
    DatabaseData* database = load_local_database(database_path);
    if (!database) {
        printf("Failed to load database.\n");
        return -1;
    }

    // 2.初始化FAISS索引并添加数据库特征
    faiss::IndexFlatL2 index(database->feature_length);
    index.add(database->num_features, database->features);

    // 3.加载RKNN模型
    model_len = read_data_from_file(model_path, &model);
    if (model == NULL) {
        printf("load model fail!\n");
        return -1;
    }

    ret = rknn_init(&context, model, model_len, 0, NULL);
    if (ret < 0) {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    // 设置NPU核心使用多个核心
    rknn_core_mask core_mask = RKNN_NPU_CORE_0_1_2;
    ret = rknn_set_core_mask(context, core_mask);
    if (ret < 0) {
        printf("rknn_set_core_mask failed! ret=%d\n", ret);
        return -1;
    }

    // 初始化libuvc上下文
    uvc_context_t *ctx;
    uvc_device_t *dev;
    uvc_device_handle_t *devh;
    uvc_stream_ctrl_t ctrl;

    uvc_error_t res = uvc_init(&ctx, NULL);
    if (res < 0) {
        uvc_perror(res, "uvc_init");
        return res;
    }

    int vendor_id = 0x2bdf;
    int product_id = 0x0102;
    res = uvc_find_device(ctx, &dev, vendor_id, product_id, NULL);
    if (res < 0) {
        uvc_perror(res, "uvc_find_device");
        return res;
    }

    res = uvc_open(dev, &devh);
    if (res < 0) {
        uvc_perror(res, "uvc_open");
        return res;
    }

    res = uvc_get_stream_ctrl_format_size(devh, &ctrl, UVC_FRAME_FORMAT_YUYV, 640, 512, 30);
    if (res < 0) {
        uvc_perror(res, "uvc_get_stream_ctrl_format_size");
        return res;
    }

    // 启动摄像头捕获线程
    std::thread camera_thread(capture_camera, devh, ctrl);

    // 配置UDP
    int udp_sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (udp_sock < 0) {
        perror("socket");
        return -1;
    }

    sockaddr_in udp_addr;
    udp_addr.sin_family = AF_INET;
    udp_addr.sin_addr.s_addr = inet_addr("192.168.1.19");
    udp_addr.sin_port = htons(12345); // Set the UDP port number

    if (bind(udp_sock, (struct sockaddr*)&udp_addr, sizeof(udp_addr)) < 0) {
        perror("bind");
        return -1;
    }

    // 启动UDP接收线程
    std::thread udp_thread(receive_udp_data, udp_sock);

    // 主线程用于处理图像和UDP数据
    while (true) {
        // 等待新数据可用
        std::unique_lock<std::mutex> lock(mtx);
        data_condition.wait(lock, [] { return image_ready && udp_data_ready; });

        if (latest_image.empty()) {
            continue;
        }
        // 获取数据并重置标志
        cv::Mat img = latest_image.clone();
        AAIR udp_data = latest_udp_data;
        image_ready = false;
        udp_data_ready = false;
        auto start1 = std::chrono::high_resolution_clock::now();
        cv::Mat image = VideoPrerocess(img, 3, udp_data.roll, udp_data.pitch, udp_data.yaw); // 预处理图像
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed1 = end1 - start1;
        printf("preprocess time: %f ms\n", elapsed1.count());

        // 设置RKNN输入
        rknn_input input[1];
        memset(input, 0, sizeof(rknn_input));
        input[0].index = 0;
        input[0].buf = image.data;
        input[0].size = image.rows * image.cols * image.channels() * sizeof(RKNN_TENSOR_FLOAT32);
        input[0].pass_through = 0;
        input[0].type = RKNN_TENSOR_FLOAT32;
        input[0].fmt = RKNN_TENSOR_NHWC;
        rknn_inputs_set(context, 1, input);

        // 进行推理
        rknn_run(context, NULL);

        // 获取推理结果
        rknn_output output[1];
        memset(output, 0, sizeof(rknn_output));
        output[0].index = 0;
        output[0].is_prealloc = 0;
        output[0].want_float = 1;
        rknn_outputs_get(context, 1, output, NULL);

        float32_t* result = (float32_t*)output[0].buf;
        auto end2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed2 = end2 - end1;
        printf("inference time: %f ms\n", elapsed2.count());

        // 后处理
        int use_best_n = 1; // 使用最相似的前N个结果
        std::pair<float32_t, float32_t> best_position = post_process(index, result, database, use_best_n);
        auto end3 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed3 = end3 - end2;
        printf("postprocess time: %f ms\n", elapsed3.count());

        

        // 获取当前时间
        auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        std::tm* tm_info = std::localtime(&now_time);

        // 格式化时间为字符串
        char time_buffer[80];
        std::strftime(time_buffer, sizeof(time_buffer), "%Y-%m-%d_%H-%M-%S", tm_info);

        // 创建文件名
        std::stringstream filename;
        filename << "image_" << time_buffer << ".jpg";
        std::cout << "Image saved as " << filename.str() << std::endl;

        // 保存图像
        cv::imwrite(filename.str(), latest_image);

        printf("predict_position: (%.2f, %.2f)\n", best_position.first, best_position.second);
        printf("real_position:(%.6f, %.6f)\n", udp_data.lat, udp_data.lng);
        // 释放输出
        rknn_outputs_release(context, 1, output);

    }

    // 停止摄像头和UDP接收线程
    camera_thread.join();
    udp_thread.join();

    // 停止视频流并释放资源
    close(udp_sock);
    uvc_close(devh);
    uvc_unref_device(dev);
    uvc_exit(ctx);
    rknn_destroy(context);
    free_database_data(database);
    return 0;
}
