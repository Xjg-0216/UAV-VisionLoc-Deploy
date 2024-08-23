#include <libuvc/libuvc.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
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
#include "utils.h"
#include <cstdio>
#include <sys/stat.h>


// 全局变量用于存储最新的摄像头图像和UDP数据
std::mutex mtx;
std::condition_variable data_condition;
cv::Mat latest_image;
AAIR latest_udp_data;
std::atomic<bool> image_ready(false);
std::atomic<bool> udp_data_ready(false);
std::atomic<bool> stop_threads(false); // 用于停止线程的标志

void capture_camera(uvc_device_handle_t *devh, uvc_stream_ctrl_t ctrl)
{
    log_message(INFO, "Starting camera capture thread...");

    uvc_stream_handle_t *stream_handle;
    uvc_error_t res;

    // 打开视频流
    res = uvc_stream_open_ctrl(devh, &stream_handle, &ctrl);
    if (res != UVC_SUCCESS)
    {
        uvc_perror(res, "uvc_stream_open_ctrl");
        return;
    }
    log_message(DEBUG, "Video stream opened successfully.");

    // 启动视频流
    res = uvc_stream_start(stream_handle, nullptr, nullptr, 0);
    if (res != UVC_SUCCESS)
    {
        uvc_perror(res, "uvc_stream_start");
        uvc_stream_close(stream_handle);
        return;
    }
    log_message(DEBUG, "Video stream started successfully.");

    while (!stop_threads.load())
    {
        uvc_frame_t *frame;
        res = uvc_stream_get_frame(stream_handle, &frame, 10000); // 10秒超时
        if (res != UVC_SUCCESS)
        {
            log_message(DEBUG, "Failed to get frame from video stream.");
            // uvc_perror(res, "uvc_stream_get_frame");

            continue;
        }
        log_message(DEBUG, "Frame captured from camera.");

        // 确保在多线程环境中使用互斥锁保护共享资源
        std::lock_guard<std::mutex> lock(mtx);

        // 将帧转换为BGR格式
        uvc_frame_t *bgr = uvc_allocate_frame(frame->width * frame->height * 3);
        if (!bgr)
        {
            std::cerr << "Unable to allocate bgr frame!" << std::endl;
            continue;
        }

        res = uvc_any2bgr(frame, bgr);
        if (res != UVC_SUCCESS)
        {
            log_message(WARN, "Failed to convert frame to BGR format.");
            uvc_perror(res, "uvc_any2bgr");
            uvc_free_frame(bgr);
            continue;
        }

        // 使用OpenCV显示图像
        latest_image = cv::Mat(cv::Size(bgr->width, bgr->height), CV_8UC3, bgr->data).clone();
        image_ready = true;
        uvc_free_frame(bgr);
        log_message(DEBUG, "Frame cloned to OpenCV Mat and ready for processing.");

        // 通知数据已准备好
        data_condition.notify_all();
    }

    // 停止视频流
    uvc_stream_stop(stream_handle);
    log_message(DEBUG, "Video stream stopped.");

    // 关闭视频流
    uvc_stream_close(stream_handle);
    log_message(DEBUG, "Video stream handle closed.");

    // 关闭设备句柄
    uvc_close(devh);
    log_message(DEBUG, "Camera device handle closed.");
}

void receive_udp_data(int sock)
{
    log_message(INFO, "Starting UDP receive thread...");
    char buffer[1024]; // Adjust size as needed
    while (!stop_threads.load())
    {
        AAIR data;
        ssize_t recv_len = recvfrom(sock, &data, sizeof(data), 0, NULL, NULL);
        if (recv_len < 0)
        {
            log_message(WARN, "Failed to receive data from UDP socket.");
            perror("recvfrom");
            udp_data_ready = false;
            continue;
        }

        // 确保接收到的数据大小与结构体大小匹配
        if (recv_len != sizeof(data))
        {
            log_message(WARN, "Received data size mismatch!");
            std::cerr << "Received data size mismatch" << std::endl;
            continue;
        }

        std::lock_guard<std::mutex> lock(mtx);
        latest_udp_data = data;
        udp_data_ready = true;
        log_message(DEBUG, "UDP data received and stored.");

        data_condition.notify_all();
    }
}



int main(int argc, char **argv)
{
    //解析日志文件
    std::string filename = argv[1];
    Config config = parseConfig(filename);

    std::string experimentDir = "./";
    // createNewExperimentDir(experimentDir);
    // 初始化日志
    initLogFile(experimentDir);
    setLogLevel(config.log_level); // 设置日志级别

    log_message(INFO, "Program started.");

    

    // if (argc != 3)
    // {
    //     log_message(ERROR, "Usage error: Not enough arguments provided.");
    //     std::cerr << "Usage: " << argv[0] << " <model_path> <database_path>" << std::endl;
    //     return -1;
    // }
    const char* model_path = config.model_path.c_str(); 
    const char* database_path = config.database_path.c_str();
    const char* udp_net = config.udp_net.c_str();
    log_message(INFO, "Model path: " + config.model_path + ", Database path: " + config.database_path);

    rknn_context context;
    char *model;
    int model_len = 0;
    int ret;

    // 1.加载本地数据库
    DatabaseData *database = load_local_database(database_path);
    if (!database)
    {
        log_message(ERROR, "Failed to load local database.");
        return -1;
    }
    log_message(DEBUG, "Database loaded successfully.");

    // 2.初始化FAISS索引并添加数据库特征
    faiss::IndexFlatL2 index(database->feature_length);
    index.add(database->num_features, database->features);
    log_message(DEBUG, "FAISS index initialized and database features added.");

    // 3.加载RKNN模型
    model_len = read_data_from_file(model_path, &model);
    if (model == NULL)
    {
        log_message(ERROR, "Failed to load model!");
        return -1;
    }
    log_message(DEBUG, "Model loaded successfully.");

    ret = rknn_init(&context, model, model_len, 0, NULL);
    if (ret < 0)
    {
        log_message(ERROR, "Failed to initialize RKNN context!");
        return -1;
    }
    log_message(DEBUG, "RKNN context initialized successfully.");

    // 设置NPU核心使用多个核心
    rknn_core_mask core_mask = RKNN_NPU_CORE_0_1_2;
    ret = rknn_set_core_mask(context, core_mask);
    if (ret < 0)
    {
        log_message(ERROR, "Failed to set RKNN core mask!");
        return -1;
    }
    log_message(DEBUG, "RKNN core mask set successfully.");

    // 初始化libuvc上下文
    uvc_context_t *ctx;
    uvc_device_t *dev;
    uvc_device_handle_t *devh;
    uvc_stream_ctrl_t ctrl;

    uvc_error_t res = uvc_init(&ctx, NULL);
    if (res < 0)
    {
        log_message(ERROR, "Failed to initialize UVC context.");
        uvc_perror(res, "uvc_init");
        return res;
    }
    log_message(DEBUG, "UVC context initialized successfully.");

    int vendor_id = 0x2bdf;
    int product_id = 0x0102;
    res = uvc_find_device(ctx, &dev, vendor_id, product_id, NULL);
    if (res < 0)
    {
        log_message(ERROR, "Failed to find UVC device.");
        uvc_perror(res, "uvc_find_device");
        return res;
    }
    log_message(DEBUG, "UVC device found successfully.");

    res = uvc_open(dev, &devh);
    if (res < 0)
    {
        log_message(ERROR, "Failed to open UVC device.");
        uvc_perror(res, "uvc_open");
        return res;
    }
    log_message(DEBUG, "UVC device opened successfully.");

    res = uvc_get_stream_ctrl_format_size(devh, &ctrl, UVC_FRAME_FORMAT_YUYV, 640, 512, 30);
    if (res < 0)
    {
        log_message(ERROR, "Failed to get stream control format size.");
        uvc_perror(res, "uvc_get_stream_ctrl_format_size");
        return res;
    }
    log_message(DEBUG, "Stream control format size obtained successfully.");

    // 启动摄像头捕获线程
    std::thread camera_thread(capture_camera, devh, ctrl);
    // 配置UDP
    int udp_sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (udp_sock < 0)
    {
        log_message(ERROR, "Failed to create UDP socket.");
        perror("socket");
        stop_threads.store(true);
        camera_thread.join();  // 等待摄像头线程结束
        uvc_close(devh);       // 关闭设备句柄
        uvc_unref_device(dev); // 释放设备
        uvc_exit(ctx);         // 退出libuvc上下文
        rknn_destroy(context); // 销毁RKNN上下文
        free(model);           // 释放模型内存
        return -1;
    }
    log_message(DEBUG, "UDP socket created successfully.");

    sockaddr_in udp_addr;
    udp_addr.sin_family = AF_INET;
    udp_addr.sin_addr.s_addr = inet_addr(udp_net);
    udp_addr.sin_port = htons(config.udp_port); // Set the UDP port number

    if (bind(udp_sock, (struct sockaddr *)&udp_addr, sizeof(udp_addr)) < 0)
    {
        log_message(ERROR, "Failed to bind UDP socket.");
        perror("bind");
        perror("bind");
        close(udp_sock);
        stop_threads.store(true);
        camera_thread.join();  // 等待摄像头线程结束
        uvc_close(devh);       // 关闭设备句柄
        uvc_unref_device(dev); // 释放设备
        uvc_exit(ctx);         // 退出libuvc上下文
        rknn_destroy(context); // 销毁RKNN上下文
        free(model);           // 释放模型内存
        return -1;
    }
    log_message(DEBUG, "UDP socket bound successfully.");

    // 启动UDP接收线程
    std::thread udp_thread(receive_udp_data, udp_sock);

    // 创建日期文件夹
    std::string date_folder = getCurrentTimeString();
    struct stat st;
    if (stat(date_folder.c_str(), &st) != 0)
    {
        std::string mkdir_command = "mkdir -p " + date_folder;
        int res1 = system(mkdir_command.c_str()); // 创建文件夹
    }

    // 主线程用于处理图像和UDP数据
    while (!stop_threads.load())
    {
        // 等待新数据可用
        std::unique_lock<std::mutex> lock(mtx);
        data_condition.wait(lock, []
                            { return image_ready && udp_data_ready; });

        if (latest_image.empty())
        {
            log_message(WARN, "No image available, continuing...");
            continue;
        }
        // 获取数据并重置标志
        cv::Mat img = latest_image.clone();
        AAIR udp_data = latest_udp_data;
        image_ready.store(false);
        udp_data_ready.store(false);
        auto start1 = std::chrono::high_resolution_clock::now();
        cv::Mat image = VideoPrerocess(img, 3, udp_data.roll, udp_data.pitch, udp_data.yaw); // 预处理图像
        auto end1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed1 = end1 - start1;
        log_message(DEBUG, "Preprocess time: " + std::to_string(elapsed1.count()) + " ms");

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
        log_message(INFO, "RKNN inference run successfully.");

        // 获取推理结果
        rknn_output output[1];
        memset(output, 0, sizeof(rknn_output));
        output[0].index = 0;
        output[0].is_prealloc = 0;
        output[0].want_float = 1;
        rknn_outputs_get(context, 1, output, NULL);

        float32_t *result = (float32_t *)output[0].buf;
        auto end2 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed2 = end2 - end1;
        log_message(DEBUG, "Inference time: " + std::to_string(elapsed2.count()) + " ms");

        // 后处理
        int use_best_n = 1; // 使用最相似的前N个结果
        std::pair<float32_t, float32_t> best_position = post_process(index, result, database, use_best_n);
        auto end3 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed3 = end3 - end2;
        log_message(DEBUG, "Postprocess time: " + std::to_string(elapsed3.count()) + " ms");

        // // 获取当前时间
        // auto now = std::chrono::system_clock::now();
        // std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        // std::tm *tm_info = std::localtime(&now_time);

        // // 格式化日期和时间为字符串
        // char date_buffer[80], time_buffer[80];
        // std::strftime(date_buffer, sizeof(date_buffer), "%Y-%m-%d", tm_info); // 获取日期

        // // 创建日期文件夹
        // std::string date_folder = date_buffer;
        // struct stat st;
        // if (stat(date_folder.c_str(), &st) != 0)
        // {
        //     std::string mkdir_command = "mkdir -p " + date_folder;
        //     (void)system(mkdir_command.c_str()); // 创建文件夹
        // }

        // 创建文件名
        std::stringstream filename;
        std::string currentTime = getCurrentTimeForFilename();
        filename << experimentDir << "/" << date_folder << "/" << currentTime << "_" << std::defaultfloat << udp_data.lat << "_" << udp_data.lng << ".jpg";
        log_message(INFO, "Saving image as " + filename.str());

        // 保存图像
        cv::imwrite(filename.str(), img);

        log_message(INFO, "Predict position(UTM): (" + std::to_string(best_position.first) + ", " + std::to_string(best_position.second) + ")");
        log_message(INFO, "Real position: (" + std::to_string(udp_data.lat) + ", " + std::to_string(udp_data.lng) + ")");

        // 添加真实的UTM
        double easting, northing;
        latLonToUTM(udp_data.lat, udp_data.lng, easting, northing);
        log_message(INFO, "Real position(UTM): (" + std::to_string(easting) + ", " + std::to_string(northing) + ")");

        // 释放输出
        rknn_outputs_release(context, 1, output);
    }

    // 停止摄像头和UDP接收线程
    camera_thread.join();
    udp_thread.join();
    log_message(INFO, "Threads stopped and joined.");

    // 停止视频流并释放资源
    close(udp_sock);
    uvc_close(devh);
    uvc_unref_device(dev);
    uvc_exit(ctx);
    rknn_destroy(context);
    free_database_data(database);

    log_message(INFO, "Resources released, program exiting.");
    closeLogFile(); // 关闭日志文件
    return 0;
}
