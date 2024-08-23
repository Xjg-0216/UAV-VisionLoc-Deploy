

#include <cstdio>  
#include <cstdlib> 
#include <string>
#include <fstream>
#include <iostream>
#include <chrono>
#include <ctime>
#include <cmath>
#include <tuple>
#include <iomanip>
#include <sstream>
#include <yaml-cpp/yaml.h>
#include <unistd.h>
#include <sys/stat.h>
#include <cstring> 
#ifndef UTILS_H_
#define UTILS_H_


#ifdef __cplusplus
extern "C" {
#endif

struct Config {
    std::string log_level;
    std::string model_path;
    std::string database_path;
    std::string udp_net;
    int udp_port;
};

int read_data_from_file(const char *path, char **out_data);

// 定义日志级别
enum LogLevel {
    DEBUG,
    INFO,
    WARN,
    ERROR
};

// 函数声明
Config parseConfig(const std::string& filename);
void setLogLevel(const std::string& levelStr);
void initLogFile(std::string& experimentDir);
void log_message(LogLevel level, const std::string &message);
void closeLogFile();
void latLonToUTM(double latitude, double longitude, double& easting, double& northing);
std::string getCurrentTimeString();
std::string getCurrentTimeForFilename();
void createNewExperimentDir(std::string& experimentDir);
#ifdef __cplusplus
}
#endif

#endif // UTILS_H_