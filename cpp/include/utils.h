

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


#ifndef UTILS_H_
#define UTILS_H_


#ifdef __cplusplus
extern "C" {
#endif

int read_data_from_file(const char *path, char **out_data);

// 定义日志级别
enum LogLevel {
    DEBUG,
    INFO,
    WARN,
    ERROR
};

// 函数声明
void setLogLevel(LogLevel level);
void initLogFile(const std::string &filename);
void log_message(LogLevel level, const std::string &message);
void closeLogFile();
void latLonToUTM(double latitude, double longitude, double& easting, double& northing);
std::string getCurrentTimeString();
std::string getCurrentTimeForFilename();
#ifdef __cplusplus
}
#endif

#endif // UTILS_H_