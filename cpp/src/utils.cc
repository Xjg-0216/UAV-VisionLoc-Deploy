

#include "utils.h"


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

// 全局日志级别控制
static LogLevel currentLogLevel = INFO;

// 全局日志文件流
static std::ofstream log_file;

void setLogLevel(LogLevel level) {
    currentLogLevel = level;
}

void initLogFile(const std::string &filename) {
    log_file.open(filename, std::ios::out | std::ios::app);
    if (log_file.is_open()) {
        log_message(INFO, "Log file initialized.");
    }
}

void log_message(LogLevel level, const std::string &message) {
    if (level < currentLogLevel) {
        return; // 不输出低于当前日志级别的信息
    }

    std::string log_level_str;
    switch (level) {
        case DEBUG: log_level_str = "DEBUG"; break;
        case INFO: log_level_str = "INFO"; break;
        case WARN: log_level_str = "WARN"; break;
        case ERROR: log_level_str = "ERROR"; break;
    }

    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::string time_str = std::ctime(&now_time);
    time_str.pop_back();
    std::string log_entry = "[" + log_level_str + "][" + time_str + "] " + message;
    
    // 输出到终端
    std::cout << log_entry << std::endl;
    
    // 写入日志文件
    if (log_file.is_open()) {
        log_file << log_entry << std::endl;
    }
}

void closeLogFile() {
    if (log_file.is_open()) {
        log_message(INFO, "Log file closed.");
        log_file.close();
    }
}



// // 常量定义（可以根据不同的参考椭球进行修改）
// constexpr double k0 = 0.9996;

// // WGS84 椭球参数
// constexpr double WGS84_a = 6378137.0; // WGS84 长半轴
// constexpr double WGS84_f = 1.0 / 298.257223563; // WGS84 扁率
// double WGS84_e = std::sqrt(2 * WGS84_f - WGS84_f * WGS84_f); // WGS84 第一偏心率


// std::tuple<double, double> utm_to_latlon(double easting, double northing, int zone, bool is_northern_hemisphere,
//                                          double a = WGS84_a, double e = WGS84_e) {
//     double e_prime_sq = (e * e) / (1 - e * e);
//     double n = a / std::sqrt(1 - e * e * std::sin(0) * std::sin(0));
//     double t = std::tan(0) * std::tan(0);
//     double c = e_prime_sq * std::cos(0) * std::cos(0);
//     double r = a * (1 - e * e) / std::pow(1 - e * e * std::sin(0) * std::sin(0), 1.5);
//     double d = easting - 500000.0;
    
//     if (!is_northern_hemisphere) {
//         northing -= 10000000.0;
//     }
    
//     double m = northing / k0;
//     double mu = m / (a * (1 - e * e / 4 - 3 * e * e * e * e / 64 - 5 * e * e * e * e * e * e / 256));
    
//     double phi1_rad = mu + (3 * e / 2 - 27 * e * e * e / 32) * std::sin(2 * mu)
//                         + (21 * e * e / 16 - 55 * e * e * e * e / 32) * std::sin(4 * mu)
//                         + (151 * e * e * e / 96) * std::sin(6 * mu)
//                         + (1097 * e * e * e * e / 512) * std::sin(8 * mu);
    
//     n = a / std::sqrt(1 - e * e * std::sin(phi1_rad) * std::sin(phi1_rad));
//     t = std::tan(phi1_rad) * std::tan(phi1_rad);
//     c = e_prime_sq * std::cos(phi1_rad) * std::cos(phi1_rad);
//     r = a * (1 - e * e) / std::pow(1 - e * e * std::sin(phi1_rad) * std::sin(phi1_rad), 1.5);
//     d = d / (n * k0);
    
//     double lat_rad = phi1_rad - (n * std::tan(phi1_rad) / r) * (d * d / 2 
//                    - (5 + 3 * t + 10 * c - 4 * c * c - 9 * e_prime_sq) * d * d * d * d / 24
//                    + (61 + 90 * t + 298 * c + 45 * t * t - 252 * e_prime_sq - 3 * c * c) * d * d * d * d * d * d / 720);
//     double latitude = lat_rad * (180.0 / M_PI);
    
//     double lon_rad = (d - (1 + 2 * t + c) * d * d * d / 6 
//                    + (5 - 2 * c + 28 * t - 3 * c * c + 8 * e_prime_sq + 24 * t * t) * d * d * d * d * d / 120) / std::cos(phi1_rad);
//     double longitude = lon_rad * (180.0 / M_PI) + (zone > 0 ? (zone * 6 - 183) : 3);
    
//     return std::make_tuple(latitude, longitude);
// }



const double pi = 3.14159265358979323846;
const double a = 6378137.0;             // 地球长半轴 (WGS84)
const double f = 1 / 298.257223563;     // 地球扁率 (WGS84)
const double k0 = 0.9996;               // 缩放因子

void latLonToUTM(double latitude, double longitude, double& easting, double& northing) {
    // 计算 UTM zone
    int zone = int((longitude + 180) / 6) + 1;

    // 转换经纬度为弧度
    double latRad = latitude * pi / 180.0;
    double lonRad = longitude * pi / 180.0;

    // 计算中央经线
    double lonOrigin = (zone - 1) * 6 - 180 + 3; // 区域中央经度
    double lonOriginRad = lonOrigin * pi / 180.0;

    // 计算椭圆体的第二偏心率
    double e = sqrt(1 - pow(1 - f, 2));
    double e1sq = e * e / (1 - e * e);

    double N = a / sqrt(1 - pow(e * sin(latRad), 2));
    double T = pow(tan(latRad), 2);
    double C = e1sq * pow(cos(latRad), 2);
    double A = cos(latRad) * (lonRad - lonOriginRad);

    // 计算 UTM 东坐标 (Easting)
    easting = k0 * N * (A + (1 - T + C) * pow(A, 3) / 6
                      + (5 - 18 * T + T * T + 72 * C - 58 * e1sq) * pow(A, 5) / 120) + 500000.0;

    // 计算 UTM 北坐标 (Northing)
    northing = k0 * N * (latRad + (1 - T + C) * pow(A, 2) / 2 + (5 - 4 * T + 6 * C) * pow(A, 4) / 24);

    if (latitude < 0) {
        northing += 10000000.0; // 南半球的偏移量
    }
}


std::string getCurrentTimeString()
{
    // 获取当前时间
    std::time_t now = std::time(nullptr);
    std::tm *tm_info = std::localtime(&now);

    // 将时间格式化为字符串 "YYYY-MM-DD_HH-MM-SS"
    std::ostringstream oss;
    oss << std::put_time(tm_info, "%Y-%m-%d_%H-%M-%S");
    return oss.str();
}

std::string getCurrentTimeForFilename()
{
    // 获取当前时间点
    auto now = std::chrono::system_clock::now();

    // 转换为 time_t 类型
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);

    // 将 time_t 转换为 tm 结构
    std::tm *tm_info = std::localtime(&now_time);

    // 使用字符串流来格式化时间
    std::ostringstream oss;
    oss << std::put_time(tm_info, "%H-%M-%S"); // 格式为 "HH-MM-SS"

    return oss.str();
}