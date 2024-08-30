

#include "utils.h"



Config parseConfig(const std::string& filename) {
    Config config;

    try {
        // 加载 YAML 文件
        YAML::Node node = YAML::LoadFile(filename);

        // 提取配置信息
        if (node["log_level"]) {
            config.log_level = node["log_level"].as<std::string>();
        }

        if (node["model_path"]) {
            config.model_path = node["model_path"].as<std::string>();
        }

        if (node["database_path"]) {
            config.database_path = node["database_path"].as<std::string>();
        }

        if (node["udp_net"]) {
            config.udp_net = node["udp_net"].as<std::string>();
        }

        if (node["udp_port"]) {
            config.udp_port = node["udp_port"].as<int>();
        }
    } catch (const YAML::Exception& e) {
        std::cerr << "YAML Exception: " << e.what() << std::endl;
    }

    return config;
}




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

void setLogLevel(const std::string& levelStr) {
    if (levelStr == "DEBUG") currentLogLevel = DEBUG;
    if (levelStr == "INFO") currentLogLevel = INFO;
    if (levelStr == "WARNING") currentLogLevel = WARN;
    if (levelStr == "ERROR") currentLogLevel = ERROR;
}

void initLogFile(std::string& experimentDir) {
    std::string currentTime = getCurrentTimeString();
    // 使用当前时间字符串生成日志文件名
    std::string logFileName = experimentDir + "/" + currentTime + ".log";

    log_file.open(logFileName, std::ios::out | std::ios::app);
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



// WGS84 椭球体参数
const double WGS84_a = 6378137.0;                // 长半轴
const double WGS84_e = 0.081819190842622;        // 偏心率
const double k0 = 0.9996;                        // UTM 投影的尺度因子
const double e1 = (1 - sqrt(1 - WGS84_e * WGS84_e)) / (1 + sqrt(1 - WGS84_e * WGS84_e)); // e1

void utm_to_latlon(double easting, double northing, int zone, bool is_northern_hemisphere, double& lat, double& lon) {
    double e_prime_sq = (WGS84_e * WGS84_e) / (1 - WGS84_e * WGS84_e);
    
    if (!is_northern_hemisphere) {
        northing -= 10000000.0;
    }
    
    double m = northing / k0;
    double mu = m / (WGS84_a * (1 - WGS84_e * WGS84_e / 4.0 - 3 * WGS84_e * WGS84_e * WGS84_e * WGS84_e / 64.0 - 5 * WGS84_e * WGS84_e * WGS84_e * WGS84_e * WGS84_e * WGS84_e / 256.0));
    
    double phi1_rad = mu + (3 * e1 / 2 - 27 * e1 * e1 * e1 / 32) * sin(2 * mu)
                        + (21 * e1 * e1 / 16 - 55 * e1 * e1 * e1 * e1 / 32) * sin(4 * mu)
                        + (151 * e1 * e1 * e1 / 96) * sin(6 * mu);
    
    double n = WGS84_a / sqrt(1 - WGS84_e * WGS84_e * sin(phi1_rad) * sin(phi1_rad));
    double t = tan(phi1_rad) * tan(phi1_rad);
    double c = e_prime_sq * cos(phi1_rad) * cos(phi1_rad);
    double r = WGS84_a * (1 - WGS84_e * WGS84_e) / pow(1 - WGS84_e * WGS84_e * sin(phi1_rad) * sin(phi1_rad), 1.5);
    double d = (easting - 500000.0) / (n * k0);
    
    double lat_rad = phi1_rad - (n * tan(phi1_rad) / r) * (d * d / 2 
                   - (5 + 3 * t + 10 * c - 4 * c * c - 9 * e_prime_sq) * d * d * d * d / 24
                   + (61 + 90 * t + 298 * c + 45 * t * t - 252 * e_prime_sq - 3 * c * c) * d * d * d * d * d * d / 720);
    lat = lat_rad * (180.0 / M_PI);
    
    double lon_rad = (d - (1 + 2 * t + c) * d * d * d / 6 
                   + (5 - 2 * c + 28 * t - 3 * c * c + 8 * e_prime_sq + 24 * t * t) * d * d * d * d * d / 120) / cos(phi1_rad);
    lon = lon_rad * (180.0 / M_PI) + (zone * 6 - 183);
}

const double pi = 3.14159265358979323846;
const double a = 6378137.0;             // 地球长半轴 (WGS84)
const double f = 1 / 298.257223563;     // 地球扁率 (WGS84)
// const double k0 = 0.9996;               // 缩放因子

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

    // 计算梅尔卡托投影弧长M
    double M = a * ((1 - e * e / 4 - 3 * e * e * e * e / 64 - 5 * e * e * e * e * e * e / 256) * latRad
                - (3 * e * e / 8 + 3 * e * e * e * e / 32 + 45 * e * e * e * e * e * e / 1024) * sin(2 * latRad)
                + (15 * e * e * e * e / 256 + 45 * e * e * e * e * e * e / 1024) * sin(4 * latRad)
                - (35 * e * e * e * e * e * e / 3072) * sin(6 * latRad));

    // 计算 UTM 东坐标 (Easting)
    easting = k0 * N * (A + (1 - T + C) * pow(A, 3) / 6
                      + (5 - 18 * T + T * T + 72 * C - 58 * e1sq) * pow(A, 5) / 120) + 500000.0;

    // 计算 UTM 北坐标 (Northing)
    northing = k0 * (M + N * tan(latRad) * (pow(A, 2) / 2
                     + (5 - T + 9 * C + 4 * C * C) * pow(A, 4) / 24
                     + (61 - 58 * T + T * T + 600 * C - 330 * e1sq) * pow(A, 6) / 720));

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

    // 获取毫秒部分
    auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()) % 1000;
    // 使用字符串流来格式化时间
    std::ostringstream oss;
    oss << std::put_time(tm_info, "%H-%M-%S"); // 格式为 "HH-MM-SS"
    oss << '-' << std::setfill('0') << std::setw(3) << milliseconds.count(); // 添加毫秒部分
    return oss.str();
}


void createNewExperimentDir(std::string& experimentDir) {
    int i = 1;
    std::string baseDir;
    std::string newDir;
    
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != nullptr) {
        // 当前工作目录
        std::string currentDir = cwd;

        // 获取上一级目录
        size_t lastSlashPos = currentDir.find_last_of('/');
        if (lastSlashPos != std::string::npos) {
            baseDir = currentDir.substr(0, lastSlashPos); // 上一级目录
        } else {
            std::cerr << "Failed to determine parent directory." << std::endl;
            return;
        }

        while (true) {
            newDir = baseDir + "/exp" + std::to_string(i);
            if (mkdir(newDir.c_str(), 0755) == 0) { // 目录创建成功
                experimentDir = newDir;
                break;
            }
            if (errno != EEXIST) { // 目录创建失败且非已存在错误
                std::cerr << "Failed to create experiment directory: " << strerror(errno) << std::endl;
                break;
            }
            ++i;
        }
    } else {
        std::cerr << "Failed to get current directory: " << strerror(errno) << std::endl;
    }
}
