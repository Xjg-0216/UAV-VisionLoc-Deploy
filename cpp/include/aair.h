#ifndef AAIR_H
#define AAIR_H
#include <cstdint>

#pragma pack(push, 1)
struct AAIR {
    uint8_t start0;   // 0x55
    uint8_t start1;   // 0xAA
    uint8_t length;   // 数据长度
    uint8_t id;       // 报文ID
    uint32_t time;    // 位置采样时间
    uint32_t actime;  // 飞机相机同步时间
    float lat;        // 目标纬度
    float lng;        // 目标经度
    float height;     // 目标高度
    float yaw;        // Yaw
    float pitch;      // Pitch
    float roll;       // Roll
    float angle;      // 航向角
    uint8_t crc;      // 包校验值固定为0xFF
};
#pragma pack(pop)

#endif // AAIR_H
