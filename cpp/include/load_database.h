#ifndef LOAD_DATABASE_H_
#define LOAD_DATABASE_H_


#ifdef __cplusplus
extern "C" {
#endif

// 定义结构体DatabaseData
typedef struct {
    float *features; // 1D array to store [20000, 4096] features
    int num_features;
    int feature_length; // 4096
    float *utms; // 1D array to store [20000, 2] UTM coordinates
    int num_utms;
    int utm_length; // 2
} DatabaseData;

// 函数声明
DatabaseData* load_local_database(const char *path_local_database);
void free_database_data(DatabaseData* db);
#ifdef __cplusplus
}
#endif

#endif // LOAD_DATABASE_H_
