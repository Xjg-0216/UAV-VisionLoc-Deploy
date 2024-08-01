#include "hdf5.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include "load_database.h"

DatabaseData *load_local_database(const char *path_local_database)
{
    if (access(path_local_database, F_OK) != -1)
    {
        printf("Loading Database feature and UTMs...\n");

        // 打开HDF5文件
        hid_t file_id = H5Fopen(path_local_database, H5F_ACC_RDONLY, H5P_DEFAULT);
        if (file_id < 0)
        {
            printf("Error opening file: %s\n", path_local_database);
            return NULL;
        }

        // 读取 database_features
        hid_t features_dataset_id = H5Dopen2(file_id, "database_features", H5P_DEFAULT);
        if (features_dataset_id < 0)
        {
            printf("Error opening database_features dataset\n");
            H5Fclose(file_id);
            return NULL;
        }

        hid_t features_dataspace_id = H5Dget_space(features_dataset_id);
        hsize_t features_dims[2];
        H5Sget_simple_extent_dims(features_dataspace_id, features_dims, NULL);
        int num_features = (int)features_dims[0];
        int feature_length = (int)features_dims[1];
        float *database_features = (float *)malloc(num_features * feature_length * sizeof(float));
        if (database_features == NULL)
        {
            printf("Error allocating memory for database_features\n");
            H5Sclose(features_dataspace_id);
            H5Dclose(features_dataset_id);
            H5Fclose(file_id);
            return NULL;
        }
        H5Dread(features_dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, database_features);

        // 读取 database_utms
        hid_t utms_dataset_id = H5Dopen2(file_id, "database_utms", H5P_DEFAULT);
        if (utms_dataset_id < 0)
        {
            printf("Error opening database_utms dataset\n");
            free(database_features);
            H5Sclose(features_dataspace_id);
            H5Dclose(features_dataset_id);
            H5Fclose(file_id);
            return NULL;
        }

        hid_t utms_dataspace_id = H5Dget_space(utms_dataset_id);
        hsize_t utms_dims[2];
        H5Sget_simple_extent_dims(utms_dataspace_id, utms_dims, NULL);
        int num_utms = (int)utms_dims[0];
        int utm_length = (int)utms_dims[1];
        float *database_utms = (float *)malloc(num_utms * utm_length * sizeof(float));
        if (database_utms == NULL)
        {
            printf("Error allocating memory for database_utms\n");
            free(database_features);
            H5Sclose(utms_dataspace_id);
            H5Dclose(utms_dataset_id);
            H5Sclose(features_dataspace_id);
            H5Dclose(features_dataset_id);
            H5Fclose(file_id);
            return NULL;
        }
        H5Dread(utms_dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, database_utms);

        printf("Loaded %d database features and %d UTM entries.\n", num_features, num_utms);

        // 释放
        H5Sclose(features_dataspace_id);
        H5Dclose(features_dataset_id);
        H5Sclose(utms_dataspace_id);
        H5Dclose(utms_dataset_id);
        H5Fclose(file_id);

        // 分配和返回数据结构
        DatabaseData *data = (DatabaseData *)malloc(sizeof(DatabaseData));
        if (data == NULL)
        {
            printf("Error allocating memory for DatabaseData\n");
            free(database_features);
            free(database_utms);
            return NULL;
        }
        data->features = database_features;
        data->num_features = num_features;
        data->feature_length = feature_length;
        data->utms = database_utms;
        data->num_utms = num_utms;
        data->utm_length = utm_length;

        return data;
    }
    else
    {
        printf("File not found: %s\n", path_local_database);
        return NULL;
    }
}

// int main(int argc, char *argv[])
// {
//     if (argc != 2) {
//         printf("Usage: %s <path_to_hdf5_file>\n", argv[0]);
//         return EXIT_FAILURE;
//     }

//     const char *path = argv[1];
//     DatabaseData *data = load_local_database(path);
//     if (data)
//     {
//         // 使用 data->features, data->utms 等
//         // 完成工作后释放内存
//         free(data->features);
//         free(data->utms);
//         free(data);
//     }
//     return 0;
// }
