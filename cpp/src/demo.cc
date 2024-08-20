// Copyright (c) 2023 by Rockchip Electronics Co., Ltd. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*-------------------------------------------
                Includes
-------------------------------------------*/
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include "rknn_api.h"
#include "preprocess.h"
#include "load_database.h"
#include "postprocess.h"
#include <chrono>



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





/*-------------------------------------------
                  Main Function
-------------------------------------------*/
int main(int argc, char** argv)
{
    if (argc != 4)
    {
        printf("%s <model_path> <image_path> <database_path>\n", argv[0]);
        return -1;
    }

    char* model_path = argv[1];
    const std::string& image_path = argv[2];
    char* database_path = argv[3];


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


    // 初始化FAISS索引并添加数据库特征
    faiss::IndexFlatL2 index(database->feature_length);
    index.add(database->num_features, database->features);

    //2. 加载RKNN model
    model_len = read_data_from_file(model_path, &model);
    if (model == NULL)
    {
        printf("load model fail!\n");
        return -1;
    }
    ret = rknn_init(&context, model, model_len, 0, NULL);

    //方式2： 直接用rknn_init加载rknn模型路径
    // ret = rknn_init(&context, model_path, 0, 0, NULL);

    if (ret < 0)
    {
        printf("rknn_init fail! ret=%d\n", ret);
        return -1;
    }

    // Set NPU core mask to use multiple cores
    rknn_core_mask core_mask = RKNN_NPU_CORE_0_1_2;  // 或者使用其他选项例如 RKNN_NPU_CORE_0 | RKNN_NPU_CORE_1
    ret = rknn_set_core_mask(context, core_mask);
    if (ret < 0) {
        printf("rknn_set_core_mask failed! ret=%d\n", ret);
        return -1;
    }

    // struct timeval start, end;
    // long seconds, useconds;
    // double elapsed;
    // gettimeofday(&start, NULL);

    auto start1 = std::chrono::high_resolution_clock::now();
    //3. 使用Opencv读取图像,并进行前处理
    cv::Mat image = preProcess(image_path);
    auto end1 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> elapsed1 = end1 - start1;

    printf("preprocess time: %f ms\n", elapsed1.count());
    
    // 调用rknn_query接口查询tensor输入输出个数
    // rknn_input_output_num io_num;
    // rknn_query(context, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    // printf("model input num:%d, output num:%d\n", io_num.n_input, io_num.n_output);


    // 调用rknn_inputs_set接口设置输入数据
    rknn_input input[1];
    memset(input, 0, sizeof(rknn_input));
    input[0].index = 0;
    input[0].buf = image.data;
    input[0].size = image.rows * image.cols * image.channels() * sizeof(RKNN_TENSOR_FLOAT32);
    input[0].pass_through = 0;
    input[0].type = RKNN_TENSOR_FLOAT32;
    input[0].fmt = RKNN_TENSOR_NHWC;
    rknn_inputs_set(context, 1, input);

    //调用rknn_run接口进行模型推理
    rknn_run(context, NULL);

    // 调用rknn_outputs_get接口获取模型推理结果， 推理结果会存放到output中的buf中
    rknn_output output[1];
    memset(output, 0, sizeof(rknn_output));
    output[0].index = 0;
    output[0].is_prealloc = 0;
    output[0].want_float = 1;
    rknn_outputs_get(context, 1, output, NULL);

    //4. postprocess
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

    printf("Best position: (%.2f, %.2f)\n", best_position.first, best_position.second);
    //释放
    rknn_outputs_release(context, 1, output);
    rknn_destroy(context);

    return 0;
}
