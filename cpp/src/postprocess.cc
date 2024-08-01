#include "postprocess.h"

std::pair<float, float> post_process(faiss::IndexFlatL2& index, float* result, DatabaseData* db, int use_best_n) {
    std::vector<faiss::idx_t> indices(use_best_n);
    std::vector<float> distances(use_best_n);

    // 启用多线程
    // faiss::omp_set_num_threads(8);

    // 搜索最相似的特征向量
    index.search(1, result, use_best_n, distances.data(), indices.data());

    int best_idx = indices[0];


    if (use_best_n == 1) {
        return { db->utms[best_idx * 2], db->utms[best_idx * 2 + 1]};
    } 
    // else {
    //     float mean = distances[sorted_idx[0]];
    //     float sigma = distances[sorted_idx[0]] / distances[sorted_idx.back()];
    //     std::vector<float> weights(use_best_n);

    //     for (int i = 0; i < use_best_n; ++i) {
    //         float X = distances[sorted_idx[i]];
    //         weights[i] = exp(-pow(X - mean, 2) / (2 * sigma * sigma));
    //     }

    //     float weight_sum = std::accumulate(weights.begin(), weights.end(), 0.0f);
    //     for (auto& w : weights) w /= weight_sum;

    //     float x = 0, y = 0;
    //     for (int i = 0; i < use_best_n; ++i) {
    //         int idx = indices[sorted_idx[i]];
    //         y += db->utms[idx * 2] * weights[i];
    //         x += db->utms[idx * 2 + 1] * weights[i];
    //     }
    //     return {y, x};
    // }
}
