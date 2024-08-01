/*
 * @Descripttion: 
 * @Author: xujg
 * @version: 
 * @Date: 2024-07-30 15:15:33
 * @LastEditTime: 2024-07-30 15:15:59
 */
#include <faiss/IndexFlat.h>
#include <vector>
#include <random>
#include <chrono>
#include <iostream>

int main() {
    int d = 4096;                     // 向量维度
    int nb = 30000;                // 数据库向量数量
    int nq = 1;                 // 查询向量数量

    std::vector<float> xb(nb * d);
    std::vector<float> xq(nq * d);

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib(0, 1);
    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++) {
            xb[i * d + j] = distrib(rng);
        }
        xb[i * d] += i / 1000.;
    }

    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < d; j++) {
            xq[i * d + j] = distrib(rng);
        }
        xq[i * d] += i / 1000.;
    }

    faiss::IndexFlatL2 index(d);    // 构建索引
    index.add(nb, xb.data());       // 添加数据库向量

    std::vector<faiss::idx_t> I(nq);
    std::vector<float> D(nq);

    auto start = std::chrono::high_resolution_clock::now(); // 开始计时

    index.search(nq, xq.data(), 1, D.data(), I.data()); // 搜索

    auto end = std::chrono::high_resolution_clock::now(); // 结束计时
    std::chrono::duration<double> duration = end - start;
    std::cout << "Search time: " << duration.count() << " seconds" << std::endl;

    return 0;
}
