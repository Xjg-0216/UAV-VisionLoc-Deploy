
### install

```bash
conda create --name faiss python=3.8
conda activate faiss
conda install conda-forge::faiss-cpu
```
!!! notey
    pypi安装没有c++库文件


python 测试

```python
import time
import faiss
import numpy as np

d = 4096                          # 向量维度
nb = 30000                     # 数据库向量数量
nq = 1                      # 查询向量数量

np.random.seed(1234)            # 设置随机种子
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

index = faiss.IndexFlatL2(d)    # 构建索引
index.add(xb)                   # 添加数据库向量

t1 = time.time()
D, I = index.search(xq, 1)      # 搜索
t2 = time.time()
print("inference time: {:.4f} s".format(t2- t1))
```

>inference time: 0.0351 s


c++ 测试

```cpp
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
```

编译
> g++ -O3 faiss_demo.cpp -I/home/xujg/anaconda3/envs/faiss/include -L/home/xujg/anaconda3/envs/faiss/lib -lfaiss -Wl,-rpath,/home/xujg/anaconda3/envs/faiss/lib -o demo

* `g++`:

调用 GNU 的 C++ 编译器。
* `-O3`:

编译器优化选项，-O3 表示进行高等级优化，包括所有的 -O2 优化，以及更多循环优化和指令优化，通常用于追求最高性能。
* `faiss_demo.cpp`:
要编译的源文件。
* `-I/home/xujg/anaconda3/envs/faiss/include`:

预处理器选项，指定头文件的搜索路径。-I 后面紧跟头文件的目录路径。
这里指定了 Faiss 库的头文件所在的目录 /home/xujg/anaconda3/envs/faiss/include。
* `-L/home/xujg/anaconda3/envs/faiss/lib`:

链接器选项，指定库文件的搜索路径。-L 后面紧跟库文件的目录路径。
这里指定了 Faiss 库的库文件所在的目录 /home/xujg/anaconda3/envs/faiss/lib。
* `-lfaiss`:

链接选项，指定要链接的库。-l 后面紧跟库的名称，不包含前缀 lib 和文件后缀 .so。
这里指定链接 libfaiss.so 库。
* `-Wl,-rpath,/home/xujg/anaconda3/envs/faiss/lib`:

链接器选项，指定运行时库搜索路径。-Wl, 命令传递给链接器，-rpath 指定运行时搜索路径。
这里指定了 /home/xujg/anaconda3/envs/faiss/lib 作为运行时库的搜索路径。
* `-o demo`:

输出选项，指定生成的可执行文件的名称。-o 后面紧跟输出文件的名称。
这里指定生成名为 demo 的可执行文件。



>
> Search time: 0.0439493 seconds