'''
Descripttion: 
Author: xujg
version: 
Date: 2024-07-30 15:03:58
LastEditTime: 2024-07-30 15:13:53
'''
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