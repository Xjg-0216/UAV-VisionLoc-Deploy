```bash
Fri Aug  2 10:57:58 2024    /home/xujg/code/UAV-VisionLoc-Deploy/python/profile_output.prof

         415788 function calls (409918 primitive calls) in 2.088 seconds

   Ordered by: cumulative time
   List reduced from 2801 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    287/1    0.008    0.000    2.089    2.089 {built-in method builtins.exec}
        1    0.001    0.001    2.089    2.089 main.py:1(<module>)
   341/10    0.004    0.000    1.005    0.101 <frozen importlib._bootstrap>:986(_find_and_load)
   341/10    0.002    0.000    1.005    0.100 <frozen importlib._bootstrap>:956(_find_and_load_unlocked)
   323/11    0.003    0.000    1.001    0.091 <frozen importlib._bootstrap>:650(_load_unlocked)
    223/9    0.001    0.000    0.999    0.111 <frozen importlib._bootstrap_external>:837(exec_module)
   483/13    0.001    0.000    0.998    0.077 <frozen importlib._bootstrap>:211(_call_with_frames_removed)
        1    0.000    0.000    0.926    0.926 main.py:36(setup_model)
    77/45    0.000    0.000    0.594    0.013 <frozen importlib._bootstrap_external>:1172(exec_module)
    77/45    0.019    0.000    0.594    0.013 {built-in method _imp.exec_dynamic}


Fri Aug  2 10:57:58 2024    /home/xujg/code/UAV-VisionLoc-Deploy/python/profile_output.prof

         415788 function calls (409918 primitive calls) in 2.088 seconds

   Ordered by: internal time
   List reduced from 2801 to 10 due to restriction <10>

   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
        9    0.351    0.039    0.351    0.039 {built-in method posix.read}
        2    0.262    0.131    0.262    0.131 /home/xujg/miniconda3/envs/vtl/lib/python3.8/site-packages/h5py/_hl/dataset.py:742(__getitem__)
        1    0.237    0.237    0.237    0.237 {built-in method faiss._swigfaiss.IndexFlatCodes_add}
        9    0.124    0.014    0.124    0.014 {built-in method _posixsubprocess.fork_exec}
    77/75    0.086    0.001    0.090    0.001 {built-in method _imp.create_dynamic}
        1    0.081    0.081    0.082    0.082 /home/xujg/miniconda3/envs/vtl/lib/python3.8/site-packages/rknnlite/api/rknn_lite.py:185(inference)
      223    0.056    0.000    0.056    0.000 {built-in method marshal.loads}
        1    0.049    0.049    0.049    0.049 {built-in method faiss._swigfaiss.IndexFlat_search}
        1    0.045    0.045    0.310    0.310 /home/xujg/miniconda3/envs/vtl/lib/python3.8/site-packages/rknnlite/api/rknn_lite.py:107(init_runtime)
      232    0.044    0.000    0.044    0.000 {method 'read' of '_io.BufferedReader' objects}
```

从性能分析结果来看，我们可以得出以下几个结论：

### 总体概述

- 总共进行了 415,788 次函数调用，其中 409,918 次是原始调用。
- 总的运行时间是 2.088 秒。

### 主要瓶颈

我们从两个方面来分析：**累计时间** 和 **内部时间**。

#### 1. 累计时间（Cumulative Time）

累计时间表示包括被调用函数在内的所有时间消耗。前10个占用最多累计时间的函数调用是：

1. **`{built-in method builtins.exec}`**：这是运行 Python 代码的内置方法，累计时间为 2.089 秒。
2. **`main.py:1(<module>)`**：这是主模块的入口，累计时间为 2.089 秒。
3. **`<frozen importlib._bootstrap>:986(_find_and_load)`** 和其他 `_find_and_load_unlocked`，`_load_unlocked`，`exec_module`，这些都是模块加载相关的操作，总计约 1 秒。
4. **`main.py:36(setup_model)`**：`setup_model`函数累计时间为 0.926 秒，表示这是一个重要的性能瓶颈。
5. **`<frozen importlib._bootstrap_external>:1172(exec_module)`** 和 `exec_dynamic`，这些也是模块加载相关的操作，总计约 0.594 秒。

#### 2. 内部时间（Internal Time）

内部时间表示仅函数本身的时间消耗，不包括被调用的函数。前10个占用最多内部时间的函数调用是：

1. **`{built-in method posix.read}`**：内部时间为 0.351 秒。
2. **`/home/xujg/miniconda3/envs/vtl/lib/python3.8/site-packages/h5py/_hl/dataset.py:742(__getitem__)`**：内部时间为 0.262 秒。
3. **`{built-in method faiss._swigfaiss.IndexFlatCodes_add}`**：内部时间为 0.237 秒。
4. **`{built-in method _posixsubprocess.fork_exec}`**：内部时间为 0.124 秒。
5. **`{built-in method _imp.create_dynamic}`**：内部时间为 0.086 秒。
6. **`/home/xujg/miniconda3/envs/vtl/lib/python3.8/site-packages/rknnlite/api/rknn_lite.py:185(inference)`**：内部时间为 0.081 秒。
7. **`{built-in method marshal.loads}`**：内部时间为 0.056 秒。
8. **`{built-in method faiss._swigfaiss.IndexFlat_search}`**：内部时间为 0.049 秒。
9. **`/home/xujg/miniconda3/envs/vtl/lib/python3.8/site-packages/rknnlite/api/rknn_lite.py:107(init_runtime)`**：内部时间为 0.045 秒，总时间 0.310 秒。
10. **`{method 'read' of '_io.BufferedReader' objects}`**：内部时间为 0.044 秒。



