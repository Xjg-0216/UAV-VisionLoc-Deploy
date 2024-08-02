# README

该仓库的目录结构为：

```bash
.
|-- README.md
|-- cpp
|   |-- 3rdparty
|   |   |-- librknn_api
|   |   |-- opencv
|   |   |-- rga
|   |   |-- rk_mpi_mmz
|   |   `-- stb
|   |-- CMakeLists.txt
|   |-- build
|   |   `-- build_linux_aarch64
|   |-- build.sh
|   |-- faiss-demo
|   |   |-- cpp
|   |   |-- py
|   |   `-- readme.md
|   |-- include
|   |   |-- load_database.h
|   |   |-- postprocess.h
|   |   `-- preprocess.h
|   |-- install
|   |   `-- UAV-VisionLoc-C_Linux
|   |-- run.sh
|   `-- src
|       |-- load_database.cc
|       |-- main.cc
|       |-- postprocess.cc
|       `-- preprocess.cc
|-- data
|   |-- database
|   |   `-- database_features.h5
|   `-- queries
|       |-- *.jpg
|       |-- test
|       `-- test_sample.h5
|-- docs
|-- model
|   |-- uvl_719.rknn
|   `-- uvl_731.rknn
`-- python
    |-- eval.py
    |-- main.py
    |-- memory
    |   `-- memory.md
    |-- profile
    |   |-- output.md
    |   |-- profile.py
    |   |-- profile_output.prof
    |   `-- verbose_log.txt
    |-- py_utils
    |   |-- __init__.py
    |   |-- __pycache__
    |   |-- datasets.py
    |   |-- onnx_executor.py
    |   |-- pytorch_executor.py
    |   `-- rknn_executor.py
    `-- result
        `-- *.jpg

26 directories
```

部分数据集及rknn模型文件见百度网盘: [链接](https://pan.baidu.com/s/1WRp7eV-7mwwrnDMuKwNaqw?pwd=xujg)

**查看CPU/NPU/DDR可用频率**
```bash
cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_available_frequencies
cat /sys/class/devfreq/fdab0000.npu/available_frequencies
cat /sys/class/devfreq/dmc/available_frequencies
```

**设置CPU/NPU/DDR频率**
```bash
# 更改调节器为userspace
echo userspace | sudo tee /sys/devices/system/cpu/cpufreq/policy4/scaling_governor
# 设置频率
echo 1200000 | sudo tee /sys/devices/system/cpu/cpufreq/policy4/scaling_setspeed

echo userspace | sudo tee /sys/class/devfreq/fdab0000.npu/governor
echo 800000000 | sudo tee /sys/class/devfreq/fdab0000.npu/userspace/set_freq


echo userspace | sudo tee /sys/class/devfreq/dmc/governor
echo 1596000000 | sudo tee /sys/class/devfreq/dmc/userspace/set_freq
```

**查看当前CPU/NPU/DDR频率**

```bash
cat /sys/devices/system/cpu/cpufreq/policy4/scaling_cur_freq
cat /sys/class/devfreq/fdab0000.npu/cur_freq
cat /sys/class/devfreq/dmc/cur_freq
```