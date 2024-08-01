# README

该仓库的目录结构为：

```bash
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

