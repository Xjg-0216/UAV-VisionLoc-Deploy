# README

## python板端部署运行步骤

### 1. 确定摄像头设备的索引
```bash
v4l2-ctl --list-devices
```

### 2. 修改配置文件

```bash
model_path: "/home/xujg/code/UAV-VisionLoc-Deploy/model/uvl_v0807.rknn"  # RKNN模型路径
img_save: true # 是否保存图像
save_path: "/home/xujg/code/UAV-VisionLoc-Deploy/python/result"  #保存图像路径
img_folder: "/home/xujg/code/UAV-VisionLoc-Deploy/data/queries"  #单图像测试，输入为摄像头时忽略
path_local_database: "/home/xujg/code/UAV-VisionLoc-Deploy/data/database/database_features.h5" # 本地数据库特征路径
logging_level: "DEBUG" # INFO， DEBUG, ...
log_path: "/home/xujg/code/UAV-VisionLoc-Deploy/python/vtl.log" # 日志路径
utm_zone_number: 50 # UTM区号
utm_zone_letter: "N" # UTM北半球
ip: "192.168.1.19" # ip
port: 16300  # port
camera_index: 82 # camera index
output_type: "latlon"  # "utm" or "latlon"
```

### 3. 执行
```bash
chmod +x run.sh
./run.sh
```









## others
部分数据集及rknn模型文件见百度网盘: [链接](https://pan.baidu.com/s/1WRp7eV-7mwwrnDMuKwNaqw?pwd=xujg)

