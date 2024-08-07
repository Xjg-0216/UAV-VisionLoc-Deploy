import os
import cv2
import sys
import argparse
import h5py
import faiss
import numpy as np
from time import time
from tqdm import tqdm
import logging
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vtl.log'),  # 将日志写入文件
        logging.StreamHandler()  # 将日志输出到控制台
    ]
)
logger = logging.getLogger()

def img_check(path):
    """检查路径是否是图像文件"""
    logger.debug("Checking if path is an image: %s", path)
    img_types = ['.jpg', '.jpeg', '.png', '.bmp']
    return any(path.lower().endswith(img_type) for img_type in img_types)

def draw(img, position):
    """在图像上绘制位置信息"""
    logger.debug("Drawing position: %s on image", position)
    cv2.putText(img, '{}'.format(position), (0, img.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

class RKInfer:
    def __init__(self, config):
        self.config = config
        self.load_local_database()
        self.faiss_index = faiss.IndexFlatL2(config["features_dim"])
        self.faiss_index.add(self.database_features)
        self.setup_model()

    def setup_model(self):
        """加载并设置模型"""
        logger.info("Setting up model from path: %s", self.config["model_path"])
        model_path = self.config["model_path"]
        from py_utils.rknn_executor import RKNN_model_container 
        self.model = RKNN_model_container(model_path, False)

    def model_inference(self, input_data):
        """模型推理"""
        logger.debug("Starting model inference")
        t_start = time()
        outputs = self.model.run([input_data])
        t_infer = time()
        position = self.post_process(outputs[0])
        t_end = time()
        logger.debug("Inference time: {:.4f} s".format(t_infer - t_start))
        logger.debug("Post-processing time: {:.4f} s".format(t_end - t_infer))
        return position

    def post_process(self, result):
        """后处理模型输出，获取最佳位置"""
        logger.debug("Starting post processing")
        distances, predictions = self.faiss_index.search(result, max(self.config["recall_values"]))
        distance = distances[0]
        prediction = predictions[0]
        sort_idx = np.argsort(distance)
        best_position = self._calculate_best_position(distance, prediction, sort_idx)
        return best_position

    def _calculate_best_position(self, distance, prediction, sort_idx):
        """根据距离和预测结果计算最佳位置"""
        if self.config["use_best_n"] == 1:
            return self.database_utms[prediction[sort_idx[0]]]
        if distance[sort_idx[0]] == 0:
            return self.database_utms[prediction[sort_idx[0]]]
        
        mean = distance[sort_idx[0]]
        sigma = distance[sort_idx[0]] / distance[sort_idx[-1]]
        X = np.array(distance[sort_idx[:self.config["use_best_n"]]]).reshape((-1,))
        weights = np.exp(-np.square(X - mean) / (2 * sigma ** 2))  # 高斯分布权重
        weights /= np.sum(weights)
        return np.average(self.database_utms[prediction[sort_idx[:self.config["use_best_n"]]]], axis=0, weights=weights)

    def load_local_database(self):
        """加载本地数据库特征和位置"""
        if os.path.exists(self.config["path_local_database"]):
            logger.info("Loading database features and utms")
            with h5py.File(self.config["path_local_database"], 'r') as hf:
                self.database_features = hf['database_features'][:]
                self.database_utms = hf['database_utms'][:]
        else:
            logger.error("Database features not found")
            print("Please extract database features first.")
            sys.exit()

def center_crop(img, target_width=512, target_height=512):
    """中心裁剪图像"""
    logger.debug("Center cropping image")
    height, width, _ = img.shape
    start_x = width // 2 - target_width // 2
    start_y = height // 2 - target_height // 2
    return img[start_y:start_y + target_height, start_x:start_x + target_width]

def euler_to_rotation_matrix(roll, pitch, yaw):
    """将欧拉角转换为旋转矩阵"""
    logger.debug("Converting Euler angles to rotation matrix")
    R_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return np.dot(R_z, np.dot(R_y, R_x))

def distort(img_path):
    """对图像进行扭曲变换"""
    logger.debug("Distorting image: %s", img_path)
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    yaw, pitch, roll = map(np.float32, img_path.split("@")[3:6])
    R = euler_to_rotation_matrix(roll, pitch, yaw)
    src_points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype='float32')
    dst_points = np.dot(src_points - np.array([width / 2, height / 2]), R[:2, :2].T) + np.array([width / 2, height / 2])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points.astype('float32'))
    return cv2.warpPerspective(img, matrix, (width, height))

def pre_process(img, contrast_factor=3):
    """预处理图像"""
    logger.debug("Pre-processing image")
    logger.debug("h, w: {},{}".format(img.shape[0], img.shape[1]))
    img = center_crop(img)
    img = cv2.resize(img,dsize=(512, 512))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_float = img_gray.astype(np.float32) / 255.0
    mean = np.mean(img_float)
    img_contrast = mean + contrast_factor * (img_float - mean)
    img_clipped = np.clip(img_contrast, 0, 1)
    img_rgb = np.repeat(img_clipped[:, :, np.newaxis], 3, axis=2)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_normalized = (img_rgb - mean) / std
    return img_normalized.astype(np.float32)

def process_images(rk_engine, img_list, config):
    """处理图像列表，进行推理"""
    for i in tqdm(range(len(img_list))):
        logger.info('Infer {}/{}'.format(i + 1, len(img_list)))
        img_name = img_list[i]
        img_path = os.path.join(config["img_folder"], img_name)
        if not os.path.exists(img_path):
            logger.warning("%s is not found", img_name)
            continue
        t1 = time()
        img = cv2.imread(img_path)
        input_data = pre_process(img)
        t2 = time()
        input_data = np.expand_dims(input_data, 0)
        position = rk_engine.model_inference(input_data)
        logger.debug("Pre-processing time: {:.4f} s".format(t2 - t1))
        logger.info("Position: %s", position)
        handle_result(img, img_name, position, config)

def process_camera(rk_engine, config):
    """处理摄像头输入"""
    cap = cv2.VideoCapture('/dev/video81')
    if not cap.isOpened():
        logger.error("Error: Could not open camera.")
        sys.exit()
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Error: Could not read frame from camera.")
            break

        t1 = time()
        input_data = pre_process(frame)
        t2 = time()
        input_data = np.expand_dims(input_data, 0)
        position = rk_engine.model_inference(input_data)
        logger.debug("Pre-processing time: {:.4f} s".format(t2 - t1))
        logger.info("Position: %s", position)
        # handle_result(frame, "camera_frame", position, config)

        if config["img_show"]:
            cv2.imshow("Camera Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

def handle_result(img, img_name, position, config):
    """处理并保存结果图像"""
    if config["img_show"] or config["img_save"]:
        logger.info('IMG: %s', img_name)
        draw(img, position)
        if config["img_save"]:
            if not os.path.exists(config["save_path"]):
                os.makedirs(config["save_path"])
            result_path = os.path.join(config["save_path"], img_name)
            cv2.imwrite(result_path, img)
            logger.info('Position result saved to %s', result_path)
        if config["img_show"]:
            cv2.imshow("Full post process result", img)
            cv2.waitKey(1)

def main():


    # 解析 config.json 文件
    with open('/home/xujg/code/UAV-VisionLoc-Deploy/config.json', 'r') as config_file:
        config = json.load(config_file)

    rk_engine = RKInfer(config)

    if config["use_camera"]:
        process_camera(rk_engine, config)
    else:
        file_list = sorted(os.listdir(config["img_folder"]))
        img_list = [path for path in file_list if img_check(path)]
        process_images(rk_engine, img_list, config)

if __name__ == '__main__':
    main()
