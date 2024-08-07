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

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as config_file:
        return json.load(config_file)

def configure_logging(config):
    """配置日志"""
    log_level = config['logging_level'].upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(config["log_path"]),  # 将日志写入文件
            logging.StreamHandler()  # 将日志输出到控制台
        ]
    )
    return logging.getLogger()

def img_check(path):
    """检查路径是否是图像文件"""
    img_types = ['.jpg', '.jpeg', '.png', '.bmp']
    return any(path.lower().endswith(img_type) for img_type in img_types)

def draw(img, position):
    """在图像上绘制位置信息"""
    cv2.putText(img, '{}'.format(position), (0, 512 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

class RKInfer:
    def __init__(self, args):
        self.args = args
        self.load_local_database()
        self.faiss_index = faiss.IndexFlatL2(args.features_dim)
        self.faiss_index.add(self.database_features)
        self.setup_model()

    def setup_model(self):
        """加载并设置模型"""
        from py_utils.rknn_executor import RKNN_model_container 
        self.model = RKNN_model_container(self.args.model_path, False)

    def model_inference(self, input_data):
        """模型推理"""
        t_start = time()
        outputs = self.model.run([input_data])
        position = self.post_process(outputs[0])
        t_end = time()
        logger.debug("Inference time: {:.4f} s".format(t_end - t_start))
        return position

    def post_process(self, result):
        """后处理模型输出，获取最佳位置"""
        distances, predictions = self.faiss_index.search(result, max(self.args.recall_values))
        sort_idx = np.argsort(distances[0])
        best_position = self._calculate_best_position(distances[0], predictions[0], sort_idx)
        return best_position

    def _calculate_best_position(self, distance, prediction, sort_idx):
        """根据距离和预测结果计算最佳位置"""
        if self.args.use_best_n == 1:
            return self.database_utms[prediction[sort_idx[0]]]
        if distance[sort_idx[0]] == 0:
            return self.database_utms[prediction[sort_idx[0]]]
        
        mean = distance[sort_idx[0]]
        sigma = distance[sort_idx[0]] / distance[sort_idx[-1]]
        X = np.array(distance[sort_idx[:self.args.use_best_n]]).reshape((-1,))
        weights = np.exp(-np.square(X - mean) / (2 * sigma ** 2))  # 高斯分布权重
        weights /= np.sum(weights)
        return np.average(self.database_utms[prediction[sort_idx[:self.args.use_best_n]]], axis=0, weights=weights)

    def load_local_database(self):
        """加载本地数据库特征和位置"""
        if os.path.exists(self.args.path_local_database):
            with h5py.File(self.args.path_local_database, 'r') as hf:
                self.database_features = hf['database_features'][:]
                self.database_utms = hf['database_utms'][:]
        else:
            logger.error("Database features not found")
            print("Please extract database features first.")
            sys.exit()

def center_crop(img, target_width=512, target_height=512):
    """中心裁剪图像"""
    height, width, _ = img.shape
    start_x = width // 2 - target_width // 2
    start_y = height // 2 - target_height // 2
    return img[start_y:start_y + target_height, start_x:start_x + target_width]

def euler_to_rotation_matrix(roll, pitch, yaw):
    """将欧拉角转换为旋转矩阵"""
    R_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return np.dot(R_z, np.dot(R_y, R_x))

def distort(img_path):
    """对图像进行透射变换"""
    img = cv2.imread(img_path)
    height, width = img.shape[:2]
    yaw, pitch, roll = map(np.float32, img_path.split("@")[3:6])
    R = euler_to_rotation_matrix(roll, pitch, yaw)
    src_points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype='float32')
    dst_points = np.dot(src_points - np.array([width / 2, height / 2]), R[:2, :2].T) + np.array([width / 2, height / 2])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points.astype('float32'))
    return cv2.warpPerspective(img, matrix, (width, height))

def pre_process(img_path, contrast_factor=3):
    """预处理图像"""
    img = distort(img_path)
    img = center_crop(img)
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

def process_images(rk_engine, img_list, args):
    """处理图像列表，进行推理"""
    for i in tqdm(range(len(img_list))):
        logger.info('Infer {}/{}'.format(i + 1, len(img_list)))
        img_name = img_list[i]
        img_path = os.path.join(args.img_folder, img_name)
        if not os.path.exists(img_path):
            logger.warning("%s is not found", img_name)
            continue
        t1 = time()
        input_data = pre_process(img_path)
        t2 = time()
        input_data = np.expand_dims(input_data, 0)
        position = rk_engine.model_inference(input_data)
        logger.debug("Pre-processing time: {:.4f} s".format(t2 - t1))
        logger.info("Position: %s", position)
        handle_result(img_path, img_name, position, args)

def handle_result(img_path, img_name, position, args):
    """处理并保存结果图像"""
    if args.img_show or args.img_save:
        img_p = cv2.imread(img_path)
        draw(img_p, position)
        if args.img_save:
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            result_path = os.path.join(args.save_path, img_name)
            cv2.imwrite(result_path, img_p)
            logger.info('Position result saved to %s', result_path)
        if args.img_show:
            cv2.imshow("Full post process result", img_p)
            cv2.waitKeyEx(0)

def main():
    config_path = '/home/xujg/code/UAV-VisionLoc-Deploy/config.json'
    config = load_config(config_path)
    global logger
    logger = configure_logging(config)

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_path', type=str, default=config["model_path"], help='Model path, could be .pt or .rknn file')
    parser.add_argument('--img_show', action='store_true', default=config["img_show"], help='Draw the result and show')
    parser.add_argument('--img_save', action='store_true', default=config["img_save"], help='Save the result')
    parser.add_argument('--save_path', default=config["save_path"], help='Path to save the result')
    parser.add_argument('--img_folder', type=str, default=config["img_folder"], help='Path to the image folder')
    parser.add_argument('--path_local_database', type=str, default=config["path_local_database"], help='Path to load local features and utms of the database')
    parser.add_argument('--features_dim', type=int, default=4096, help='NetVLAD output dims.')
    parser.add_argument('--recall_values', type=int, default=[1, 5, 10, 20], nargs="+", help='Recalls to be computed, such as R@5.')
    parser.add_argument('--use_best_n', type=int, default=1, help='Calculate the position from weighted averaged best n. If n = 1, then it is equivalent to top 1')

    args = parser.parse_args()

    rk_engine = RKInfer(args)

    file_list = sorted(os.listdir(args.img_folder))
    img_list = [path for path in file_list if img_check(path)]

    process_images(rk_engine, img_list, args)

if __name__ == '__main__':
    main()
