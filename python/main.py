from py_utils.utils import load_config
import argparse
from py_utils.process_manager import ProcessManager
from py_utils.utils import utm_to_latlon, AAIR
import os
import time
from py_utils.pre_process import pre_process
import numpy as np
from py_utils.logger_config import configure_logging, logger
import socket
import cv2
from ctypes import sizeof, memmove, addressof
import threading
import queue


class CameraAndAttitudeCapture:
    def __init__(self, args):
        self.jbSocket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.jbSocket.bind((args.ip, args.port))
        self.cap = cv2.VideoCapture(args.camera_index)
        self.img_save = args.img_save
        self.save_path = args.save_path
        if not self.cap.isOpened():
            logger.error("Unable to open camera.")
        self.g_air = AAIR()
        
    def capture_image(self):
        ret, frame = self.cap.read()
        if not ret:
            logger.error("Failed to read data from camera")
        timestamp = time.strftime("%Y%m%d-%H%M%S", time.localtime())
        filename = f"capture_{timestamp}.jpg"
        logger.debug((f"img: {filename}"))
        if self.img_save:
            cv2.imwrite(os.path.join(self.save_path, filename), frame)
        return frame

    def receive_attitude_data(self):
        try:
            # 接收姿态数据
            recv_data, _ = self.jbSocket.recvfrom(sizeof(AAIR))
            memmove(addressof(self.g_air), recv_data, sizeof(AAIR))
            # 校验数据包
            if self.g_air.start0 != 0x55 or self.g_air.start1 != 0xAA or self.g_air.crc != 0xFF:
                logger.error("Received packet error")
        except Exception as e:
            logger.warning("Error in data packet process.")
        attitude_info = (
            f"Receive data: ",
            f"start0: {hex(self.g_air.start0)}, "
            f"start1: {hex(self.g_air.start1)}, "
            f"length: {self.g_air.length}, "
            f"id: {self.g_air.id}, "
            f"time: {self.g_air.time}, "
            f"actime: {self.g_air.actime}, "
            f"lat: {self.g_air.lat}, "
            f"lng: {self.g_air.lng}, "
            f"height: {self.g_air.height}, "
            f"yaw: {self.g_air.yaw}, "
            f"pitch: {self.g_air.pitch}, "
            f"roll: {self.g_air.roll}, "
            f"angle: {self.g_air.angle}, "
            f"crc: {hex(self.g_air.crc)}"
        )
        logger.info(attitude_info)
        return [self.g_air.yaw, self.g_air.pitch, self.g_air.roll]


def receive_data_thread(ca, data_queue):
    while True:
        attitude_data = ca.receive_attitude_data()
        if not data_queue.full():
            data_queue.put(attitude_data)
        else:
            # 如果队列已满，丢弃最早的数据，只保留最新的数据
            data_queue.get()
            data_queue.put(attitude_data)


def process_data_thread(ca, pm, args, data_queue):
    while True:
        if not data_queue.empty():
            # 只处理最新的数据
            while not data_queue.empty():
                attitude_data = data_queue.get()
            
            # 捕获图像
            frame = ca.capture_image()
            input_data = pre_process(frame, attitude_data)
            # input_data = cv2.resize(input_data, (512, 512))
            input_data = np.expand_dims(input_data, 0)
            position = pm.model_inference(input_data)
            if args.output_type == "lonlat":
                position = utm_to_latlon(position, zone_number=args.zone_number, zone_letter=args.zone_letter)

            logger.info("Position: {}".format(position))
        


def main():
    config_path = '/home/xujg/code/UAV-VisionLoc-Deploy/config.yaml'
    config = load_config(config_path)
    configure_logging(config)

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_path', type=str, default=config["model_path"], help='Model path, could be .pt or .rknn file')
    parser.add_argument('--img_save', action='store_true', default=config["img_save"], help='Save the result')
    parser.add_argument('--save_path', default=config["save_path"], help='Path to save the result')
    parser.add_argument('--path_local_database', type=str, default=config["path_local_database"], help='Path to load local features and utms of the database')
    parser.add_argument('--features_dim', type=int, default=4096, help='NetVLAD output dims.')
    parser.add_argument('--zone_number', type=int, default=config['utm_zone_number'], help='zone number of utm.')
    parser.add_argument('--zone_letter', type=str, default=config['utm_zone_letter'], help='zone letter of utm.')
    parser.add_argument('--output_type', type=str, default=config['output_type'], help='output type.')
    parser.add_argument('--ip', type=str, default=config['ip'], help='ip')
    parser.add_argument('--port', type=int, default=config['port'], help='port')
    parser.add_argument('--camera_index', type=int, default=config['camera_index'], help='camera index')
    # parser.add_argument('--recall_values', type=int, default=[1, 5, 10, 20], nargs="+", help='Recalls to be computed, such as R@5.')
    parser.add_argument('--recall_values', type=int, default=[1], nargs="+", help='Recalls to be computed, such as R@5.')
    parser.add_argument('--use_best_n', type=int, default=1, help='Calculate the position from weighted averaged best n. If n = 1, then it is equivalent to top 1')

    args = parser.parse_args()

    pm = ProcessManager(args)
    ca = CameraAndAttitudeCapture(args)

    data_queue = queue.Queue(maxsize=10)  # 设置队列的最大容量

    # 启动接收数据的线程
    receive_thread = threading.Thread(target=receive_data_thread, args=(ca, data_queue))
    receive_thread.start()

    # 启动处理数据的线程
    process_thread = threading.Thread(target=process_data_thread, args=(ca, pm, args, data_queue))
    process_thread.start()

    receive_thread.join()
    process_thread.join()


if __name__ == '__main__':
    main()
