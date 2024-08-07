import os
import argparse
from py_utils.logger_config import configure_logging, load_config
from py_utils.utils import img_check, pre_process, draw
from py_utils.rk_infer import RKInfer
from time import time
from tqdm import tqdm
import numpy as np
import cv2
import logging

def process_images(rk_engine, img_list, args):
    for i in tqdm(range(len(img_list))):
        logging.getLogger().info('Infer {}/{}'.format(i + 1, len(img_list)))
        img_name = img_list[i]
        img_path = os.path.join(args.img_folder, img_name)
        if not os.path.exists(img_path):
            logging.getLogger().warning("%s is not found", img_name)
            continue
        t1 = time()
        input_data = pre_process(img_path)
        t2 = time()
        input_data = np.expand_dims(input_data, 0)
        position = rk_engine.model_inference(input_data)
        logging.getLogger().debug("Pre-processing time: {:.4f} s".format(t2 - t1))
        logging.getLogger().info("Position: %s", position)
        handle_result(img_path, img_name, position, args)

def handle_result(img_path, img_name, position, args):
    if args.img_show or args.img_save:
        img_p = cv2.imread(img_path)
        draw(img_p, position)
        if args.img_save:
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            result_path = os.path.join(args.save_path, img_name)
            cv2.imwrite(result_path, img_p)
            logging.getLogger().info('Position result saved to %s', result_path)
        if args.img_show:
            cv2.imshow("Full post process result", img_p)
            cv2.waitKeyEx(0)

def main():
    config_path = '/home/xujg/code/UAV-VisionLoc-Deploy/config.json'
    config = load_config(config_path)
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
