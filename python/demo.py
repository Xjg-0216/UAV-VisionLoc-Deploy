

from py_utils.utils import load_config
import argparse
from py_utils.process_manager import ProcessManager
from py_utils.utils import img_check, utm_to_latlon, handle_result
from py_utils.logger_config import configure_logging
import os
from tqdm import tqdm
from time import time
from py_utils.pre_process import pre_process
import numpy as np
from py_utils.logger_config import configure_logging, logger


def process_images(pm, args):
    """load local images and model inference"""

    file_list = sorted(os.listdir(args.img_folder))
    img_list = [path for path in file_list if img_check(path)]
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
        position = pm.model_inference(input_data)
        if args.output_type == "latlon":
            position = utm_to_latlon(position, zone_number=args.zone_number, zone_letter=args.zone_letter)
        elif args.output_type != "utm":
            logger.error("output type error!")
        logger.debug("pre processing time: {:.4f} s".format(t2 - t1))
        logger.info("Position: {}".format(position))
        handle_result(img_path, img_name, position, args)



def main():
    config_path = '/home/xujg/code/UAV-VisionLoc-Deploy/config.yaml'
    config = load_config(config_path)
    configure_logging(config)

    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--model_path', type=str, default=config["model_path"], help='Model path, could be .pt or .rknn file')
    parser.add_argument('--img_save', action='store_true', default=config["img_save"], help='Save the result')
    parser.add_argument('--save_path', default=config["save_path"], help='Path to save the result')
    parser.add_argument('--img_folder', type=str, default=config["img_folder"], help='Path to the image folder')
    parser.add_argument('--path_local_database', type=str, default=config["path_local_database"], help='Path to load local features and utms of the database')
    parser.add_argument('--features_dim', type=int, default=4096, help='NetVLAD output dims.')
    parser.add_argument('--zone_number', type=int, default=config['utm_zone_number'], help='zone number of utm.')
    parser.add_argument('--zone_letter', type=str, default=config['utm_zone_letter'], help='zone letter of utm.')
    parser.add_argument('--output_type', type=str, default=config['output_type'], help='output type.')
    parser.add_argument('--recall_values', type=int, default=[1, 5, 10, 20], nargs="+", help='Recalls to be computed, such as R@5.')
    parser.add_argument('--use_best_n', type=int, default=1, help='Calculate the position from weighted averaged best n. If n = 1, then it is equivalent to top 1')


    args = parser.parse_args()
    pm = ProcessManager(args)
    logger.info("demo, load local queries path. ")
    process_images(pm, args)

if __name__ == '__main__':
    main()