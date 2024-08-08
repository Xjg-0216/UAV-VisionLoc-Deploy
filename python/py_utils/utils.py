import cv2
import os
import json
import pyproj
from .logger_config import logger



def draw(img, position):
    cv2.putText(img, '{}'.format(position), (0, 512 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def img_check(path):
    img_types = ['.jpg', '.jpeg', '.png', '.bmp']
    return any(path.lower().endswith(img_type) for img_type in img_types)

def load_config(config_path):
    """load config file"""
    with open(config_path, 'r') as config_file:
        return json.load(config_file)
    

def utm_to_latlon(position, zone_number, zone_letter):
    """
    将UTM坐标转换为经纬度。
    
    params:
    easting:
    northing: 
    zone_number: 
    zone_letter: 
    
    return:
    纬度，经度
    """
    if zone_letter >= 'N':
        hemisphere = 'north'
    else:
        hemisphere = 'south'
    
    proj_utm = pyproj.Proj(proj='utm', zone=zone_number, hemisphere=hemisphere)
    proj_latlon = pyproj.Proj(proj='latlong', datum='WGS84')
    transformer = pyproj.Transformer.from_proj(proj_utm, proj_latlon)
    lon, lat = transformer.transform(position[0], position[1])
    return [lon, lat]


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
            logger.info('Position result saved to {}'.format(result_path))
        if args.img_show:
            cv2.imshow("Full post process result", img_p)
            cv2.waitKeyEx(0)
