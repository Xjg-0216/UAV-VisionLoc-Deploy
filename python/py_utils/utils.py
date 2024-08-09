import cv2
import os
import yaml
import pyproj
from .logger_config import logger
from ctypes import Structure, c_ubyte, c_uint, c_float


def draw(img, position):
    cv2.putText(img, '{}'.format(position), (0, 512 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

def img_check(path):
    img_types = ['.jpg', '.jpeg', '.png', '.bmp']
    return any(path.lower().endswith(img_type) for img_type in img_types)

def load_config(config_path):
    """load yaml file"""
    with open(config_path, 'r') as config_file:
        return yaml.safe_load(config_file)
    

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
    if args.img_save:
        img_p = cv2.imread(img_path)
        draw(img_p, position)
        if args.img_save:
            if not os.path.exists(args.save_path):
                os.makedirs(args.save_path)
            result_path = os.path.join(args.save_path, img_name)
            cv2.imwrite(result_path, img_p)
            logger.info('Position result saved to {}'.format(result_path))
        # if args.img_show:
        #     cv2.imshow("Full post process result", img_p)
        #     cv2.waitKeyEx(0)


class AAIR(Structure):
    _pack_ = 1                          #让结构体内存连续
    _fields_ = [("start0",   c_ubyte),  #0x55
                ("start1",   c_ubyte),  #0xAA
                ("length",   c_ubyte),  #数据长度,41个字节
                ("id",       c_ubyte),  #报文ID,151
                ("time",     c_uint),   #位置采样时间,
                ("actime",   c_uint),   #飞机相机同步时间
                ("lat",      c_float),  #目标纬度
                ("lng",      c_float),  #目标经度
                ("height",   c_float),  #目标高度，目标高度，固定值，根据实际情况定义一个常量值
                ("yaw",      c_float),  #Yaw
                ("pitch",    c_float),  #pitch
                ("roll",     c_float),  #roll
                ("angle",    c_float),  #航向角                    
                ("crc", c_ubyte)]       #包校验值固定为0xFF   
