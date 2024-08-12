import numpy as np
import cv2
from .logger_config import logger
import sys


def center_crop(img, target_width=512, target_height=512):
    height, width, _ = img.shape
    start_x = width // 2 - target_width // 2
    start_y = height // 2 - target_height // 2
    return img[start_y:start_y + target_height, start_x:start_x + target_width]

def euler_to_rotation_matrix(yaw, pitch, roll):
    R_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]])
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    return np.dot(R_z, np.dot(R_y, R_x))

def distort(img, at):
    height, width = img.shape[:2]
    R = euler_to_rotation_matrix(at[0], at[1], at[2])
    src_points = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype='float32')
    dst_points = np.dot(src_points - np.array([width / 2, height / 2]), R[:2, :2].T) + np.array([width / 2, height / 2])
    matrix = cv2.getPerspectiveTransform(src_points, dst_points.astype('float32'))
    return cv2.warpPerspective(img, matrix, (width, height))

def pre_process(data, at=[0., 0., 0.]):

    if isinstance(data, np.ndarray):
        img = data
    elif isinstance(data, str):
        img = cv2.imread(data)
        at = list(map(np.float32, data.split("@")[3:6]))
    at_info = (
        f"yaw: {at[0]}, "
        f"pitch: {at[1]}, "
        f"roll: {at[2]}, "
    )
    logger.debug(at_info)
    img = distort(img, at)
    img = center_crop(img)
    # if img.shape[0] != 512 or img.shape[1] != 512:
    #     logger.error("input size error.")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_float = img_gray.astype(np.float32) / 255.0
    mean = np.mean(img_float)
    img_contrast = mean + 3 * (img_float - mean)
    img_clipped = np.clip(img_contrast, 0, 1)
    img_rgb = np.repeat(img_clipped[:, :, np.newaxis], 3, axis=2)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_normalized = (img_rgb - mean) / std
    return img_normalized.astype(np.float32)



