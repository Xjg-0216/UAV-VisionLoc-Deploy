import os
import cv2
import sys
import argparse
import h5py
import faiss
import numpy as np
from time import time
from tqdm import tqdm
from memory_profiler import profile

IMG_SIZE = (512, 512)  # (width, height), such as (1280, 736)



def img_check(path):
    img_type = ['.jpg', '.jpeg', '.png', '.bmp']
    for _type in img_type:
        if path.endswith(_type) or path.endswith(_type.upper()):
            return True
    return False


def draw(img, position):
    cv2.putText(img, '{}'.format(position),
                (0, 512 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


class RKInfer:
    def __init__(self, args):
        self.args = args
        self.load_local_database()
        self.faiss_index = faiss.IndexFlatL2(args.features_dim)
        self.faiss_index.add(self.database_features)

    def setup_model(self):
        model_path = self.args.model_path
        from py_utils.rknn_executor import RKNN_model_container 
        self.model = RKNN_model_container(model_path, False)

    def model_inference(self, input_data):

        t4 = time()
        outputs = self.model.run([input_data])
        t5 = time()
        position = self.post_process(outputs[0])
        t6 = time()
        print("inference_time: {:.4f} s".format(t5-t4) )
        print("post_time: {:.4f} s".format(t6-t5) )
        return position

    def post_process(self, result):
        distances, predictions = self.faiss_index.search(
            result, max(self.args.recall_values)
            )
        distance = distances[0]
        prediction = predictions[0]
        sort_idx = np.argsort(distance)
        if self.args.use_best_n == 1:
            best_position = self.database_utms[prediction[sort_idx[0]]]
        else:
            if distance[sort_idx[0]] == 0:
                best_position = self.database_utms[prediction[sort_idx[0]]]
            else:
                mean = distance[sort_idx[0]]
                sigma = distance[sort_idx[0]] / distance[sort_idx[-1]]
                X = np.array(distance[sort_idx[:self.args.use_best_n]]).reshape((-1,))
                weights = np.exp(-np.square(X - mean) / (2 * sigma ** 2))  # gauss
                weights = weights / np.sum(weights)

                x = y = 0
                for p, w in zip(self.database_utms[prediction[sort_idx[:self.args.use_best_n]]], weights.tolist()):
                    y += p[0] * w
                    x += p[1] * w
                best_position = (y, x)
        return best_position
    @profile
    def load_local_database(self):
        if os.path.exists(self.args.path_local_database):
            print("loading Database feature and utms ...")
            # load local features and utms of database.
            with h5py.File(self.args.path_local_database, 'r') as hf:
                self.database_features = hf['database_features'][:]
                self.database_utms = hf['database_utms'][:]

        else:
            print("please extracting database features first.")
            sys.exit()


def center_crop(img, target_width=512, target_height=512):
    height, width, _ = img.shape
    start_x = width//2 - target_width//2
    start_y = height//2 - target_height//2
    return img[start_y:start_y+target_height, start_x:start_x+target_width]

def euler_to_rotation_matrix(roll, pitch, yaw):
    """
    Convert Euler angles to a rotation matrix.
    """
    R_x = np.array([[1, 0, 0],
                    [0, np.cos(roll), -np.sin(roll)],
                    [0, np.sin(roll), np.cos(roll)]])

    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                    [0, 1, 0],
                    [-np.sin(pitch), 0, np.cos(pitch)]])

    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])

    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def distort(img_path):
    img = cv2.imread(img_path)
    # Image dimensions
    height, width = img.shape[:2]
    yaw, pitch, roll = np.float32(img_path.split("@")[3]), np.float32(img_path.split("@")[4]), np.float32(img_path.split("@")[5])
    # Get the rotation matrix
    R = euler_to_rotation_matrix(roll, pitch, yaw)

    # Define the source points (four corners of the image)
    src_points = np.array([[0, 0],
                           [width - 1, 0],
                           [width - 1, height - 1],
                           [0, height - 1]], dtype='float32')

    # Apply the rotation to the source points
    dst_points = np.dot(src_points - np.array([width / 2, height / 2]), R[:2, :2].T) + np.array([width / 2, height / 2])
    dst_points = dst_points.astype('float32')

    # Compute the perspective transform matrix
    matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply the perspective transformation
    adjusted_image = cv2.warpPerspective(img, matrix, (width, height))
    return adjusted_image

def pre_process(img_path, contrast_factor=3):

    # Read image using OpenCV

    img = distort(img_path)
    img = center_crop(img)
    # img = cv2.resize(img, (512, 512))
    image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert 
    # 调整对比度
    # image = cv2.convertScaleAbs(image, alpha=contrast_factor, beta=0)

    # 将图像转换为浮点类型以避免溢出
    img_float = image.astype(np.float32) / 255.0

    # 计算图像的平均值
    mean = np.mean(img_float)

    # 按照 PyTorch 的方式调整对比度
    image = mean + contrast_factor * (img_float - mean)

    # image = np.clip(image, 0, 255)
    image = np.clip(image, 0, 1)
    # 在灰度图像上增加一个维度并复制
    image = np.repeat(image[:, :, np.newaxis], 3, axis=2)

    # 转换为浮点型并归一化
    # image = image.astype(np.float32) / 255.0

    # 归一化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_rgb = (image - mean) / std
    image_rgb = image_rgb.astype(np.float32)

    # 转换为 (C, H, W) 格式
    # return image_rgb.transpose(2, 0, 1)

    return image_rgb


# @profile
def main():

    parser = argparse.ArgumentParser(description='Process some integers.')
    # basic params
    parser.add_argument('--model_path', type=str, default= "/home/xujg/code/UAV-VisionLoc-Deploy/model/uvl_731.rknn", help='model path, could be .pt or .rknn file')

    parser.add_argument('--img_show', action='store_true', default=False, help='draw the result and show')
    parser.add_argument('--img_save', action='store_true', default=True, help='save the result')
    parser.add_argument('--save_path', default="/home/xujg/code/UAV-VisionLoc-Deploy/python/result", help='save the result')
    # data params
    parser.add_argument('--img_folder', type=str, default='/home/xujg/code/UAV-VisionLoc-Deploy/data/queries', help='img folder path')
    parser.add_argument('--path_local_database', type=str, default='/home/xujg/code/UAV-VisionLoc-Deploy/data/database/database_features.h5', help='load local features and utms of database')

    parser.add_argument(
        "--features_dim",
        type=int,
        default=4096,
        help="NetVLAD output dims.",
    )
    # retrieval params
    parser.add_argument(
        "--recall_values",
        type=int,
        default=[1, 5, 10, 20],
        nargs="+",
        help="Recalls to be computed, such as R@5.",
    )
    parser.add_argument(
        "--use_best_n",
        type=int,
        default=1,
        help="Calculate the position from weighted averaged best n. If n = 1, then it is equivalent to top 1"
    )

    args = parser.parse_args()

    rk_engine = RKInfer(args)

    # init model
    rk_engine.setup_model()

    file_list = sorted(os.listdir(args.img_folder))
    img_list = []
    for path in file_list:
        if img_check(path):
            img_list.append(path)

    # run test
    for i in tqdm(range(len(img_list))):
        print('infer {}/{}'.format(i+1, len(img_list)), end='\r')

        img_name = img_list[i]
        img_path = os.path.join(args.img_folder, img_name)
        if not os.path.exists(img_path):
            print("{} is not found", img_name)
            continue
        t1 = time()
        input_data = pre_process(img_path)
        t2 = time()
        input_data = np.expand_dims(input_data, 0)
        position = rk_engine.model_inference(input_data)
        print("pre_time: {:.4f} s".format(t2-t1) )
        print("position: ", position)
        # t3 = time()



        if args.img_show or args.img_save:
            print('\n\nIMG: {}'.format(img_name))
            img_p = cv2.imread(img_path)
            draw(img_p, position)

            if args.img_save:
                if not os.path.exists(args.save_path):
                    os.mkdir(args.save_path)
                result_path = os.path.join(args.save_path, img_name)
                cv2.imwrite(result_path, img_p)
                print('Position result save to {}'.format(result_path))

            if args.img_show:
                cv2.imshow("full post process result", img_p)
                cv2.waitKeyEx(0)
    

if __name__ == '__main__':

    main()