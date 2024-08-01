import os
import cv2
import sys
import argparse
import h5py
import faiss
import numpy as np
from time import time
from tqdm import tqdm
from py_utils import datasets
from sklearn.neighbors import NearestNeighbors

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

    def setup_model(self):
        model_path = self.args.model_path
        from py_utils.rknn_executor import RKNN_model_container 
        self.model = RKNN_model_container(model_path)

    def model_inference(self, input_data):

        # t4 = time()
        outputs = self.model.run([input_data])
        # t5 = time()
        # position = self.post_process(outputs[0])
        # t6 = time()
        # print("inference_time: {:.4f} s".format(t5-t4) )
        # print("post_time: {:.4f} s".format(t6-t5) )
        return outputs[0]




if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process some integers.')
    # basic params
    parser.add_argument('--model_path', type=str, default= "/home/xujg/code/UAV-VisionLoc/model/uvl_731.rknn", help='model path, could be .pt or .rknn file')
    
    parser.add_argument('--img_show', action='store_true', default=False, help='draw the result and show')
    parser.add_argument('--img_save', action='store_true', default=False, help='save the result')

    # data params
    parser.add_argument('--img_folder', type=str, default='home/xujg/code/UAV-VisionLoc/python/test_queries_imgs/', help='img folder path')
    parser.add_argument('--path_local_database', type=str, default='/home/xujg/code/UAV-VisionLoc/data/database/database_features.h5', help='load local features and utms of database')

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
    queries_folder_h5_path = "/home/xujg/code/UAV-VisionLoc/data/queries/test_sample.h5"
    # init model
    rk_engine.setup_model()

    queries_dataset = datasets.QueriesDatasetOpencv(queries_folder_h5_path)

    queries_features = np.empty((queries_dataset.queries_num, args.features_dim), dtype="float32")


    for inputs, indices in tqdm(queries_dataset, ncols=100):

        # inputs = inputs.numpy()
        inputs = np.expand_dims(inputs, 0)
        features = rk_engine.model_inference(inputs)
        queries_features[indices, :] = features


    print(f"Final feature dim: {queries_features.shape[1]}")

    ## get database features , utms and positives
    if os.path.exists(args.path_local_database):
        print("loading Database feature and utms")
        # load local features and utms of database.
        with h5py.File(args.path_local_database, 'r') as hf:
            database_features = hf['database_features'][:]
            database_utms = hf['database_utms'][:]

    else:
        print("please extracting database features first.")


    # Find soft_positives_per_query, which are within val_positive_dist_threshold (deafult 60 meters)
    knn = NearestNeighbors(n_jobs=-1)
    knn.fit(database_utms)
    positives_per_query = knn.radius_neighbors(
        queries_dataset.queries_utms,
        radius=60,
        return_distance=False,
    )

    print("Calculating recalls")
    faiss_index = faiss.IndexFlatL2(args.features_dim)
    faiss_index.add(database_features)
    distances, predictions = faiss_index.search(
        queries_features, max(args.recall_values)
    )

    # args.recall_values by default is [1, 5, 10, 20]
    recalls = np.zeros(len(args.recall_values))
    for query_index, pred in enumerate(predictions):
        for i, n in enumerate(args.recall_values):
            if np.any(np.in1d(pred[:n], positives_per_query[query_index])):
                recalls[i:] += 1
                break
    # Divide by the number of queries*100, so the recalls are in percentages
    recalls = recalls / queries_dataset.queries_num * 100
    recalls_str = ", ".join(
        [f"R@{val}: {rec:.1f}" for val,
            rec in zip(args.recall_values, recalls)]
    )
    print(f"{recalls_str}")


