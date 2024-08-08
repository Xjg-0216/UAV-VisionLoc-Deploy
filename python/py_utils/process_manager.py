import os
import h5py
import faiss
import numpy as np
import sys
from time import time
from py_utils.rknn_executor import RKNN_model_container 
from .logger_config import logger

class ProcessManager:
    def __init__(self, args):
        self.args = args
        self.load_local_database()
        self.faiss_init()
        self.setup_model()
    
    def faiss_init(self):
        
        logger.debug("faiss init...")
        self.faiss_index = faiss.IndexFlatL2(self.args.features_dim)
        self.faiss_index.add(self.database_features)

    def setup_model(self):
        """load and setup model"""

        logger.debug("setup model...")
        self.model = RKNN_model_container(self.args.model_path, False)

    def model_inference(self, input_data):
        """model inference"""

        logger.debug("rknn model inference...")
        t_start = time()
        outputs = self.model.run([input_data])
        t_end = time()
        position = self.post_process(outputs[0])
        t_end2 = time()
        logger.debug("Inference time: {:.4f} s".format(t_end - t_start))
        logger.debug("post process time: {:.4f} s".format(t_end2 - t_end))
        return position

    def post_process(self, result):
        """"post process and get best position"""

        logger.debug("post process...")
        distances, predictions = self.faiss_index.search(result, max(self.args.recall_values))
        sort_idx = np.argsort(distances[0])
        best_position = self._calculate_best_position(distances[0], predictions[0], sort_idx)
        return best_position

    def _calculate_best_position(self, distance, prediction, sort_idx):
        """"faiss for queries and local database features"""

        if self.args.use_best_n == 1:
            return self.database_utms[prediction[sort_idx[0]]]
        if distance[sort_idx[0]] == 0:
            return self.database_utms[prediction[sort_idx[0]]]
        
        mean = distance[sort_idx[0]]
        sigma = distance[sort_idx[0]] / distance[sort_idx[-1]]
        X = np.array(distance[sort_idx[:self.args.use_best_n]]).reshape((-1,))
        weights = np.exp(-np.square(X - mean) / (2 * sigma ** 2))
        weights /= np.sum(weights)
        return np.average(self.database_utms[prediction[sort_idx[:self.args.use_best_n]]], axis=0, weights=weights)

    def load_local_database(self):
        """"loading local database features and utms"""

        logger.debug("load local database features...")
        if os.path.exists(self.args.path_local_database):
            with h5py.File(self.args.path_local_database, 'r') as hf:
                self.database_features = hf['database_features'][:]
                self.database_utms = hf['database_utms'][:]
        else:
            logger.error("Database features not found")
            print("Please extract database features first.")
            sys.exit()
