
import cv2
import numpy as np
import h5py

class QueriesDatasetOpencv:
    """Dataset with images from database and queries, used for inference (testing and building cache)."""

    def __init__(self, queries_folder_h5_path):

        # Redirect data folder path to h5
        self.queries_folder_h5_path = queries_folder_h5_path

        queries_folder_h5_df = h5py.File(self.queries_folder_h5_path, "r")

        # Map name to index
        self.queries_name_dict = {}

        # Duplicated elements are removed below
        for index, queries_image_name in enumerate(queries_folder_h5_df["image_name"]):
            self.queries_name_dict[queries_image_name.decode("UTF-8")] = index

        self.queries_paths = sorted(self.queries_name_dict)
        # The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
        self.queries_utms = np.array(
            [(path.split("@")[1], path.split("@")[2])
             for path in self.queries_paths]
        ).astype(np.float32)

        # Add database, queries prefix
        for i in range(len(self.queries_paths)):
            self.queries_paths[i] = "queries_" + self.queries_paths[i]

        self.queries_num = len(self.queries_paths)

        # Close h5 and initialize for h5 reading in __getitem__
        self.queries_folder_h5_df = None
        queries_folder_h5_df.close()

    def __getitem__(self, index):
        # Init
        if self.queries_folder_h5_df is None:
            self.queries_folder_h5_df = h5py.File(self.queries_folder_h5_path, "r")
        img = self.opencv_process(self._find_img_in_h5(index), contrast_factor=3)
        return img, index

    def __len__(self):
        return len(self.queries_paths)

    def _find_img_in_h5(self, index, database_queries_split=None):
        # Find inside index for h5
        if database_queries_split is None:
            image_name = "_".join(self.queries_paths[index].split("_")[1:])
            database_queries_split = self.queries_paths[index].split("_")[0]
            img = self.queries_folder_h5_df["image_data"][self.queries_name_dict[image_name]]
        else:
            raise KeyError("Don't find correct database_queries_split!")
        return img

    def opencv_process(self, image, contrast_factor):
        # image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        img_float = image.astype(np.float32) / 255.0
        
        mean = np.mean(img_float)
        
        image = mean + contrast_factor * (img_float - mean)
        
        # image = np.clip(image, 0, 255)
        image = np.clip(image, 0, 1)
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image_rgb = (image - mean) / std
        image_rgb = image_rgb.astype(np.float32)

        # 转换为 (C, H, W) 
        # return image_rgb.transpose(2, 0, 1)

        return image_rgb

