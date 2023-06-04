import re
import os
import numpy as np
from fire import Fire
from natsort import natsorted
from loguru import logger
import pandas as pd
import laspy

from datasets.preprocessing.base_preprocessing import BasePreprocessing

class LASPreprocessing(BasePreprocessing):
    def __init__(
            self,
            data_dir: str = "./data/las",
            save_dir: str = "./data/processed/las",
            modes: tuple = ("train", "validation", "test"),
            n_jobs: int = 1,
            sample_proportion: float = 1.0
    ):
        super().__init__(data_dir, save_dir, modes, n_jobs)

        CLASS_LABELS = ["Other", "Trees"]
        # the "Other" class contains the ground and low vegetation
        VALID_CLASS_IDS = np.array([1])  

        # consider adding ground class
        self.class_map = {
            "Other": 0,
            "Trees": 1,
        }

        self.color_map = [
            [0, 255, 0],   # Other
            [0, 0, 255]]   # Trees

        self.create_label_database()

        for mode in self.modes:
            filepaths = []
            for scene_path in [f.path for f in os.scandir(self.data_dir / mode)]:
                filepaths.append(scene_path)
            self.files[mode] = natsorted(filepaths)

    def create_label_database(self):
        label_database = dict()
        for class_name, class_id in self.class_map.items():
            label_database[class_id] = {
                'color': self.color_map[class_id],
                'name': class_name,
                'validation': True
            }

        self._save_yaml(self.save_dir / "label_database.yaml", label_database)
        return label_database

    def process_file(self, filepath, mode):
        """process_file.
        Please note, that for obtaining segmentation labels ply files were used.
        Args:
            filepath: path to the main ply file
            mode: train, test or validation
        Returns:
            filebase: info about file
        """
        filebase = {
            "filepath": filepath,
            "scene": filepath.split("/")[-1],
            "raw_filepath": str(filepath),
            "file_len": -1,
        }

        with laspy.open(filepath) as fh:
            las = fh.read()
        
        points = las.points.array
        
        # subsample the array
        if self.sample_proportion < 1.0:
            points = points[np.random.choice(points.shape[0],
                                             int(points.shape[0] * self.sample_proportion),
                                             replace=False)]
            
        
        # remove points that are unannotated (0) or trees that were not labelled (3)
        points = np.delete(points, np.where(
            (points["raw_classification"] == 0) | 
            (points["raw_classification"] == 3)),
            axis = 0)

        # following the stpls3d format
        column_names = ["X", "Y", "Z", "red", "green", "blue", "treeSP", "treeID"]
        points = points[column_names]
    
        
        # for now, all tree species are mapped to a single value 
        points["treeSP"][points["treeSP"] > 1] = 1
    
        
        # rescale colours to between 0-255
        points["red"] = ((points["red"] - points["red"].min()) * 
                         (1/(points["red"].max() - points["red"].min()) * 255))
        
        points["green"] = ((points["green"] - points["green"].min()) * 
                         (1/(points["green"].max() - points["green"].min()) * 255))
        
        points["blue"] = ((points["blue"] - points["blue"].min()) * 
                         (1/(points["blue"].max() - points["blue"].min()) * 255))
        
        
        
        # this chunk of code converts the "void" type to float32
        temp = np.expand_dims(points["X"].astype(np.float32), axis = 1)
        for column in column_names[1:]:
            temp = np.hstack((temp, np.expand_dims(points[column].astype(np.float32), axis = 1)))
            
        points = np.copy(temp)
        temp = None
        
        points[:, 0] = ((points[:, 0] - points[:, 0].min()) * 
                         (1/(points[:, 0].max() - points[:, 0].min()) * 490))
        
        points[:, 1] = ((points[:, 1] - points[:, 1].min()) * 
                         (1/(points[:, 1].max() - points[:, 1].min()) * 490))
        
        points[:, 2] = ((points[:, 2] - points[:, 2].min()) * 
                         (1/(points[:, 2].max() - points[:, 2].min()) * 490))
        
        """
        # rescale X, Y, Z to be in the range (0, 490) as in stpls3d
        # these are rescaled such that the dimensions are not stretched
        
        min_coord = np.amax(points[:, 0:3])
        max_coord = np.amin(points[:, 0:3])
        
        points[:, 0] = ((points[:, 0] - min_coord) * 
                         (1/(max_coord - min_coord) * 490))
        
        points[:, 1] = ((points[:, 1] - min_coord) * 
                         (1/(max_coord - min_coord) * 490))
        
        points[:, 2] = ((points[:, 2] - min_coord) * 
                         (1/(max_coord - min_coord) * 490))
        """
        
        
        filebase["raw_segmentation_filepath"] = ""

        # add segment id as additional feature (DUMMY)
        points = np.hstack((points,
                            np.ones(points.shape[0])[..., None],   # normal 1
                            np.ones(points.shape[0])[..., None],   # normal 2
                            np.ones(points.shape[0])[..., None],   # normal 3
                            np.ones(points.shape[0])[..., None]))  # segments


        points = points[:, [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 6, 7]]  # move segments after RGB

        # move point clouds to be in positive range (important for split pointcloud function)
        points[:, :3] = points[:, :3] - points[:, :3].min(0)

        if mode == "test": 
            # drop the semantic class and instance id for testing
            points = points[:, :-2]
        else:
            points[points[:, -1] == 0., -1] = -1  # in the instance id, -1 indicates "no instance"

        file_len = len(points)
        filebase["file_len"] = file_len

        processed_filepath = self.save_dir / mode / f"{filebase['scene'].replace('.las', '')}.npy"
        if not processed_filepath.parent.exists():
            processed_filepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(processed_filepath, points.astype(np.float32))
        filebase["filepath"] = str(processed_filepath)

        if mode in ["validation", "test"]:
            blocks = self.splitPointCloud(points, size=50.0, stride=50)
            
            print("len(blocks)", len(blocks))

            filebase["instance_gt_filepath"] = []
            filebase["filepath_crop"] = []
                
            for block_id, block in enumerate(blocks):
                if len(block) > 10:
                    if mode == "validation":
                        new_instance_ids = np.unique(block[:, -1], return_inverse=True)[1]

                        assert new_instance_ids.shape[0] == block.shape[0]
                        # == 0 means -1 == no instance
                        # new_instance_ids[new_instance_ids == 0]
                        assert new_instance_ids.max() < 1000, "we cannot encode when there are more than 999 instances in a block"

                        gt_data = (block[:, -2]) * 1000 + new_instance_ids

                        processed_gt_filepath = self.save_dir / "instance_gt" / mode / f"{filebase['scene'].replace('.las', '')}_{block_id}.txt"
                        if not processed_gt_filepath.parent.exists():
                            processed_gt_filepath.parent.mkdir(parents=True, exist_ok=True)
                        np.savetxt(processed_gt_filepath, gt_data.astype(np.int32), fmt="%d")
                        filebase["instance_gt_filepath"].append(str(processed_gt_filepath))

                    processed_filepath = self.save_dir / mode / f"{filebase['scene'].replace('.las', '')}_{block_id}.npy"
                    if not processed_filepath.parent.exists():
                        processed_filepath.parent.mkdir(parents=True, exist_ok=True)
                    np.save(processed_filepath, block.astype(np.float32))
                    filebase["filepath_crop"].append(str(processed_filepath))
                else:
                    print("block was smaller than 10 points")
                    assert False

        filebase["color_mean"] = [
            float((points[:, 3] / 255).mean()),
            float((points[:, 4] / 255).mean()),
            float((points[:, 5] / 255).mean()),
        ]
        filebase["color_std"] = [
            float(((points[:, 3] / 255) ** 2).mean()),
            float(((points[:, 4] / 255) ** 2).mean()),
            float(((points[:, 5] / 255) ** 2).mean()),
        ]
        return filebase

    def compute_color_mean_std(
            self, train_database_path: str = "./data/processed/las/train_database.yaml"
    ):
        train_database = self._load_yaml(train_database_path)
        color_mean, color_std = [], []
        for sample in train_database:
            color_std.append(sample["color_std"])
            color_mean.append(sample["color_mean"])

        color_mean = np.array(color_mean).mean(axis=0)
        color_std = np.sqrt(np.array(color_std).mean(axis=0) - color_mean ** 2)
        feats_mean_std = {
            "mean": [float(each) for each in color_mean],
            "std": [float(each) for each in color_std],
        }
        self._save_yaml(self.save_dir / "color_mean_std.yaml", feats_mean_std)

    def splitPointCloud(self, cloud, size=50.0, stride=50):
        limitMax = np.amax(cloud[:, 0:3], axis=0)
        width = int(np.ceil((limitMax[0] - size) / stride)) + 1
        depth = int(np.ceil((limitMax[1] - size) / stride)) + 1
        cells = [(x * stride, y * stride) for x in range(width) for y in range(depth)]
        blocks = []
        for (x, y) in cells:
            xcond = (cloud[:, 0] <= x + size) & (cloud[:, 0] >= x)
            ycond = (cloud[:, 1] <= y + size) & (cloud[:, 1] >= y)
            cond = xcond & ycond
            block = cloud[cond, :]
            blocks.append(block)
        return blocks

    @logger.catch
    def fix_bugs_in_labels(self):
        pass

    def _parse_scene_subscene(self, name):
        scene_match = re.match(r"scene(\d{4})_(\d{2})", name)
        return int(scene_match.group(1)), int(scene_match.group(2))


if __name__ == "__main__":
   Fire(LASPreprocessing)


