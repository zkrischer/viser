"""
Code for getting all the data from the files into the gui etc

Author: Ze'ev Krischer, Australian Centre for Robotics/Data61
<zeev.krischer@sydney.edu.au>s
"""

import os
import numpy as np
import time
import torch

from pathlib import Path
from pointcept.datasets import build_dataset, collate_fn
from collections import OrderedDict
import pointcept.utils.comm as comm
from scipy.spatial import cKDTree
from pointcept.engines.defaults import default_config_parser

class CollaborativeDataloader():
    def __init__(self, dataset_name):
        """
        A Class that handles all the conversions from datasets to formats readable by open3d and the model
        """
        
        self.cfg = default_config_parser(str(Path("/workspace/collab3dPerception/configs") / dataset_name / "attentionInteractiveTest.py"), options={})

        # Use this to classify the dataset type
        self.dataset_name = dataset_name
        # Use this to classify the dataset type
        if "scannet" in self.dataset_name:
            self.dataset_type = "ScanNetDataset"
        elif "Furniture" in self.dataset_name or "furniture" in self.dataset_name:
            self.dataset_type = "FurnitureDataset"
        elif "prism" in self.dataset_name:
            self.dataset_type = "PrismDataset"
        elif "WildScenes" in self.dataset_name or "wildscenes" in self.dataset_name:
            self.dataset_type = "WildScenesDataset"
        else:
            print(f"Dataset {self.dataset_name} is not yet supported, either code it in or leave it alone")
            exit(1)
        if self.dataset_type == "ScanNetDataset":
            self.dataset_path = Path("/workspace/collab3dPerception/data") / self.dataset_name / "val"
        else:
            self.dataset_path = Path("/workspace/collab3dPerception/data") / self.dataset_name / "test"
        self.data_root = "/workspace/collab3dPerception/data/" + self.dataset_name
        self.dataloader = self.build_data_loader()
        

        self.dataloader_iter = iter(self.dataloader)
        self.dataloader_list = list(self.dataloader)
        self.current_index = -1
        self.dataset = self.dataloader.dataset
        self.scene_name_to_idx = {
            self.dataset[i]["name"]: i
            for i in range(len(self.dataset))
        }

 
 
 
    def build_data_loader(self):

        #! Currently this only works to test a model trained on one dataset on the same dataset
        dataset = build_dataset(self.cfg.data.test)
        
        if comm.get_world_size() > 1:
            test_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        else:
            test_sampler = None
        
        test_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            pin_memory=True,
            sampler=test_sampler,
            collate_fn=self.__class__.collate_fn,
        )
        return test_loader
 
    @staticmethod
    def collate_fn(batch):
        return batch

 
    def load_next_scene(self):
        try:    
            self.scene_data = next(self.dataloader_iter)[0]
        except StopIteration:
            # self.dataloader_iter = iter(self.dataloader)
            # scene_data = next(self.dataloader_iter)[0]
            """
            Potentially this is the method I could use to make it go backwards
            Reinitialise it as a reversed iterator or something?
            """
            return None, None, None, None
        # Current scene
        self.scene_name = self.scene_data["name"]



        # Convert it from model scope to point cloud scope
        # coord = self.scene_data["fragment_list"]["coord"].cpu().numpy()
        # color = self.scene_data["color"].cpu().numpy()

        # if "segment" in self.scene_data.keys():
        #     segment = self.scene_data["segment"].cpu().numpy()
        #     return coord, color, segment
        # else:
        #     return coord, color, None

        # Instead of converting from model scope to point cloud scope we are just going to
        # directly load from the dataset
        coord = np.load(self.dataset_path / self.scene_name / "coord.npy")
        color = np.load(self.dataset_path / self.scene_name / "color.npy")

        data = self.scene_data["fragment_list"][0]
        # print(data["feat"].shape)



        coord = data["coord"].cpu().numpy()
        color = data["feat"][:,:3].cpu().numpy()

        # self.tree = cKDTree(coord)
        # self.voxel_tree = cKDTree(data["grid_coord"].cpu().numpy())

        if "segment" in data.keys():
            # segment = np.load(self.dataset_path / self.scene_name / "segment.npy")
            segment = data["segment"].cpu().numpy()
            return coord, color, segment, self.scene_data
        else:
            return coord, color, None, self.scene_data


    def load_scene_list(self):
        try:    
            self.scene_data = self.dataloader_list[self.current_index][0]
        except IndexError:
            # self.dataloader_iter = iter(self.dataloader)
            # scene_data = next(self.dataloader_iter)[0]
            """
            Potentially this is the method I could use to make it go backwards
            Reinitialise it as a reversed iterator or something?
            """
            print("Finished the loop")
            return None, None, None, None
        # Current scene
        self.scene_name = self.scene_data["name"]



        # Convert it from model scope to point cloud scope
        # coord = self.scene_data["fragment_list"]["coord"].cpu().numpy()
        # color = self.scene_data["color"].cpu().numpy()

        # if "segment" in self.scene_data.keys():
        #     segment = self.scene_data["segment"].cpu().numpy()
        #     return coord, color, segment
        # else:
        #     return coord, color, None

        # Instead of converting from model scope to point cloud scope we are just going to
        # directly load from the dataset
        coord = np.load(self.dataset_path / self.scene_name / "coord.npy")
        color = np.load(self.dataset_path / self.scene_name / "color.npy")

        data = self.scene_data["fragment_list"][0]
        # print(data["feat"].shape)



        coord = data["coord"].cpu().numpy()
        color = data["feat"][:,:3].cpu().numpy()

        # self.tree = cKDTree(coord)
        # self.voxel_tree = cKDTree(data["grid_coord"].cpu().numpy())

        if "segment" in data.keys():
            # segment = np.load(self.dataset_path / self.scene_name / "segment.npy")
            segment = data["segment"].cpu().numpy()
            return coord, color, segment, self.scene_data
        else:
            return coord, color, None, self.scene_data




    def load_scene_by_name(self, scene_name: str):
        if scene_name not in self.scene_name_to_idx:
            print(self.dataset_name)
            raise ValueError(f"Scene '{scene_name}' not found in dataset")

        idx = self.scene_name_to_idx[scene_name]
        self.scene_data = self.dataset[idx]
        self.scene_name = scene_name

        data = self.scene_data["fragment_list"][0]
        
        coord = data["coord"].cpu().numpy()
        color = data["feat"][:, :3].cpu().numpy()
        # color = np.load(self.dataset_path / self.scene_name / "color.npy")/255
        

        self.tree = cKDTree(coord)
        self.voxel_tree = cKDTree(data["grid_coord"].cpu().numpy())

        if "segment" in data:
            segment = data["segment"].cpu().numpy()
            # print(np.unique(segment))
            return coord, color, segment, self.scene_data
        else:
            return coord, color, None, self.scene_data

 
 
 
 
 
 
    #     self.__load_scene_data()
    #     self.current_scene_data = None

    #     # Stuff for the model

    #     self.test_setup = dict(
    #         type=self.dataset_type,
    #         split="test",
    #         data_root=self.data_root,
    #         transform=[
    #             dict(type="CenterShift", apply_z=True),
    #             dict(type="NormalizeColor"),
    #         ],
    #         test_mode=True,
    #         interactive=True,
    #         test_cfg=dict(
    #             voxelize=dict(
    #                 type="GridSample",
    #                 grid_size=0.02,
    #                 hash_type="fnv",
    #                 mode="interactive test",
    #                 return_grid_coord=True,
    #             ),
    #             crop=None,
    #             post_transform=[
    #                 dict(type="CenterShift", apply_z=False),
    #                 dict(type="ToTensor"),
    #                 dict(
    #                     type="Collect",
    #                     keys=("coord", "grid_coord", "index", "num_points", "segment"), # Index is causing issues but is needed to convert back
    #                     feat_keys=("color", "normal"),
    #                 ),
    #             ],
    #             aug_transform=[
    #                 [
    #                     dict(
    #                         type="RandomRotateTargetAngle",
    #                         angle=[0],
    #                         axis="z",
    #                         center=[0, 0, 0],
    #                         p=1,
    #                     )
    #                 ]
    #             ],
    #         ),
    #     )

    #     self.test_loader = self.build_test_loader()

    # def __load_scene_data(self):
    #     """
    #     Get's the data from each scene and stores it in an iter
    #     """
    #     # print(self.dataset_path)
    #     self.scene_data = {}
    #     if not os.path.exists(self.dataset_path):
    #         print(f"Error: {self.dataset_path} does not exist")
    #         exit(1)

    #     for scene_folder in os.listdir(self.dataset_path):
    #         folder_path = self.dataset_path / scene_folder

    #         if not os.path.isdir(folder_path):
    #             continue
            
    #         data = {}
    #         for file_name in os.listdir(folder_path):
    #             if file_name.endswith(".npy"):
    #                 file = file_name.replace(".npy", "")
    #                 key = file  # Use the whole file name (without .npy) as key for raw data
    #                 data[key] = np.load(os.path.join(folder_path, file_name))
    #         self.scene_data[scene_folder] = data


    #     self.scene_data_iter = iter(self.scene_data.items())

    # def load_next_scene(self):
    #     try:
    #         scene_name, data = next(self.scene_data_iter)
    #     except StopIteration:
    #         """
    #         Potentially this is the method I could use to make it go backwards
    #         Reinitialise it as a reversed iterator or something?
    #         """
    #         print("No more scenes available.")
    #         exit(1)
    #     # print("\n----------------------------------------------------------------------")

    #     self.scene_name = scene_name
    #     self.current_scene_data = data
        

    # def build_test_loader(self):
    #     # test_dataset = build_dataset(self.test_setup)
    #     # if comm.get_world_size() > 1:
    #     #     test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
    #     # else:
    #     #     test_sampler = None
    #     # test_loader = torch.utils.data.DataLoader(
    #     #     test_dataset,
    #     #     batch_size=1,
    #     #     shuffle=False,
    #     #     num_workers=1,
    #     #     pin_memory=True,
    #     #     sampler=test_sampler,
    #     #     collate_fn=self.__class__.collate_fn,
    #     # )
    #     return False#test_loader

