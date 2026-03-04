import numpy as np
import time
import torch
import open3d as o3d
import torch.nn.functional as F

from pointcept.engines.defaults import default_config_parser
from pathlib import Path
from pointcept.models import build_model
from collections import OrderedDict
from pointcept.datasets import build_dataset, collate_fn, point_collate_fn
import pointcept.utils.comm as comm
from pointcept.engines.defaults import create_ddp_model
from scipy.spatial import cKDTree
from pointcept.interaction.utils import shannon_entropy


class CollaborativeSegmentationModel:
    def __init__(self, dataset, exp_name, buffer_size=5):

        self.exp_config_file_path = Path("/workspace/collab3dPerception/exp") / dataset / exp_name / "config.py"
        self.buffer_size = buffer_size
        self.cfg = default_config_parser(str(self.exp_config_file_path), options={})
        self.radius = self.cfg.training_model.radius
        # print("Radius for interaction feature:", self.radius)
        model = build_model(dict(
            type="CollaborativeSegmentor_GUI",
            backbone=self.cfg.model.backbone,
            num_classes=self.cfg.model.num_classes,
            buffer_size=self.buffer_size
        ))

        self.model = create_ddp_model(
            model.cuda(),
            broadcast_buffers=False,
            find_unused_parameters=self.cfg.find_unused_parameters,
        )

        # Load the weights
        weight_path = Path("/workspace/collab3dPerception/exp") / dataset / exp_name / "model/model_best.pth"
        checkpoint = torch.load(
            weight_path,
            map_location=lambda storage, loc: storage.cuda(),
            weights_only=False,
        )        

        weight = OrderedDict()
        for key, value in checkpoint["state_dict"].items():
            if not key.startswith("module."):
                key = "module." + key  # xxx.xxx -> module.xxx.xxx
            # Now all keys contain "module." no matter DDP or not.
            # if self.keywords in key:
            #     key = key.replace(self.keywords, self.replacement, 1)
            if comm.get_world_size() == 1:
                key = key[7:]  # module.xxx.xxx -> xxx.xxx
            weight[key] = value

        self.model.load_state_dict(weight, strict=False)
        self.model.eval()

        # Data handling
    #     self.dataset_name = dataset
    #     # Use this to classify the dataset type
    #     if "scannet" in self.dataset_name:
    #         self.dataset_type = "ScanNetDataset"
    #     elif "Furniture" in self.dataset_name or "furniture" in self.dataset_name:
    #         self.dataset_type = "FurnitureDataset"
    #     elif "prism" in self.dataset_name:
    #         self.dataset_type = "PrismDataset"
    #     elif "WildScenes" in self.dataset_name or "wildscenes" in self.dataset_name:
    #         self.dataset_type = "WildScenesDataset"
    #     else:
    #         print(f"Dataset {self.dataset_name} is not yet supported, either code it in or leave it alone")
    #         exit(1)

    #     self.dataset_path = Path("/workspace/collab3dPerception/data") / self.dataset_name / "test"
    #     self.data_root = "/workspace/collab3dPerception/data/" + self.dataset_name
    #     self.dataloader = self.build_data_loader()

    #     self.dataloader_iter = iter(self.dataloader)
        

    # def build_data_loader(self):

    #     #! Currently this only works to test a model trained on one dataset on the same dataset
    #     dataset = build_dataset(self.cfg.data.test)
    #     if comm.get_world_size() > 1:
    #         test_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    #     else:
    #         test_sampler = None
    #     test_loader = torch.utils.data.DataLoader(
    #         dataset,
    #         batch_size=1,
    #         shuffle=False,
    #         num_workers=1,
    #         pin_memory=True,
    #         sampler=test_sampler,
    #         collate_fn=self.__class__.collate_fn,
    #     )
    #     return test_loader

    # @staticmethod
    # def collate_fn(batch):
    #     return batch
    


        
    # def convert_click_2_feature(self, clicks):
    #     """
    #     Takes in a list of dictionaries which are clicks and converts them into a feature vector
    #     """
    #     data = self.scene_data["fragment_list"][0]
    #     grid = data["grid_coord"].cpu().numpy()
    #     coord = data["coord"].cpu().numpy()
    #     segment = data["segment"].cpu().numpy()
    #     grid_size = self.cfg.data.test.test_cfg.voxelize.grid_size
    #     radius_voxels = (self.radius / grid_size)

    #     min_coord = coord.min(axis=0)
    #     print("coord.min:", min_coord, "coord.max:", coord.max(axis=0))
    #     print("grid.min:", grid.min(axis=0), "grid.max:", grid.max(axis=0))

    #     interaction_feature = torch.zeros((grid.shape[0], self.cfg.model.num_classes), device=data["coord"].device)

    #     for click in clicks:
    #         point = np.array(click["position"])
            
    #         distances, indices = self.tree.query(point)
    #         nearest_coord = coord[indices]
    #         # print("Click Location:", point)
    #         # print("Nearest point:", nearest_coord, "at index", indices, "distance", distances)
    #         # print("Click Class:", click["class"], "Nearest Point Class", segment[indices])

    #         voxel_coord = np.floor((nearest_coord - min_coord) / grid_size).astype(grid.dtype)
    #         voxel_dist, voxel_idx = self.voxel_tree.query(voxel_coord)
    #         nearest_voxel = torch.tensor(grid[voxel_idx], device=data["coord"].device, dtype=torch.float32)
    #         # print("voxel:", voxel_coord, "-> nearest voxel:", nearest_voxel)
    #         # print("Any match:", np.any(np.all(grid == nearest_voxel, axis=1)))

    #         # Now we have the voxel we want to convert it to a feature
    #         nearest_coord = torch.tensor(coord[indices], device=data["coord"].device, dtype=torch.float32)

    #         distances = torch.norm(data["coord"] - nearest_coord, dim=1)
    #         # print(distances)
    #         mask = distances <= self.radius
    #         # print(mask)
    #         if mask.any():
    #             # print("hi")
    #             masked_distances = distances[mask]
    #             normed = 1 - (masked_distances / self.radius).clamp(0, 1)

    #             interaction_feature[mask, click["class"]] = normed
            
    #         # print((interaction_feature[:, click["class"]] > 0).sum().item())
    #     return interaction_feature

    def convert_click_2_feature(self, clicks, scene_data):
        """
        Converts a list of click dictionaries into a feature tensor based on proximity in the point cloud.
        """
        data = scene_data["fragment_list"][0]
        grid = data["grid_coord"]
        coord = data["coord"]  # keep as torch.Tensor
        segment = data["segment"]
        radius = self.radius

        interaction_feature = torch.zeros((grid.shape[0], self.cfg.model.num_classes), device=coord.device)
        # print(interaction_feature.shape)
        for click in clicks:
            print("Processing click:", click["position"], "Class:", click["class"])
            # Convert click position to a torch tensor on the same device
            point = torch.tensor(click["position"], device=coord.device, dtype=coord.dtype).unsqueeze(0)  # shape (1, 3)

            # Compute distances to all points in the cloud
            distances = torch.cdist(coord, point)[:, 0]  # shape (N,)

            # Find nearest point index
            nearest_idx = distances.argmin()
            nearest_coord = coord[nearest_idx].unsqueeze(0)  # shape (1, 3)

            # Create mask of points within the radius
            mask = torch.norm(coord - nearest_coord, dim=1) <= radius

            if mask.any():
                masked_distances = torch.norm(coord[mask] - nearest_coord, dim=1)
                normed = 1 - (masked_distances / radius).clamp(0, 1)
                interaction_feature[mask, click["class"]] = normed

        return interaction_feature
    


    def clicks_to_feature(self, clicks, scene_data):
        coord = scene_data["fragment_list"][0]["coord"]
        num_classes = self.cfg.num_classes
        radius = self.radius

        N = coord.shape[0]
        features = torch.zeros((N, num_classes), dtype=torch.float32, device=coord.device)

        if len(clicks) == 0:
            return features

        if coord.dtype == torch.int64:
            coord = coord.float()

        # ---- clicks → tensors (K, 3) and (K,) ----
        click_points_np = np.asarray(
            [c["position"] for c in clicks],
            dtype=np.float32
        )

        click_points = torch.from_numpy(click_points_np).to(
            device=coord.device,
            dtype=coord.dtype,
        )

        click_classes = torch.as_tensor(
            [c["class"] for c in clicks],
            device=coord.device,
            dtype=torch.long,
        )
        # ---- distances: (N, K) ----
        distances = torch.cdist(coord, click_points)

        # ---- radius + intensity ----
        within_radius = distances <= radius
        intensities = (1.0 - distances / radius).clamp(min=0.0, max=1.0)

        # ---- max-pool per class (identical logic) ----
        for cls in torch.unique(click_classes):
            cls_mask = click_classes == cls
            if cls_mask.any():
                features[:, cls] = intensities[:, cls_mask].max(dim=1).values

        return features
    
    def predict(self, click_idx=None, scene_name = None,scene_data=None):
        # Now we need to prepare the input_dict
        # print(self.scene_data["fragment_list"][0].keys())
        if scene_data is None or scene_name is None:
            exit("Scene data must be provided for prediction.")
            exit(1)

        data = scene_data["fragment_list"][0]

        # print("coord.min:", data["coord"].min(axis=0).values.cpu().numpy(), "coord.max:", data["coord"].max(axis=0).values.cpu().numpy())
        if click_idx is not None:
            # If click_idx is provided, turn it into an interaction feat
            
            # We only want the clicks of the most recent iteration (cause the model has a buffer)
            # We convert those into a feature using the interaction thingo
            
            # Get the data from the last key in click_idx
            last_key = list(click_idx.keys())[-1]
            
            clicks = click_idx[last_key]
            if len(clicks) > 0:
                interaction_feat = self.clicks_to_feature(clicks=clicks, scene_data=scene_data)

            else:
                interaction_feat = torch.zeros((data["coord"].shape[0], self.cfg.model.num_classes), device=data["coord"].device)
            # interaction_feat = click_idx[last_key]

        for key in data.keys():
            data[key] = data[key].cuda()

        input_dict = dict(
            scene_name=scene_name,
            coord=data["coord"],
            grid_coord=data["grid_coord"],
            feat=data["feat"],
            offset=data["offset"],
            interactions_feat=interaction_feat
        )
        
        output = self.model(input_dict)
        pred_logits = output["prediction"].data.cpu()
        pred_points = F.softmax(output["prediction"],-1).max(1)[1].data.cpu().numpy()

        # Now calculate the shannon entropy
        entropy = shannon_entropy(pred_logits).numpy()
        corrections = torch.clamp(interaction_feat, max=1).cpu().numpy()

        # Set all points != 1 to -10
        sparse = np.where(corrections == 1, corrections, -10)
        
        # We then collapse this to a single dimension vector
        interactions = np.argmax(sparse, axis=1)
        rows_with_no_ones = np.all(sparse == -10, axis=1)
        interactions[rows_with_no_ones] = -10
        
        return dict(prediction=pred_points, entropy=entropy, corrections=corrections, interactions=interactions)

