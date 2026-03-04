import viser
import os
import numpy as np
import trimesh
import trimesh.creation
import h5py
import time

from gui.constants import (
    PRISM_CLASS_COLORS,
    PRISM_CLASS_IDS,
    FURNITURE_CLASS_COLORS,
    FURNITURE_CLASS_IDS,
    SCANNET_CLASS_IDS,
    SCANNET_CLASS_COLORS,
    TETRA_VERTS,
    TETRA_FACES
    )
from gui.utils import get_boundary_points, get_extremity_points
from gui.dataloader import CollaborativeDataloader
from pathlib import Path
from pointcept.utils.misc import intersection_and_union


DEPTH_SCALE = 0.25     # tune: how quickly size grows with distance
MIN_SIZE    = 0.5    # clamp for near points
MAX_SIZE    = 50     # clamp for far points


class visPlatform:
    """
        A platform for visualising the results or just the dataset
    """

    def __init__(self, config):
        self.config = config
        self.dataset = config.dataset

        if config.gui_user is not None and config.gui_user != "None":
            # If the path does not exist try a gui session version, if that doesn't work too exit with error
            self.user_name = config.gui_user

            self.test_results = f"/workspace/collab3dPerception/gui_sessions/{self.user_name}.h5"

        elif config.exp_name is not None and config.exp_name != "None":
            self.exp_name = config.exp_name
            self.test_results = f"/workspace/collab3dPerception/exp/{self.dataset}/{self.exp_name}/test_results.h5"
            self.user_name = None
        else:
            self.exp_name = None
            self.user_name = None

        self.dataloader = CollaborativeDataloader(dataset_name=self.dataset)
        
        if self.dataloader.dataset_type == "ScanNetDataset":
            scene_names = ["scene0144_00", "scene0690_01", "scene0652_00", "scene0616_00", "scene0314_00", "scene0378_00", "scene0307_00", "scene0435_03", "scene0019_00", "scene0046_01", "scene0050_02", "scene0011_00", "scene0300_00", "scene0328_00", "scene0633_00", "scene0678_01", "scene0139_00", "scene0598_00"]
        elif self.dataloader.dataset_type == "FurnitureDataset":
            scene_names = ["scene_1527", "scene_1589", "scene_1634", "scene_1698", "scene_1721", "scene_1806", "scene_1912", "scene_1975"]
        elif self.dataloader.dataset_type == "PrismDataset":
            scene_names = ["scene_47", "scene_93", "scene_128", "scene_196", "scene_274", "scene_319", "scene_386", "scene_442", "scene_497"]

        if len(scene_names) != 0:
            self.scene_iter = iter(scene_names)
        else:
            self.scene_iter = None

        # Now we can organise the colour storage
        self.base_colours = None
        if self.dataloader.dataset_type == "ScanNetDataset":
            self.class_colour_map = SCANNET_CLASS_COLORS
            self.class_labels = SCANNET_CLASS_IDS
        elif self.dataloader.dataset_type == "FurnitureDataset":
            self.class_colour_map = FURNITURE_CLASS_COLORS
            self.class_labels = FURNITURE_CLASS_IDS
        elif self.dataloader.dataset_type == "PrismDataset":
            self.class_colour_map = PRISM_CLASS_COLORS
            self.class_labels = PRISM_CLASS_IDS

        self.num_classes = len(self.class_labels)
        self.predictions = None

        # Visualisation considerations
        self.current_vis_period = 0
        self.current_clicks = 0
        self.total_budget = 30

        self.vis_ground_truth = False
        self.vis_entropy = False
        self.vis_base_colours = True

        self.current_visualisation = "None"
        self.meshes = []


        # Now we set up the whole system
        self.server = viser.ViserServer()
        self.shutdown_flag = False
        self.shutdown = False
        self.server.gui.configure_theme(control_width="large")

        # Point cloud handling
        self.point_cloud = self.server.scene.add_point_cloud(
            "point_cloud",
            points=np.zeros((1, 3), dtype=np.float32),
            colors=np.zeros((1, 3), dtype=np.float32),
            visible=False,
            point_size= 0.01,
        )
        self.point_mesh = None

        # Add lights for proper shading.
        self.server.scene.add_light_directional("/light/key", intensity=1.2, cast_shadow=True)
        self.server.scene.add_light_ambient("/light/amb", intensity=0.3)

        self.current_scene_name = "N/A"
        self.current_scene_data = None
        self.current_iou = "N/A"
        self.current_acc = "N/A"

        # By default base colour visualisation is true
        # Then we'll have a button for each of the different visualisations
        # And a drop down for all the predictions
        self.text = self.server.gui.add_markdown(
            f"""
            **Current Information**\n
            *Scene Name:* {self.current_scene_name}\n
            *Visualisation:* {self.current_visualisation}\n
            *Clicks Applied:* N/A\n
            *Remaining Budget:* N/A\n\n
            **Metrics**\n
            *Intersection over Union:* {self.current_iou}\n
            *Accuracy:* {self.current_acc}:
            """
        )

        
        self.base_button = self.server.add_button(
            label="Base Colour",
            color="gray"
        )
        self.gt_button = self.server.add_button(
            label="Ground Truth",
            color="gray"
        )
        self.entropy_button = self.server.add_button(
            label="Entropy",
            color="gray",
            visible=False,
            disabled=True
        )

        self.prediction_selector = self.server.gui.add_dropdown(
            label="Prediction Visualisation",
            options = ["None"], #+ [[f"Interaction {f}" for f, _ in enumerate(self.predictions)] if self.predictions is not None else []]
        )

        self.clicks_visible = self.server.gui.add_checkbox(
            label="Clicks Visible",
            initial_value=False,
            disabled=True
        )
        self.scene_search = self.server.gui.add_text(
            label="Scene Search",
            initial_value= "",
            multiline=True
        )
            
        self.scene_nav = self.server.gui.add_button_group(
            label="Scene Navigation",
            options = ("Previous", "Next"),
        )

        self.save_quit_button = self.server.gui.add_button(
            "Save and Quit [Esc]",
        )

        if self.exp_name is None and self.user_name is None:
            # Entropy stays disabled because the system is not ready for it
            # self.entropy_button.disabled = True
            self.clicks_visible.disabled=True
            self.prediction_selector.disabled = True

            self.clicks_visible.visible = False
            self.prediction_selector.visible = False


        # Callback definitions
        @self.scene_nav.on_click
        def _(_) -> None:
            if self.scene_nav.value == "Previous":
                # print("Previously on Total Drama Island")
                self.__prev_scene()

            elif self.scene_nav.value == "Next":
                # print("Next time on Total Drama Island")
                self.__next_scene()

        @self.save_quit_button.on_click
        def _(_) -> None:
            # print("Shutting down Viser GUI")
            if self.shutdown == True:
                return
            
            self.shutdown = True
            time.sleep(0.5)
            self.server.stop()
            self.shutdown_flag = True

        @self.base_button.on_click
        def _(_) -> None:
            if self.current_scene_data is not None:
                self._visualise_cloud(self.base_colors)
                self.current_visualisation = "Base Colours" 
                self.prediction_selector.value = "None"
                self.current_acc = "N/A"
                self.current_iou = "N/A"


            self._update_scene_text()

        @self.gt_button.on_click
        def _(_) -> None:
            if self.current_scene_data is not None:
                self._visualise_cloud(np.array([self.class_colour_map[label] for label in self.ground_truth]))
                self.current_visualisation = "Ground Truth" 
                self.prediction_selector.value = "None"
                self.current_acc = "N/A"
                self.current_iou = "N/A"

            self._update_scene_text()
        @self.entropy_button.on_click
        def _(_) -> None:
            pass

        @self.prediction_selector.on_update
        def _(_) -> None:
            
            if self.prediction_selector.value is None or self.prediction_selector.value == "None":
                pass
            else:
                self.current_visualisation = self.prediction_selector.value
                self._select_prediction(int(self.prediction_selector.value.strip("_")[-1]))
            
            self._update_scene_text()

        @self.clicks_visible.on_update
        def _(_) -> None:
            self._visualise_clicks(self.clicks_visible.value)


        @self.scene_search.on_update
        def _(_) -> None:
            val = self.scene_search.value

            if "\n" in val:
                self.scene_search.value = val.strip("\n")

                # Then we try to load a new scene with a specific scene_name
                self._load_scene(self.scene_search.value)
        # Client related issues
        @self.server.on_client_connect
        def _(client: viser.ClientHandle) -> None:
            self.client = client

            # @self.client.camera.on_update
            # def _(camera: viser.CameraHandle) -> None:
            #     cam_pos = np.array(camera.position, dtype=np.float32)
            #     if self.current_scene_data is not None:
            #         dists = np.linalg.norm(self.points - cam_pos[None, :], axis=1)
            #         sizes = np.clip(DEPTH_SCALE / (1e-6 + dists), MIN_SIZE, MAX_SIZE).astype(np.float32)
            #         self.point_mesh.batched_scales = sizes.reshape(-1, 1)  # (N, 1)



        @self.server.on_client_disconnect
        def _(client: viser.ClientHandle) -> None:
            self.client = None
            # We also can do something later here to save the state of the session so it can be effectively resumed?

    def _load_scene(self, scene_name):
        

        # Load the next scene through the iterator
        try:
            self.points, self.base_colors, self.ground_truth, self.current_scene_data = self.dataloader.load_scene_by_name(scene_name) 
        except ValueError:
            self.client.add_notification("Scene Error", "No scene with that name in available data")
            return

        self.clicks = []
        self.clicks_visible.value = False
        self.current_scene_name = self.current_scene_data["name"]

        self.usability_mask = self.ground_truth != -1
        self.current_acc = "N/A"
        self.current_iou = "N/A"

        
        # Now we load in the base colours of the point cloud as the original view
        # Original Scene Loading
        self._visualise_cloud(self.base_colors)
        self.current_visualisation = "Base Colours"
        self.prediction_selector.value = "None"

        if self.exp_name is not None or self.user_name is not None:
            self._load_predictions(self.current_scene_name)

        self._update_scene_text()


    def __next_scene(self):
        self.clicks = []
        self.clicks_visible.value = False

        # Load the next scene through the iterator
        if self.scene_iter is not None:
            scene_name = next(self.scene_iter, None)
            if scene_name is None:
                self.client.add_notification(title="End of Scenes", body="No more scenes available.", auto_close_seconds=5.0)
                print("nothing left to load")
                return
            self.points, self.base_colors, self.ground_truth, self.current_scene_data = self.dataloader.load_scene_by_name(scene_name) 
        else:
            self.dataloader.current_index += 1 
            self.points, self.base_colors, self.ground_truth, self.current_scene_data = self.dataloader.load_scene_list()
            if self.current_scene_data is None:
                self.client.add_notification(title="End of Scenes", body="No more scenes available.", auto_close_seconds=5.0)
                print("nothing left to load")
                return


        self.current_scene_name = self.current_scene_data["name"]

        self.usability_mask = self.ground_truth != -1
        self.current_acc = "N/A"
        self.current_iou = "N/A"

        
        # Now we load in the base colours of the point cloud as the original view
        # Original Scene Loading
        self._visualise_cloud(self.base_colors)
        self.current_visualisation = "Base Colours"
        self.prediction_selector.value = "None"

        if self.exp_name is not None or self.user_name is not None:
            self._load_predictions(self.current_scene_name)

        self._update_scene_text()


    def __prev_scene(self):
        if self.dataloader.current_index <= 0:
            self.client.add_notification(title="Nothing Previous", body="Can't display previous scene until we have two scenes", auto_close_seconds=5.0)
            return

        self.clicks = []
        self.clicks_visible.value = False

        # Load the next scene through the iterator
        if self.scene_iter is not None:
            scene_name = next(self.scene_iter, None)
            if scene_name is None:
                self.client.add_notification(title="End of Scenes", body="No more scenes available.", auto_close_seconds=5.0)
                print("nothing left to load")
                return
            self.points, self.base_colors, self.ground_truth, self.current_scene_data = self.dataloader.load_scene_by_name(scene_name) 
        else:
            self.dataloader.current_index -= 1
            self.points, self.base_colors, self.ground_truth, self.current_scene_data = self.dataloader.load_scene_list()
            if self.current_scene_data is None:
                self.client.add_notification(title="End of Scenes", body="No more scenes available.", auto_close_seconds=5.0)
                print("nothing left to load")
                return


        self.current_scene_name = self.current_scene_data["name"]

        self.usability_mask = self.ground_truth != -1
        self.current_acc = "N/A"
        self.current_iou = "N/A"

        
        # Now we load in the base colours of the point cloud as the original view
        # Original Scene Loading
        self._visualise_cloud(self.base_colors)
        self.current_visualisation = "Base Colours"
        self.prediction_selector.value = "None"

        if self.exp_name is not None or self.user_name is not None:
            self._load_predictions(self.current_scene_name)

        self._update_scene_text()

    def _visualise_cloud(self, colours):

        # if self.point_mesh is not None: 
        #     self.point_mesh.remove()
        #     self.point_mesh = None

        # self.point_mesh = self.server.scene.add_batched_meshes_simple(
        #     name="Point Mesh",
        #     vertices=TETRA_VERTS,
        #     faces=TETRA_FACES,
        #     batched_positions = self.points,
        #     batched_wxyzs= np.tile([1,0,0,0], (self.points.shape[0],1)).astype(np.float32),
        #     batched_colors=colours,
        #     cast_shadow=True, receive_shadow=True,
        # )
        self.point_cloud.points = self.points
        # colours[~self.usability_mask] = (1,0,0)
        self.point_cloud.colors = colours
        self.point_cloud.visible = True

    def _update_scene_text(self):
        self.text.content = f"""
            **Current Information**\n
            *Scene Name:* {self.current_scene_name}\n
            *Visualisation:* {self.current_visualisation}\n
            *Clicks Applied:* {self.current_clicks}\n
            *Remaining Budget:* {self.total_budget - self.current_clicks}\n\n\n
            **Metrics**\n
            *Intersection over Union:* {self.current_iou}\n
            *Accuracy:* {self.current_acc}:
            """

        


    def _load_predictions(self, scene_name):
        with h5py.File(self.test_results, "r") as results_file:
            self.predictions = results_file[scene_name]["predictions"][:]
            self.corrections = results_file[scene_name]["corrections"][:]
            self.interactions = results_file[scene_name]["interactions"][:]
            self.remaining_budget = results_file[scene_name].attrs["remaining_budget"]
        # print(self.points.shape)
        # print(self.predictions.shape)
        # print(self.corrections.shape)
        # print(self.interactions.shape)
        # print(self.remaining_budget)
        
        self.prediction_selector.options = ["None"] + [f"Prediction {i}" for i in range(0,self.predictions.shape[-1])]

        
        
        
    def _select_prediction(self, prediction_num):

        current_prediction = self.predictions[:,prediction_num]
        self.clicks = []
        self.clicks_visible.value = False
        if prediction_num > 0:
            current_corrections = self.corrections[:,:,prediction_num-1]
            total_interactions = self.interactions[:,:prediction_num]
            
            self.current_clicks = int(np.count_nonzero(total_interactions != -10))

        elif prediction_num == 0:
            current_corrections = None
            current_interaction = 0
            self.current_clicks = 0

        self.current_vis_period = prediction_num


        # Now we can visualise the point colours
        colours = np.array([self.class_colour_map[label] for label in current_prediction])
        colours[~self.usability_mask] = 0
        self._visualise_cloud(colours)

        # We need to visualise the clicks for the next current period, if the prediction_num is > self.interactions.shape[1]
        if prediction_num > self.interactions.shape[1]-1:
            print("No clicks available for final prediction")
            self.client.add_notification(title="Clicks Unavailable", body="No clicks were made on the final prediction.", auto_close_seconds=5.0)

            self.clicks_visible.disabled = True
        else:
            current_interactions = self.interactions[:,prediction_num]
            clicks = current_interactions[current_interactions != -10]
            # We then want to compute which points the clicks are at
            click_pos = self.points[current_interactions != -10]

            
            for i, click in enumerate(clicks):
                self.clicks.append(
                    {
                        "class": int(click),
                        "position": click_pos[i,:]
                    }
                )
            self.clicks_visible.disabled = False


        # We then calculate the intersection over union
        intersection, union, target = intersection_and_union(
            current_prediction, self.ground_truth, self.num_classes
        )

        self.current_acc = f"{round(np.sum(intersection) / (np.sum(target) + 1e-10) * 100,2)}%"
        self.current_iou = f"{round(np.sum(intersection) / (np.sum(union) + 1e-10) * 100,2)}%"


    def _visualise_clicks(self, visualise):
        if visualise:
            for i, click in enumerate(self.clicks):
                sphere = trimesh.creation.icosphere(radius=0.1)
                sphere.apply_translation(click["position"])
                color = (
                    self.class_colour_map[click["class"]]
                )

                color = np.asarray(color, dtype=np.float32)

                # If colors are 0–255 scale
                if color.max() > 1.0:
                    dark_color = np.clip(color * 0.6, 0, 255).astype(np.uint8)
                else:
                    # If colors are 0–1 scale
                    dark_color = np.clip(color * 0.6, 0, 1.0)

                sphere.visual.face_colors = np.tile(
                    dark_color,
                    (sphere.faces.shape[0], 1),
                )

                mesh = self.server.scene.add_mesh_trimesh(
                    name=f"click_{self.current_vis_period}_{i}",
                    mesh=sphere,
                )

                self.meshes.append(mesh)

        else:
            for mesh in self.meshes:
                mesh.remove()
            self.meshes = []