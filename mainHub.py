"""
Code for constructing the GUI

Author: Ze'ev Krischer, Australian Centre for Robotics/Data61
<zeev.krischer@sydney.edu.au>s
"""

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import os
import numpy as np
import time
import torch
import argparse

from gui.dataloader import CollaborativeDataloader
from gui.model import CollaborativeSegmentationModel
from gui.constants import (
    OBJECT_CLICK_COLOR, 
    BACKGROUND_CLICK_COLOR, 
    PRISM_CLASS_COLORS,
    PRISM_CLASS_IDS,
    FURNITURE_CLASS_COLORS,
    FURNITURE_CLASS_IDS,
    SCANNET_COLOR_MAP_20,
    CLASS_LABELS_20,
    )
from gui.utils import get_boundary_points, get_extremity_points

class CollaborativeSegmentationGUI:
    """GUI for a collaborative Segmentation Task"""
    def __init__(self, config):

        # Object storage 
        # self.dataloader = CollaborativeDataloader(config.dataset)
        self.model = CollaborativeSegmentationModel(config)
        # Point Colour Storage
        self.base_colors = None
        if self.model.dataset_type == "ScanNetDataset":
            self.class_colour_map = SCANNET_COLOR_MAP_20
            self.class_labels = CLASS_LABELS_20
        elif self.model.dataset_type == "FurnitureDataset":
            self.class_colour_map = FURNITURE_CLASS_COLORS
            self.class_labels = FURNITURE_CLASS_IDS
        elif self.model.dataset_type == "PrismDataset":
            self.class_colour_map = PRISM_CLASS_COLORS
            self.class_labels = PRISM_CLASS_IDS
        else:
            self.class_colour_map = None
            self.class_labels = None

        self.num_classes = len(self.class_labels) if self.class_labels is not None else 0
        # Point cloud handling
        self.point_cloud = None
        self.current_scene_name = None

        # Click Management
        # The click is a dictionary of interaction periods, each period contains a list of clicks
        # Each click is a dictionary with keys 'position', 'type' (object/background), 'time'
        self.click_idx = {'0': []}
        self.click_time_idx = {'0': []}
        self.click_positions = {'0': []}
        self.cur_obj_idx = -1
        self.cur_obj_name = None
        self.last_key_pressed_time = round(time.time() * 1000)
        self.num_clicks = 0#{"0": 0}
        self.interactive_periods = 0
        self.radius = 0.1

        # Visualisation Considerations
        self.vis_ground_truth = False
        self.current_period = 0
        self.current_vis_period = 0
        # Application Definition
        self.app = gui.Application.instance
        self.app.initialize()
        self.window = self.app.create_window("Dumb^2")
        self.font = self.app.add_font(gui.FontDescription("sans-serif"))     
        self.em = self.window.theme.font_size
        self.separation_height = int(round(0.5 * self.em))
        
        standard_margin = gui.Margins(0.5*self.em, 0.5*self.em, 0.5*self.em, 0.5*self.em)
        zero_margin = gui.Margins(0, 0, 0, 0)


        ### User Tools and Information ###
        # Info about usage
        self.user_guide = UserInstruction(spacing=0, margin=standard_margin, font=self.font, separation_height=self.separation_height)
        
        # A collapsible list of all the classes could be here
        # It's here for agile but not sure if I need it here yet

        self.click_info = gui.Label(f"Number of Clicks: {self.num_clicks}") # Only visible once an object is available for clicking
        self.click_info.text_color = gui.Color(1.0, 0.5, 0.0)
        self.click_info.font_id = self.font
        self.object_info = gui.Label("Current Object: None Selected")
        self.scene_name = gui.Label("Scene: None Loaded")

        # Run Button - Run's the segmentation
        self.run_seg_button = gui.Button("Run/Save [Enter]")
        self.run_seg_button.horizontal_padding_em = 2
        self.run_seg_button.vertical_padding_em = 2
        self.run_seg_button.set_on_clicked(self.__run_segmentation)
        self.run_seg_button.background_color = gui.Color(1.0, 0.5, 0.0)

        # Load previous/next scene buttons
        #* Should I give it the option to select scenes from a list?
        self.scene_select_widget = gui.Horiz(0, zero_margin)
        self.prev_button = gui.Button("Previous Scene")
        self.prev_button.horizontal_padding_em = 0.5
        self.prev_button.vertical_padding_em = 0.5
        self.prev_button.set_on_clicked(self.__prev_scene)

        self.next_button = gui.Button("Next Scene")
        self.next_button.horizontal_padding_em = 0.5
        self.next_button.vertical_padding_em = 0.5
        self.next_button.set_on_clicked(self.__next_scene)

        self.scene_select_widget.add_child(self.prev_button)
        self.scene_select_widget.add_stretch()
        self.scene_select_widget.add_child(self.next_button)


        # Add the save and quit buttons
        self.save_quit_button = gui.Button("Save and Quit")
        self.save_quit_button.horizontal_padding_em = 0.5
        self.save_quit_button.vertical_padding_em = 0.5
        self.save_quit_button.set_on_clicked(self.__save_and_quit)

        # Instead of adding these to the right side widget or something I'd like to make them part
        # of a pop up window that I can open and close
        # self.popup_items = gui.Vert(0, gui.Margins(self.em, self.em, self.em, self.em))
        # self.popup_items.add_child(self.user_guide)
        # self.popup_items.add_fixed(self.separation_height)
        # self.popup_items.add_child(self.click_info)
        # self.popup_items.add_fixed(self.separation_height)
        # self.popup_items.add_stretch()
        # self.popup_items.add_fixed(self.separation_height)
        # self.popup_items.add_child(self.run_seg_button)
        # self.popup_items.add_fixed(self.separation_height)
        # self.popup_items.add_child(self.scene_select_widget)

        # Now we handle the left side
        self.viewer_3d = gui.SceneWidget()
        self.window.add_child(self.viewer_3d)
        self.viewer_3d.scene = rendering.Open3DScene(self.window.renderer)

        self.material_record = rendering.MaterialRecord()
        self.material_record.shader = "defaultUnlit"      # Possible values: "defaultLit", "defaultUnlit", "normals", "depth"
        self.material_record.point_size = 2 * self.window.scaling

        # Not sure if I need this since I plan to have a seperate window for the control of points
        # self.window.set_on_layout(self.__on_layout)
        self.window.set_on_key(self.__on_key)
        self.viewer_3d.set_on_mouse(self.__mouse_event)
        self.last_key_pressed_time = round(time.time() * 1000)
        self.mouse_event = None
        self._request_depth = False
        # self.window.set_on_tick_event(self.__on_tick)
        self._render_in_progress = False
        
        #? We have a measure for checking if the user scrolls too far so the points don't get too small?
        self.scrolling_beyond = 0

        # Setting up camera
        self.__reset_view()


    # def __on_layout(self, layout_context):
    #     """ This purely ensures that the scene

    # def __on_tick(self):
    #     if self._request_depth and not self._render_in_progress:
    #         self._request_depth = False
    #         self._render_in_progress = True
    #         def _callback(depth_image):
    #             self._render_in_progress = False
    #             self.__point_clicked_event(depth_image)
    #         self.viewer_3d.scene.scene.render_to_depth_image(_callback)

    def __run_segmentation(self):
        """
        When trigered either by the button or hitting enter this will put the list of clicks
        through the system and run a prediction via the model
        """

        pass
    
    def __update_scene_text(self):
        if hasattr(self.model, "scene_name"):
            self.scene_name.text = f"Scene: {self.model.scene_name}"
            self.click_info.text = f"Number of Clicks: {self.num_clicks}"
        if self.cur_obj_name is not None:
            self.object_info.text = f"Current Object: {self.cur_obj_name}"
            self.object_info.text_color = gui.Color(*self.class_colour_map[self.cur_obj_idx])
        else:
            self.object_info.text = "Current Object: None Selected"

    def __cycle_interactions(self):
        if len(self.predictions) > 0:
            if self.current_vis_period + 1 < len(self.predictions):
                self.current_vis_period += 1
            else:
                self.current_vis_period = 0

            prediction_colors = np.array([self.class_colour_map[label] for label in self.predictions[self.current_vis_period]])
            self.point_cloud.colors = o3d.utility.Vector3dVector(prediction_colors)
            self.viewer_3d.scene.clear_geometry()
            self.viewer_3d.scene.add_geometry("point_cloud", self.point_cloud, self.material_record)


    def __next_scene(self):
        """
        Loads the next scene through the dataloader 
        """
        self.points, self.base_colors, self.ground_truth = self.model.load_next_scene()
        self.boundary_points = get_boundary_points(self.points, self.ground_truth)
        self.extremity_points = get_extremity_points(self.points, self.ground_truth)
        # Print the min max of self.points
        print("Points min:", np.min(self.points, axis=0), "max:", np.max(self.points, axis=0))
        
        self.predictions = []
        self.click_idx = {'0': []}

        self.scene_start_time = time.time()
        self.__model_predict()
        # self.predictions.append(self.model.predict(click_idx=None))
                
        # self.point_cloud = o3d.geometry.PointCloud()
        # self.point_cloud.points = o3d.utility.Vector3dVector(self.points)
        # self.point_cloud.colors = o3d.utility.Vector3dVector(self.base_colors)

        # This successfully loads the scene in the viewer
        # self.viewer_3d.scene.clear_geometry()
        # self.viewer_3d.scene.add_geometry("point_cloud", self.point_cloud, self.material_record)

        # # This centers the view on the loaded point cloud
        # self.__reset_view()

        # Update the command window
        gui.Application.instance.post_to_main_thread(self.window, self.__update_scene_text)


        # pass
    def __model_predict(self):
        prediction = self.model.predict(click_idx=self.click_idx)

        if len(self.predictions) == 0:
            self.current_period = 0
        else:
            self.current_period += 1
        
        self.predictions.append(prediction)

        self.click_idx[str(self.current_period)] = []

        # Map each prediction label to its corresponding color
        # Print the number of points for each class
        unique, counts = np.unique(prediction, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"Class {label}: {count} points")

        print("------------------------------------\n")
        prediction_colors = np.array([self.class_colour_map[label] for label in prediction])

        # Compare the difference between this prediction and the last one

        # print("Loaded points shape from data's .npy file", self.points.shape)
        # print("output prediction shape", prediction.shape)
        # print("Loaded ground truth shape", self.ground_truth.shape)

        # Visualise the prediction
        self.point_cloud = o3d.geometry.PointCloud()
        self.point_cloud.points = o3d.utility.Vector3dVector(self.points)
        self.point_cloud.colors = o3d.utility.Vector3dVector(prediction_colors) 

        self.viewer_3d.scene.clear_geometry()
        self.viewer_3d.scene.add_geometry("point_cloud", self.point_cloud, self.material_record)

        # This centers the view on the loaded point cloud
        self.__reset_view()


    def __visualise_ground_truth(self):
        if self.point_cloud is None:
            pass
        elif self.current_vis_period == -1:
            self.current_vis_period = 0

            prediction_colors = np.array([self.class_colour_map[label] for label in self.predictions[self.current_vis_period]])
            self.point_cloud.colors = o3d.utility.Vector3dVector(prediction_colors) 
            self.viewer_3d.scene.clear_geometry()
            self.viewer_3d.scene.add_geometry("point_cloud", self.point_cloud, self.material_record)
            
           

        elif self.ground_truth is not None:
            # Map each ground truth label to its corresponding color
            gt_colors = np.array([self.class_colour_map[label] for label in self.ground_truth])

            self.gt_point_cloud = o3d.geometry.PointCloud()
            self.gt_point_cloud.points = o3d.utility.Vector3dVector(self.points)
            self.gt_point_cloud.colors = o3d.utility.Vector3dVector(gt_colors)

            self.viewer_3d.scene.clear_geometry()
            self.viewer_3d.scene.add_geometry("point_cloud", self.gt_point_cloud, self.material_record)
            
            self.current_vis_period = -1
        else:
            print("No Ground Truth available for this point cloud")

    def __mouse_event(self, event):
        """
        Callback for user mouse event
        """
        # Add a delay to prevent rapid multiple clicks
        click_delay_ms = 300  # 300 milliseconds
        now = round(time.time() * 1000)
        if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(gui.KeyModifier.CTRL) and self.point_cloud is not None and self.cur_obj_idx != -1:
            if now - self.last_key_pressed_time < click_delay_ms:
                # Ignore click if within delay window
                print("CLICK TOO FAST, SLOW DOWN")
                return gui.Widget.EventCallbackResult.HANDLED
            self.last_key_pressed_time = now

            if event.buttons == 1:
                # Left Click
                self.mouse_event = event
                # print("CLICKITY CLACKETY GET OFF MY PROPERTY")
                self.viewer_3d.scene.scene.render_to_depth_image(self.__point_clicked_event)
                # self.app.post_to_main_thread(self.window, self.__delayed_renderer)
                # self._request_depth = True
                # self.window.post_redraw()

                return gui.Widget.EventCallbackResult.HANDLED
            elif event.buttons == 4:
                # Right Click
                pass
            elif event.buttons == 2:
                # Middle Click
                pass
        elif self.cur_obj_idx == -1 and event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(gui.KeyModifier.CTRL):
            print("No class selected for segmentation, please select a class first.")
            return gui.Widget.EventCallbackResult.HANDLED

        # if event.type == gui.MouseEvent.WHEEL:
        #     # scroll automatically adapts size of point cloud, but stays within a threshold so the points don't vanish
        #     if self.material_record.point_size <= self.window.scaling*2.5 and event.wheel_dy >= 0:
        #         self.scrolling_beyond += event.wheel_dy
        #     elif self.material_record.point_size <= self.window.scaling*2.5 and self.scrolling_beyond > 0:
        #         self.scrolling_beyond += event.wheel_dy
        #     else:
        #         self.material_record.point_size -= 0.7*event.wheel_dy # linear change, TODO: make consistent (issue opened)
        #     gui.Application.instance.post_to_main_thread(
        #         self.window, self.__update_pc_size)
        #     return gui.Widget.EventCallbackResult.HANDLED

        return gui.Widget.EventCallbackResult.IGNORED
    
    def __delayed_renderer(self):
        print("Rendering to depth image after delay")
        self.viewer_3d.scene.scene.render_to_depth_image(self.__point_clicked_event)


    def __point_clicked_event(self, depth_image):
        """
        Called by mouse event, extracts the coordinate frin event to figure out the point clicked
        """
        # Coordinates are expressed relative to the window not the scene widget
        # So we dereference teh image to take this into account
        x = self.mouse_event.x - self.viewer_3d.frame.x
        y = self.mouse_event.y - self.viewer_3d.frame.y

        # Get the depth value (asarray reverses the axes)
        depth = np.asarray(depth_image)[y, x]
        # print("Depth value of clicked pixel:", depth)

        if depth == 1.0:
            print("Clicked on Nothing")
        else:
            point = self.viewer_3d.scene.camera.unproject(self.mouse_event.x, self.mouse_event.y, depth,
                                                                self.viewer_3d.frame.width, self.viewer_3d.frame.height)
            # print(click)
            # click = np.asarray([click[0], click[1], click[2]])  # Convert to list for easier handling
            # print("Clicked on point:", point)

            # Now we need to store the click in self.click_idx
            self.click_idx[str(self.current_period)].append(
                {
                    'position': point,
                    "class": self.cur_obj_idx,
                    'time': time.time() - self.scene_start_time

                }
            )
            self.num_clicks += 1
            print(f"Registered click for object {self.cur_obj_name} at position {point} during period {self.current_period}")
            # Display the click as an object in the viewer
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.radius)
            sphere.translate(point)
            sphere.paint_uniform_color(self.class_colour_map[self.cur_obj_idx])
            self.viewer_3d.scene.add_geometry(f"click_{self.current_period}_{len(self.click_idx[str(self.current_period)])}", sphere, self.material_record)

    def __toggle_click_visibility(self):
        """Toggles the visibility of all click spheres in the viewer"""
        for period in range(self.current_period + 1):
            for idx in range(len(self.click_idx[str(period)])):
                geom_name = f"click_{period}_{idx+1}"
                if self.viewer_3d.scene.has_geometry(geom_name):
                    current_vis = self.viewer_3d.scene.geometry_is_visible(geom_name)
                    self.viewer_3d.scene.show_geometry(geom_name, not current_vis)
        
    def __update_pc_size(self):
        """called when user zooms in, is posted to main thread to update changes in the point size"""
        self.viewer_3d.scene.modify_geometry_material("point_cloud", self.material_record)
    def __prev_scene(self):
        """
        Loads the previous scene through the dataloader (if not at the first scene)
        """

        pass

    def __save_and_quit(self):
        """
        Does as described. Saves and quits from the program
        """
        self.app.quit()


    def __reset_view(self):
        """
        Resets the view to a central locaiton
        """
        bounds = self.viewer_3d.scene.bounding_box
        center = bounds.get_center()
        self.viewer_3d.setup_camera(35, bounds, center)
        self.viewer_3d.look_at(center, [0, 15, 10], [0, 0, 1]) # current dataset has its data at [0, 0, 0]


    def __on_close_command_window(self):
        self.command_window = None

        return True


    def __show_boundary_points(self):
        """
        Use utils to get the boundary points and visualise them 
        """
        if not hasattr(self, "boundary_points"):
            
            data = self.model.scene_data["fragment_list"][0]
            points = data["coord"].cpu().numpy()
            labels = data["segment"].cpu().numpy()

            self.boundary_points = get_boundary_points(points, labels)

        self.viewer_3d.scene.clear_geometry()
        color = np.array([1,1,1])
        white = np.tile(color, (np.asarray(self.point_cloud.points).shape[0], 1))
        self.colorless_pc = o3d.geometry.PointCloud()
        self.colorless_pc.points = self.point_cloud.points
        self.colorless_pc.colors = o3d.utility.Vector3dVector(white)
        self.viewer_3d.scene.add_geometry("point_cloud", self.colorless_pc, self.material_record)
        print(self.boundary_points.keys())
        # print(self.boundary_points)
        for classs in self.boundary_points.keys():
            bn_points = self.boundary_points[(classs)]
            if bn_points.shape[0] == 0:
                continue
            geom_name = f"boundary_{classs}"
            # if self.viewer_3d.scene.has_geometry(geom_name):
            #     current_vis = self.viewer_3d.scene.geometry_is_visible(geom_name)
            #     self.viewer_3d.scene.show_geometry(geom_name, not current_vis)
            # else:
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(bn_points)
            # Set all boundary points to the color of the current class
            color = np.array(self.class_colour_map[classs])
            prediction_colors = np.tile(color, (bn_points.shape[0], 1))
            point_cloud.colors = o3d.utility.Vector3dVector(prediction_colors)

            self.viewer_3d.scene.add_geometry(geom_name, point_cloud, self.material_record)
                

    def __show_extremity_points(self):
        """
        Use utils to get the extremity points and visualise them 
        """
        if not hasattr(self, "extremity_points"):
            
            data = self.model.scene_data["fragment_list"][0]
            points = data["coord"].cpu().numpy()
            labels = data["segment"].cpu().numpy()

            self.extremity_points = get_extremity_points(points, labels)

        self.viewer_3d.scene.clear_geometry()
        color = np.array([1,1,1])
        white = np.tile(color, (np.asarray(self.point_cloud.points).shape[0], 1))
        self.colorless_pc = o3d.geometry.PointCloud()
        self.colorless_pc.points = self.point_cloud.points
        self.colorless_pc.colors = o3d.utility.Vector3dVector(white)
        self.viewer_3d.scene.add_geometry("point_cloud", self.colorless_pc, self.material_record)
        print(self.extremity_points.keys())
        # print(self.boundary_points)
        for classs in self.extremity_points.keys():
            ex_points = self.extremity_points[(classs)]
            if ex_points.shape[0] == 0:
                continue
            geom_name = f"extremity_{classs}"
            # if self.viewer_3d.scene.has_geometry(geom_name):
            #     current_vis = self.viewer_3d.scene.geometry_is_visible(geom_name)
            #     self.viewer_3d.scene.show_geometry(geom_name, not current_vis)
            # else:
            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(ex_points)
            # Set all boundary points to the color of the current class
            color = np.array(self.class_colour_map[classs])
            prediction_colors = np.tile(color, (ex_points.shape[0], 1))
            point_cloud.colors = o3d.utility.Vector3dVector(prediction_colors)

            self.viewer_3d.scene.add_geometry(geom_name, point_cloud, self.material_record)
                


    def __set_current_object(self, class_id, class_name):
        self.cur_obj_idx = class_id
        self.cur_obj_name = class_name
        # print(f"Current object set to {self.cur_obj_name} with class id {self.cur_obj_idx}")
        gui.Application.instance.post_to_main_thread(self.window, self.__update_scene_text)

    def __create_command_window(self):
        if not hasattr(self, "command_window") or self.command_window is None:
            self.command_window = gui.Application.instance.create_window("Command Window", 400, 750)

            popup_items = gui.Vert(0, gui.Margins(self.em, self.em, self.em, self.em))

            # Create widgets here instead of in __init__
            user_guide = UserInstruction(spacing=0, margin=gui.Margins(self.em, self.em, self.em, self.em),
                                        font=self.font,
                                        separation_height=self.separation_height)

            class_names = gui.CollapsableVert("Class Legend")
            class_names.set_is_open(True)
            legend_layout = gui.Vert()

            for class_id, class_name in self.class_labels.items():
                row = gui.Horiz()
                button = gui.Button(f"\t{class_name}")
                button.background_color = gui.Color(*self.class_colour_map[class_id])
                button.set_on_clicked(lambda cid=class_id, cname=class_name: self.__set_current_object(cid, cname))
                row.add_child(button)
                legend_layout.add_child(row)
            class_names.add_child(legend_layout)

            if hasattr(self.model, "scene_name"):
                self.scene_name = gui.Label(f"Scene: {self.model.scene_name}")
            else:
                self.scene_name = gui.Label("Scene: None Loaded")

            if self.cur_obj_name is not None:
                self.object_info = gui.Label(f"Current Object: {self.cur_obj_name}")
            else:
                self.object_info = gui.Label("Current Object: None Selected")

            self.click_info = gui.Label(f"Number of Clicks: {self.num_clicks}")
            self.click_info.text_color = gui.Color(1.0, 0.5, 0.0)
            # self.click_info.font = self.font

            run_seg_button = gui.Button("Run/Save [Enter]")
            run_seg_button.set_on_clicked(self.__model_predict_passthrough)

            scene_select_widget = gui.Horiz(0, gui.Margins(0,0,0,0))
            prev_button = gui.Button("Previous Scene")
            prev_button.set_on_clicked(self.__prev_scene)
            next_button = gui.Button("Next Scene")
            next_button.set_on_clicked(self.__next_scene)
            scene_select_widget.add_child(prev_button)
            scene_select_widget.add_stretch()
            scene_select_widget.add_child(next_button)

            # Add all to popup
            # popup_items.add_child(user_guide)
            # popup_items.add_fixed(self.separation_height)
            popup_items.add_child(class_names)
            popup_items.add_fixed(self.separation_height)
            popup_items.add_child(self.scene_name)
            popup_items.add_fixed(self.separation_height)
            popup_items.add_child(self.object_info)
            popup_items.add_fixed(self.separation_height)
            popup_items.add_child(self.click_info)
            popup_items.add_fixed(self.separation_height)
            popup_items.add_stretch()
            popup_items.add_fixed(self.separation_height)
            popup_items.add_child(run_seg_button)
            popup_items.add_fixed(self.separation_height)
            popup_items.add_child(scene_select_widget)

            self.command_window.add_child(popup_items)
            self.command_window.set_on_close(self.__on_close_command_window)
        else:
            if self.command_window.is_visible:
                self.command_window.show(False)
            else:
                self.command_window.show(True)

    def __model_predict_passthrough(self):
        
        if self.point_cloud is None:
            print("Need to load a point cloud first")
            return
        else:
            self.__model_predict()

    def __on_key(self, key_event):
        """
        Handles all the key's that are pressed during the process
        """
        if key_event.key == gui.KeyName.Q.value and key_event.type != gui.KeyEvent.UP:
            self.__save_and_quit()

        if key_event.key == gui.KeyName.R.value and key_event.type != gui.KeyEvent.UP:
            self.__reset_view()

        if key_event.key == gui.KeyName.M.value and key_event.type != gui.KeyEvent.UP:
            self.__create_command_window()

        if key_event.key == gui.KeyName.N.value and key_event.type != gui.KeyEvent.UP:
            self.__next_scene()

        if key_event.key == gui.KeyName.G.value and key_event.type != gui.KeyEvent.UP:
            self.__visualise_ground_truth()

        if key_event.key == gui.KeyName.C.value and key_event.type != gui.KeyEvent.UP:
            self.__toggle_click_visibility()
        
        if key_event.key == gui.KeyName.ENTER.value and key_event.type != gui.KeyEvent.UP and self.point_cloud != None:
            self.__model_predict()

        if key_event.key == gui.KeyName.I.value and key_event.type != gui.KeyEvent.UP and self.point_cloud != None:
            self.__cycle_interactions()

        if key_event.key == gui.KeyName.B.value and key_event.type != gui.KeyEvent.UP and self.point_cloud != None:
            self.__show_boundary_points()

        if key_event.key == gui.KeyName.E.value and key_event.type != gui.KeyEvent.UP and self.point_cloud != None:
            self.__show_extremity_points()












#! The below is all from the agile3D github and the above is heavily based on it
class UserInstruction(gui.CollapsableVert):
    """Some simple collapsable vertical User Instructions"""
    def __init__(self, spacing, margin, font, separation_height):
        gui.CollapsableVert.__init__(self, "Instructions", spacing, margin)
        descr_obj = gui.Label("{: <30}Object".format("[NUMBER + Click]"))
        descr_obj.text_color = gui.Color(*OBJECT_CLICK_COLOR)
        descr_obj.font_id = font
        descr_bg = gui.Label("{: <30}Background".format("[CTRL + Click]"))
        descr_bg.text_color = gui.Color(*BACKGROUND_CLICK_COLOR)
        descr_bg.font_id = font
        descr_unselect = gui.Label("{: <30}Unselect".format("[CTRL + SHIFT + Click]"))
        descr_unselect.text_color = gui.Color(0.8, 0.8, 0.8)
        descr_unselect.font_id = font
        desr_toggle_colors = gui.Label("{: <30}Toggle Colors".format("[O]"))
        desr_toggle_colors.text_color = gui.Color(0.8, 0.8, 0.8)
        desr_toggle_colors.font_id = font

        self.add_child(desr_toggle_colors)
        self.add_child(descr_obj)
        self.add_child(descr_bg)
        # self.add_child(descr_unselect)
        self.background_color = gui.Color(0.450, 0.454, 0.447, 0.5)


class Objects(gui.CollapsableVert):
    """Class that handles all the data and button functionalities for already created objects"""
    def __init__(self, app, spacing, margin, font, separation_height, em):
        gui.CollapsableVert.__init__(self, "Objects", spacing, margin)
        self.current_object_idx = None
        self.objects = []
        self.app = app
        self.separation_height = separation_height
        self.em = em
        # add new object
        textfield_description = gui.Label("Name a new object:")
        textfield_description.text_color = gui.Color(1.0, 0.5, 0.0)
        textfield_description.font_id = font
        new_object_widget = gui.Horiz(0, gui.Margins(0,0,0,0))
        self.object_textfield = gui.TextEdit()
        self.object_textfield.text_value = "- enter name here-"
        new_object_button = gui.Button("Create")
        new_object_button.horizontal_padding_em = 0.1
        new_object_button.vertical_padding_em = 0.1
        new_object_button.set_on_clicked(self.create_object)
        new_object_widget.add_child(self.object_textfield)
        new_object_widget.add_child(new_object_button)
        # already created objects
        self.toggle_info = gui.Label("") # only visible if at least one object available
        self.toggle_info.text_color = gui.Color(1.0, 0.5, 0.0)
        self.toggle_info.font_id = font
        self.toggle_info.visible = False
        self.dynamic_object_widget = gui.WidgetProxy()
        
        self.add_child(self.dynamic_object_widget)

    def update_buttons(self):
        """used to alter the buttons to change the object to segment 
        the dynamic object button widget is able to delete prior information and add buttons as well, which is important for changing the scene
        usual widgets cannot delete prior children, which poses a problem for changing the scene"""
        self.objects_buttons = gui.Vert(0, gui.Margins(0.5*self.em, 0.5*self.em, 0.5*self.em, 0.5*self.em))
        # objects_buttons.frame.height = 3
        for object_idx, object_name in enumerate(self.objects):

            new_obj_row = gui.Horiz(0, gui.Margins(0,0,0,0))
            new_obj_textfield = gui.TextEdit()
            new_obj_textfield.text_value = "- enter name here-"

            butt = gui.Button(object_name)
            butt.horizontal_padding_em = 1
            butt.vertical_padding_em = 0.1

            butt.background_color = gui.Color(*get_obj_color(object_idx+1, normalize=True))
            butt.set_on_clicked(lambda name=object_name: self.switch_object(name)) # switch object is called by all object buttons and will recognize the object by its id
            butt.tooltip = f"Object '{object_name}'"

            new_obj_row.add_child(butt)
            new_obj_row.add_fixed(self.separation_height)
            new_obj_row.add_child(new_obj_textfield)

            self.objects_buttons.add_child(new_obj_row)
            self.objects_buttons.add_fixed(self.separation_height/3)
        self.dynamic_object_widget.set_widget(self.objects_buttons)
    
    def create_object(self, object_name = None, load_colors = False):
        """Button "Create" pressed by User to create new object or load objects for new scene"""
        # if object_name is None:
        #     object_name = self.underscore_to_blank(self.object_textfield.text_value)
        #     self.object_textfield.text_value = "- enter name here-"
        # if object_name in self.objects:
        #     self.app.window.show_message_box("Object Name Duplicate", "That object name already exists. Please choose another name!")
        #     return
        # if object_name in ["", ""]:
        #     self.app.window.show_message_box("Invalid Object Name", "Please enter a valid object name.")
        #     return
        num_objs = len(self.objects)
        object_name = 'object ' + str(num_objs+1)

        self.objects.append(object_name)
        self.current_object_idx = self.objects.index(object_name) + 1
        self.app.cur_obj_idx = self.current_object_idx
        self.app.cur_obj_name = object_name
        self.app.model.load_object(object_name, load_colors = load_colors) # The model calls the functions to adapt the GUI
    
    def switch_object(self, object_name):
        """Object Button pressed by User"""
        if not self.app.vis_mode_semantics:
            self.app.window.show_message_box("Toggle Object Colors/Semantics", "Please untoggle the scene color with Key <o> first!")
            return
        self.current_object_idx = self.objects.index(object_name) + 1
        self.app.cur_obj_idx = self.current_object_idx
        self.app.cur_obj_name = object_name
        self.app.model.load_object(object_name, load_colors = False) # The model calls the functions to adapt the GUI
    

    @staticmethod
    def underscore_to_blank(name):
        return name.replace('_', ' ')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Setup arguments
    parser.add_argument("--user_name", type=str, default="user_00", help="Name of the user, used to select the profile to attach to the model")
    parser.add_argument("--experiment_name", type=str, default="nonInteractive", help="The experiment from which to load the pretraining weights.")
    parser.add_argument("--dataset", type=str, default="furniture", help="The dataset to get the test scenes from")

    config = parser.parse_args()
    
    visGUI = CollaborativeSegmentationGUI(config)
    # Run the segmentation process - I think this should be part of an overall GUI class rather than the model like AGILE does it
    visGUI.app.run()
