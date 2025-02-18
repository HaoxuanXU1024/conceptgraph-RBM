import ast
import configparser

import imageio
from conceptgraph.cg_process.obj_edge_processor import ObjEdgeProcessor, FeatureMergeDataset, ConfigDict, ObjFeatureGenerator, time_logger, combine_pose, Twc_to_Thc, Thc_to_Twc, mute_print

import os
import random
import numpy as np
import h5py
import json
from os import path
from numpy import save
import quaternion
from requests import get
from omegaconf import OmegaConf, DictConfig
from typing import Dict, Any, List, Optional
import gzip, pickle

import argparse
from pathlib import Path
import re
from PIL import Image
import cv2

import open_clip
import distinctipy
import torch
import torchvision
import supervision as sv

from contextlib import contextmanager
try:
    
    import quaternion
    import trimesh
    from scipy.spatial.transform import Rotation as R
    
    from conceptgraph.dataset.datasets_common import GradSLAMDataset, as_intrinsics_matrix, R2RDataset

    from conceptgraph.utils.model_utils import compute_clip_features
    import torch.nn.functional as F

    from gradslam.datasets import datautils
    from conceptgraph.slam.utils import gobs_to_detection_list
    from conceptgraph.slam.cfslam_pipeline_batch import BG_CLASSES

    # Local application/library specific imports
    from conceptgraph.dataset.datasets_common import get_dataset
    from conceptgraph.utils.vis import OnlineObjectRenderer, vis_result_fast, vis_result_slow_caption
    from conceptgraph.utils.ious import (
        compute_2d_box_contained_batch
    )
    from conceptgraph.utils.general_utils import to_tensor

    from conceptgraph.slam.slam_classes import MapObjectList, DetectionList
    from conceptgraph.slam.utils import (
        create_or_load_colors,
        merge_obj2_into_obj1, 
        denoise_objects,
        filter_objects,
        merge_objects, 
        gobs_to_detection_list,
        get_classes_colors
    )
    from conceptgraph.slam.mapping import (
        compute_spatial_similarities,
        compute_visual_similarities,
        aggregate_similarities,
        merge_detections_to_objects
    )

    import gc
    import open3d

    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

    from habitat.core.vector_env import VectorEnv

    from conceptgraph.dataset.datasets_common import load_dataset_config

    try: 
        from groundingdino.util.inference import Model
        from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator
    except ImportError as e:
        print("Import Error: Please install Grounded Segment Anything following the instructions in README.")
        raise e

    # Set up some path used in this script
    # Assuming all checkpoint files are downloaded as instructed by the original GSA repo
    if "GSA_PATH" in os.environ:
        GSA_PATH = os.environ["GSA_PATH"]
    else:
        raise ValueError("Please set the GSA_PATH environment variable to the path of the GSA repo. ")
        
    import sys
    if "TAG2TEXT_PATH" in os.environ:
        TAG2TEXT_PATH = os.environ["TAG2TEXT_PATH"]
        
    EFFICIENTSAM_PATH = os.path.join(GSA_PATH, "EfficientSAM")
    sys.path.append(GSA_PATH) # This is needed for the following imports in this file
    sys.path.append(TAG2TEXT_PATH) # This is needed for some imports in the Tag2Text files
    sys.path.append(EFFICIENTSAM_PATH)

    import torchvision.transforms as TS
    try:
        from ram.models import ram
        from ram.models import tag2text
        from ram import inference_tag2text, inference_ram
    except ImportError as e:
        print("RAM sub-package not found. Please check your GSA_PATH. ")
        raise e

    # Disable torch gradient computation
    # torch.set_grad_enabled(False)
    # Don't set it in global, just set it in the function that needs it.
    # Using with torch.set_grad_enabled(False): is better.
    # Or using with torch.no_grad(): is also good.
        
    # GroundingDINO config and checkpoint
    GROUNDING_DINO_CONFIG_PATH = os.path.join(GSA_PATH, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
    GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./groundingdino_swint_ogc.pth")

    # Segment-Anything checkpoint
    SAM_ENCODER_VERSION = "vit_h"
    SAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "./sam_vit_h_4b8939.pth")

    # Tag2Text checkpoint
    TAG2TEXT_CHECKPOINT_PATH = os.path.join(TAG2TEXT_PATH, "./tag2text_swin_14m.pth")
    RAM_CHECKPOINT_PATH = os.path.join(TAG2TEXT_PATH, "./ram_swin_large_14m.pth")

    FOREGROUND_GENERIC_CLASSES = [
        "item", "furniture", "object", "electronics", "wall decoration", "door"
    ]

    FOREGROUND_MINIMAL_CLASSES = [
        "item"
    ]
except:
    Warning("The conceptgraph is not installed. The conceptgraph related functions will not work.")

L2C_TRANSFORM = np.array(
    [   [-0.00859563,-0.999714,-0.0223111,0.01927],
        [-0.0428589,0.0227041,-1.00387,0.699481],
        [0.999075,-0.00764551,-0.0423244,0.0611374],
        [0,0,0,1]])

C2G_TRANSFORM = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -0.06], [0.0, 0.0, 0.0, 1.0]])

G2L_TRANSFORM = np.array(
    [   [-0.00859563, -0.999714, -0.0223111, 0.01927], 
        [-0.0428589, 0.0227041, -1.00387, 0.699481], 
        [0.999075, -0.00764551, -0.0423244, 0.0], 
        [0.0, 0.0, 0.0, 1.0]])


class ReconstructionDataset(GradSLAMDataset):
    
    def __init__(
        self,
        config_dict,
        basedir,
        sequence, # the scan_id
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 224,
        desired_width: Optional[int] = 224,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        relative_pose: Optional[bool] = False,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        # self.pose_path = os.path.join(self.input_folder,"camera_parameter", "camera_parameter_"+ sequence + ".conf")
        self.pose_path = os.path.join(self.input_folder,"camera_parameter", "camera_parameter" + ".conf")
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            relative_pose=relative_pose,
            **kwargs,
        )

    def get_filepaths(self):
        color_paths = []
        depth_paths = []
        config = configparser.ConfigParser()
        config.read(self.pose_path)
        #print("pose path, the config file path",pose_path)
        for sec_idx in config.sections():
            # Use ast.literal_eval to convert string representation of list to actual list
            rgb_name = config.get(sec_idx, 'rgb_name')
            depth_name = config.get(sec_idx, 'depth_name')
            color_paths.append(os.path.join(self.input_folder,"color_image", rgb_name))
            depth_paths.append(os.path.join(self.input_folder,"depth_image",depth_name))

        embedding_paths = []
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        poses = []
        config = configparser.ConfigParser()
        config.read(self.pose_path)

        for sec_idx in config.sections():
            data = config.get(sec_idx, 'lpose')
            lines = data.strip('[]').split('\n')
            pose = [float(item) for line in lines for item in line.strip('[]').split()]
            T_l2w = np.reshape(np.array(pose), (4, 4))
            T_l2c = L2C_TRANSFORM
            T_c2w = np.dot(T_l2w, np.linalg.inv(T_l2c))
            poses.append(torch.tensor(T_c2w))

        return poses

    
 
class GimbalReconstructionDataset(GradSLAMDataset):
    
    def __init__(
        self,
        config_dict,
        basedir,
        sequence, # the scan_id
        trajectory,
        trajectory_all = True,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 224,
        desired_width: Optional[int] = 224,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        relative_pose: Optional[bool] = False,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        # self.pose_path = os.path.join(self.input_folder,"camera_parameter", "camera_parameter_"+ sequence + ".conf")
        self.pose_path = os.path.join(self.input_folder,"camera_parameter", "camera_parameter" + ".conf")
        
        if trajectory_all:
            self.trajectory = self.get_trajectory()
        else:
            self.trajectory = trajectory
        
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            relative_pose=relative_pose,
            **kwargs,
        )
        
    def get_trajectory(self):
        config = configparser.ConfigParser()
        config.read(self.pose_path)
        trajectory = []
        for sec_idx in config.sections():
            trajectory.append(sec_idx)
        return trajectory

    def get_filepaths(self):
        color_paths = []
        depth_paths = []
        config = configparser.ConfigParser()
        config.read(self.pose_path)
        #print("pose path, the config file path",pose_path)
        # for sec_idx in config.sections():
        #     # Use ast.literal_eval to convert string representation of list to actual list
        #     rgb_name = config.get(sec_idx, 'rgb_name')
        #     depth_name = config.get(sec_idx, 'depth_name')
        #     color_paths.append(os.path.join(self.input_folder,"color_image", rgb_name))
        #     depth_paths.append(os.path.join(self.input_folder,"depth_image",depth_name))
            
        for sec_idx in self.trajectory:
            rgb_name = config.get(sec_idx, 'rgb_name')
            depth_name = config.get(sec_idx, 'depth_name')
            color_paths.append(os.path.join(self.input_folder,"color_image", rgb_name))
            depth_paths.append(os.path.join(self.input_folder,"depth_image",depth_name))

        embedding_paths = []
        return color_paths, depth_paths, embedding_paths

    def load_poses(self):
        poses = []
        config = configparser.ConfigParser()
        config.read(self.pose_path)

        # for sec_idx in config.sections():
        for sec_idx in self.trajectory:
            data = config.get(sec_idx, 'lpose')
            # Define array in the local scope
            local_scope = {'array': np.array}
            # Evaluate the string representation of the arrays
            arrays = eval(data, {"__builtins__": None}, local_scope)
            
            heading_list = config.get(sec_idx, 'heading')
            
            for array in arrays:
                T_

        return poses
    
    
class Reconstruction(object):
    
    def __init__(self) -> None:
        self.obj_edge_processor = ObjEdgeProcessor() # M2G
        _concept_graph_path = "/home/lg1/peteryu_workspace/m2g_concept_graph/conceptgraph"
        self.config_dict = ConfigDict(
            dataset_config = _concept_graph_path + "/cg_process/m2g_config_files/dataset_r2r_finetune.yaml",
            detection_config= _concept_graph_path + "/cg_process/m2g_config_files/detection_r2r_finetune.yaml",
            merge_config= _concept_graph_path + "/cg_process/m2g_config_files/merge_r2r_finetune.yaml",
            edge_config= _concept_graph_path + "/cg_process/m2g_config_files/edge_r2r_finetune.yaml"
            )
        # self.obj_feature_generator = ObjFeatureGenerator(generator_device=self.device)
        self.obj_feature_generator = ObjFeatureGenerator()
        
        self._init_obj_edge_processor()
        self._init_obj_feature_generator()

    def _init_obj_edge_processor(self):
        self.obj_edge_processor = ObjEdgeProcessor(
            objs_hdf5_save_dir="/data0/vln_datasets/preprocessed_data/finetune_cg_hdf5",
            objs_hdf5_save_file_name="finetune_cg_data.hdf5",
            edges_hdf5_save_dir="/data0/vln_datasets/preprocessed_data/edges_hdf5",
            edges_hdf5_save_file_name="edges.hdf5",
            connectivity_dir="/home/lg1/peteryu_workspace/BEV_HGT_VLN/datasets/R2R/connectivity",
            connectivity_file_name="scans.txt",
            obj_feature_name="clip",
            obj_pose_name="bbox_np",
            allobjs_dict={},
            alledges_dict={},
            allvps_pos_dict={}
            )    
        
    def _init_obj_feature_generator(self):
        self.obj_feature_generator.get_config(
            self.config_dict.detection_config, 
            self.config_dict.merge_config
            )
        self.obj_feature_generator.init_model()
        
    def reconstruction(self, dataset_name="test_1", _start=0, _end=-1, _stride=None):
        
        all_objs = MapObjectList()
        
        dataset = ReconstructionDataset(
            config_dict=self.config_dict.dataset_config,
            basedir="/home/lg1/peteryu_workspace/m2g_concept_graph/dataset/1101_dataset/loop/",
            sequence=dataset_name,
            desired_height=224,
            desired_width=224,
            start=_start,
            stride=_stride,
            end=_end
        )
        
        detection_list, classes_list = self.obj_feature_generator.obj_feature_generate(dataset)
        
        fg_detections_list, bg_detections_list = self.obj_feature_generator.process_detections_for_merge(
                    detection_list, 
                    classes_list, 
                    dataset,
                    )
        
        vp_000 = np.array([0, 0, 0])
        vp_list = [vp_000]
        
        _cur_surround_objs = MapObjectList()
        _cur_surround_objs = self.obj_feature_generator.detections_to_objs(_cur_surround_objs, fg_detections_list, bg_detections_list, self.config_dict.merge_config)
        all_objs = self.obj_feature_generator.merge_objs_objs(
            all_objs, _cur_surround_objs[0], self.config_dict.merge_config, vp_path_list=vp_list, pcd_save_path=dataset_name)
        
        print("We have reconstructed the objects!!!!!!!!")
        
    def reconstruction_gimbal(self, dataset_name="test_1", _start=0, _end=-1, _stride=None):
        
        all_objs = MapObjectList()
        
        dataset = GimbalReconstructionDataset(
            config_dict=self.config_dict.dataset_config,
            basedir="/home/lg1/peteryu_workspace/m2g_concept_graph/dataset/1101_dataset/gimbal/",
            sequence=dataset_name,
            trajectory=[],
            trajectory_all=True,
            desired_height=224,
            desired_width=224,
            start=_start,
            stride=_stride,
            end=_end
        )
    
        detection_list, classes_list = self.obj_feature_generator.obj_feature_generate(dataset)
        
        fg_detections_list, bg_detections_list = self.obj_feature_generator.process_detections_for_merge(
                    detection_list, 
                    classes_list, 
                    dataset,
                    )
        
        vp_000 = np.array([0, 0, 0])
        vp_list = [vp_000]
        
        _cur_surround_objs = MapObjectList()
        _cur_surround_objs = self.obj_feature_generator.detections_to_objs(_cur_surround_objs, fg_detections_list, bg_detections_list, self.config_dict.merge_config)
        all_objs = self.obj_feature_generator.merge_objs_objs(
            all_objs, _cur_surround_objs[0], self.config_dict.merge_config, vp_path_list=vp_list, pcd_save_path=dataset_name)
        
        print("We have reconstructed the objects!!!!!!!!")

if __name__ == "__main__":
    reconstruction = Reconstruction()
    # reconstruction.reconstruction("1101_test_1_1")
    # reconstruction.reconstruction("1101_test_1_2")
    # reconstruction.reconstruction("1101_test_2_1")
    # reconstruction.reconstruction("1101_test_2_2")
    # reconstruction.reconstruction("1101_test_3_1")
    reconstruction.reconstruction("1101_test_3_2_cut")
    
    # ### Gimbal
    # reconstruction.reconstruction_gimbal("1101_test_3_2")
        