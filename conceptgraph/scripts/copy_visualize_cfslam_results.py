import copy
import json
import os
import pickle
import gzip
import argparse
import random
import matplotlib
import numpy as np
import pandas as pd
import open3d as o3d
import torch
import torch.nn.functional as F
import open_clip

import distinctipy

from gradslam.structures.pointclouds import Pointclouds

from conceptgraph.slam.slam_classes import MapObjectList
from conceptgraph.utils.vis import LineMesh
from conceptgraph.slam.utils import filter_objects, merge_objects
import heapq

def a_star(start, goal_pcd, obstacles, grid_size=0.05, goal_threshold=0.1):
    """
    A* pathfinding algorithm in 2D.
    
    Args:
    - start (tuple): (x, y, z) coordinates for the start position.
    - goal_pcd (o3d.geometry.PointCloud): Point cloud of the goal object.
    - obstacles (list): List of point clouds of obstacles.
    - grid_size (float): Size of the grid for pathfinding.
    - goal_threshold (float): Distance threshold to consider the goal reached.
    
    Returns:
    - path (list): List of (x, y, z) coordinates for the path.
    """
    def heuristic(a, b):
        return np.linalg.norm(np.array(a[:2]) - np.array(b[:2]))
    
    def is_collision(point, obstacles, threshold=0.07):
        for pcd in obstacles:
            points = np.asarray(pcd.points)
            # 只计算 z 轴在 -0.5 到 -0.8 范围内的点云
            mask = (points[:, 2] >= -1.5) & (points[:, 2] <= -0.8)
            filtered_points = points[mask]
            distances = np.linalg.norm(filtered_points[:, :2] - np.array(point[:2]), axis=1)
            if np.any(distances < threshold):
                return True
        return False
    
    def is_goal(point, goal_pcd, threshold):
        points = np.asarray(goal_pcd.points)
        distances = np.linalg.norm(points[:, :2] - np.array(point[:2]), axis=1)
        return np.any(distances < threshold)
    
    start = (start[0], start[1], -0.5)  # 固定 z 轴坐标为 -0.5
    goal_center = np.mean(np.asarray(goal_pcd.points), axis=0)
    goal_center = (goal_center[0], goal_center[1], -0.5)  # 固定 z 轴坐标为 -0.5
    
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal_center)}
    
    closest_point = start
    closest_distance = heuristic(start, goal_center)
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if is_goal(current, goal_pcd, goal_threshold):
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            print(f"start point: {start}")  # 添加调试信息
            print(f"Generated path: {path}")  # 添加调试信息
            print(f"Success to get the goal area")
            return path
        
        neighbors = [
            (current[0] + grid_size, current[1], -0.5),
            (current[0] - grid_size, current[1], -0.5),
            (current[0], current[1] + grid_size, -0.5),
            (current[0], current[1] - grid_size, -0.5),
        ]
        
        for neighbor in neighbors:
            if is_collision(neighbor, obstacles):
                continue
            
            tentative_g_score = g_score[current] + heuristic(current, neighbor)
            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal_center)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))
                
                # 更新离目标最近的点
                distance_to_goal = heuristic(neighbor, goal_center)
                if distance_to_goal < closest_distance:
                    closest_point = neighbor
                    closest_distance = distance_to_goal
    
    # 如果没有找到路径，返回离目标最近的路径
    path = []
    current = closest_point
    while current in came_from:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    print("No valid path found. Returning the closest path.")  # 添加调试信息
    print(f"start point: {start}")  # 添加调试信息
    print(f"Generated path: {path}")  # 添加调试信息
    print(f"Fail to get the goal area")
    return path
    
def create_ball_mesh(center, radius, color=(0, 1, 0)):
    """
    Create a colored mesh sphere.
    
    Args:
    - center (tuple): (x, y, z) coordinates for the center of the sphere.
    - radius (float): Radius of the sphere.
    - color (tuple): RGB values in the range [0, 1] for the color of the sphere.
    
    Returns:
    - o3d.geometry.TriangleMesh: Colored mesh sphere.
    """
    mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    mesh_sphere.translate(center)
    mesh_sphere.paint_uniform_color(color)
    return mesh_sphere

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_path", type=str, default=None)
    parser.add_argument("--rgb_pcd_path", type=str, default=None)
    parser.add_argument("--edge_file", type=str, default=None)
    
    parser.add_argument("--no_clip", action="store_true", 
                        help="If set, the CLIP model will not init for fast debugging.")
    
    # To inspect the results of merge_overlap_objects
    # This is mainly to quickly try out different thresholds
    parser.add_argument("--merge_overlap_thresh", type=float, default=-1)
    parser.add_argument("--merge_visual_sim_thresh", type=float, default=-1)
    parser.add_argument("--merge_text_sim_thresh", type=float, default=-1)
    parser.add_argument("--obj_min_points", type=int, default=0)
    parser.add_argument("--obj_min_detections", type=int, default=0)
    
    return parser

# def load_result(result_path):
#     with gzip.open(result_path, "rb") as f:
#         results = pickle.load(f)
    
#     if isinstance(results, dict):
#         objects = MapObjectList()
#         objects.load_serializable(results["objects"])
        
#         if results['bg_objects'] is None:
#             bg_objects = None
#         else:
#             bg_objects = MapObjectList()
#             bg_objects.load_serializable(results["bg_objects"])

#         class_colors = results['class_colors']
#     elif isinstance(results, list):
#         objects = MapObjectList()
#         objects.load_serializable(results)

#         bg_objects = None
#         class_colors = distinctipy.get_colors(len(objects), pastel_factor=0.5)
#         class_colors = {str(i): c for i, c in enumerate(class_colors)}
#     else:
#         raise ValueError("Unknown results type: ", type(results))
        
#     return objects, bg_objects, class_colors

def load_result(result_path):
    # check if theres a potential symlink for result_path and resolve it
    potential_path = os.path.realpath(result_path)
    if potential_path != result_path:
        print(f"Resolved symlink for result_path: {result_path} -> \n{potential_path}")
        result_path = potential_path
    with gzip.open(result_path, "rb") as f:
        results = pickle.load(f)

    if not isinstance(results, dict):
        raise ValueError("Results should be a dictionary! other types are not supported!")
    
    objects = MapObjectList()
    objects.load_serializable(results["objects"])
    bg_objects = MapObjectList()
    bg_objects.extend(obj for obj in objects if obj['is_background'])
    if len(bg_objects) == 0:
        bg_objects = None
    class_colors = results['class_colors']
        
    
        
    return objects, bg_objects, class_colors

def create_robot_initial_position(center, radius=0.1, color=(0, 1, 0)):
    """
    Create a robot initial position represented as a small sphere.
    
    Args:
    - center (tuple): (x, y, z) coordinates for the center of the sphere.
    - radius (float): Radius of the sphere.
    - color (tuple): RGB values in the range [0, 1] for the color of the sphere.
    
    Returns:
    - o3d.geometry.TriangleMesh: Colored mesh sphere representing the robot.
    """
    return create_ball_mesh(center, radius, color)

    
def main(args):
    result_path = args.result_path
    # result_path = '/data1/haoxuan/concept-graphs/conceptgraph/Datasets/Replica/office0/pcd_saves/full_pcd_ram_withbg_allclasses_overlap_maskconf0.25_simsum1.2_dbscan.1_post.pkl.gz'

    rgb_pcd_path = args.rgb_pcd_path
    global goal_pcd, robot_initial_position, line_mesh
    robot_initial_position = None
    line_mesh = None
    goal_pcd = None
    assert not (result_path is None and rgb_pcd_path is None), \
        "Either result_path or rgb_pcd_path must be provided."

    if rgb_pcd_path is not None:        
        pointclouds = Pointclouds.load_pointcloud_from_h5(rgb_pcd_path)
        global_pcd = pointclouds.open3d(0, include_colors=True)
        
        if result_path is None:
            print("Only visualizing the pointcloud...")
            o3d.visualization.draw_geometries([global_pcd])
            exit()
        
    objects, bg_objects, class_colors = load_result(result_path)
    
    if args.edge_file is not None:
        # Load edge files and create meshes for the scene graph
        scene_graph_geometries = []
        with open(args.edge_file, "r") as f:
            edges = json.load(f)
        
        classes = objects.get_most_common_class()
        colors = [class_colors[str(c)] for c in classes]
        obj_centers = []
        for obj, c in zip(objects, colors):
            pcd = obj['pcd']
            bbox = obj['bbox']
            points = np.asarray(pcd.points)
            center = np.mean(points, axis=0)
            extent = bbox.get_max_bound()
            extent = np.linalg.norm(extent)
            # radius = extent ** 0.5 / 25
            radius = 0.10
            obj_centers.append(center)

            # remove the nodes on the ceiling, for better visualization
            ball = create_ball_mesh(center, radius, c)
            scene_graph_geometries.append(ball)
            
        for edge in edges:
            if edge['object_relation'] == "none of these":
                continue
            id1 = edge["object1"]['id']
            id2 = edge["object2"]['id']

            line_mesh = LineMesh(
                points = np.array([obj_centers[id1], obj_centers[id2]]),
                lines = np.array([[0, 1]]),
                colors = [1, 0, 0],
                radius=0.02
            )

            scene_graph_geometries.extend(line_mesh.cylinder_segments)
    
    if not args.no_clip:
        print("Initializing CLIP model...")
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
        clip_model = clip_model.to("cuda")
        clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
        print("Done initializing CLIP model.")

    cmap = matplotlib.colormaps.get_cmap("turbo")
    
    if bg_objects is not None:
        indices_bg = np.arange(len(objects), len(objects) + len(bg_objects))
        objects.extend(bg_objects)
        
    # Sub-sample the point cloud for better interactive experience
    for i in range(len(objects)):
        pcd = objects[i]['pcd']
        pcd = pcd.voxel_down_sample(0.05)
        objects[i]['pcd'] = pcd
    
    pcds = copy.deepcopy(objects.get_values("pcd"))
    bboxes = copy.deepcopy(objects.get_values("bbox"))
    
    # Get the color for each object when colored by their class
    object_classes = []
    for i in range(len(objects)):
        obj = objects[i]
        pcd = pcds[i]
        obj_classes = np.asarray(obj['class_id'])
        # Get the most common class for this object as the class
        values, counts = np.unique(obj_classes, return_counts=True)
        obj_class = values[np.argmax(counts)]
        object_classes.append(obj_class)
    
    # Set the title of the window
    vis = o3d.visualization.VisualizerWithKeyCallback()

    if result_path is not None:
        vis.create_window(window_name=f'Open3D - {os.path.basename(result_path)}', width=1280, height=720)
    else:
        vis.create_window(window_name=f'Open3D', width=1280, height=720)

    # Add geometry to the scene
    for geometry in pcds + bboxes:
        vis.add_geometry(geometry)
        
    main.show_bg_pcd = True
    def toggle_bg_pcd(vis):
        if bg_objects is None:
            print("No background objects found.")
            return
        
        for idx in indices_bg:
            if main.show_bg_pcd:
                vis.remove_geometry(pcds[idx], reset_bounding_box=False)
                vis.remove_geometry(bboxes[idx], reset_bounding_box=False)
            else:
                vis.add_geometry(pcds[idx], reset_bounding_box=False)
                vis.add_geometry(bboxes[idx], reset_bounding_box=False)
        
        main.show_bg_pcd = not main.show_bg_pcd
        
    main.show_global_pcd = False
    def toggle_global_pcd(vis):
        if args.rgb_pcd_path is None:
            print("No RGB pcd path provided.")
            return
        
        if main.show_global_pcd:
            vis.remove_geometry(global_pcd, reset_bounding_box=False)
        else:
            vis.add_geometry(global_pcd, reset_bounding_box=False)
        
        main.show_global_pcd = not main.show_global_pcd
        
    main.show_scene_graph = False
    def toggle_scene_graph(vis):
        if args.edge_file is None:
            print("No edge file provided.")
            return
        
        if main.show_scene_graph:
            for geometry in scene_graph_geometries:
                vis.remove_geometry(geometry, reset_bounding_box=False)
        else:
            for geometry in scene_graph_geometries:
                vis.add_geometry(geometry, reset_bounding_box=False)
        
        main.show_scene_graph = not main.show_scene_graph
        
    def color_by_class(vis):
        for i in range(len(objects)):
            pcd = pcds[i]
            obj_class = object_classes[i]
            pcd.colors = o3d.utility.Vector3dVector(
                np.tile(
                    class_colors[str(obj_class)],
                    (len(pcd.points), 1)
                )
            )

        for pcd in pcds:
            vis.update_geometry(pcd)
            
    def color_by_rgb(vis):
        for i in range(len(pcds)):
            pcd = pcds[i]
            pcd.colors = objects[i]['pcd'].colors
        
        for pcd in pcds:
            vis.update_geometry(pcd)
            
    def color_by_instance(vis):
        instance_colors = cmap(np.linspace(0, 1, len(pcds)))
        for i in range(len(pcds)):
            pcd = pcds[i]
            pcd.colors = o3d.utility.Vector3dVector(
                np.tile(
                    instance_colors[i, :3],
                    (len(pcd.points), 1)
                )
            )
            
        for pcd in pcds:
            vis.update_geometry(pcd)

    def randomize_robot_position(vis):
        """
        Randomize the robot's initial position and replan the path to the target.
        """
        global robot_initial_position, line_mesh, goal_pcd

        # Remove the old robot position and path
        if robot_initial_position is not None:
            vis.remove_geometry(robot_initial_position)
        if line_mesh is not None:
            for segment in line_mesh.cylinder_segments:
                vis.remove_geometry(segment)

        # Randomize the robot's initial position
        # new_center = (random.uniform(-2, 2), random.uniform(-2, 2), -0.5)
    
        new_center = (random.uniform(-2, 2), random.uniform(-1, 1), -0.8 )
        # point1 = (new_center[0] + 0.5, new_center[1], new_center[2])
        # point2 = (new_center[0], new_center[1] + 0.5, new_center[2])
        # point3 = (new_center[0], new_center[1], new_center[2] + 0.5)

        robot_initial_position = create_robot_initial_position(new_center)
        # new_center_point1 = create_ball_mesh(point1, 0.05, (1 ,0 ,0))
        # new_center_point2 = create_ball_mesh(point2, 0.05, (1 ,0 ,0))
        # new_center_point3 = create_ball_mesh(point3, 0.05, (0 ,1 ,0))
        # vis.add_geometry(new_center_point1)
        # vis.add_geometry(new_center_point2)
        # vis.add_geometry(new_center_point3)
        vis.add_geometry(robot_initial_position)
        print(f"New robot initial position at {new_center}")
        # Get obstacles from the 3D map
        obstacles = [obj['pcd'] for obj in objects]
            # 打印障碍物信息
        for pcd in obstacles:
            print(f"Obstacle points: {np.asarray(pcd.points).shape}")

    
        # Plan the path using A* algorithm
        path = a_star(new_center, goal_pcd, obstacles)
        if not path or len(path) < 2:
            print("No valid path found.")
            return


        # Create the path as a line mesh
        path_points = np.array(path)
        path_lines = np.array([[i, i + 1] for i in range(len(path) - 1)])
            # 添加调试信息
        print(f"path_points shape: {path_points.shape}")
        print(f"path_lines shape: {path_lines.shape}")
        path_colors = [[0, 0, 1]] * len(path_lines)  # 蓝色
        line_mesh = LineMesh(points=path_points, lines=path_lines, colors=path_colors, radius=0.05)
        for segment in line_mesh.cylinder_segments:
            vis.add_geometry(segment)
        print("Replanned path from new initial position to target")

        vis.update_renderer()

    def color_by_clip_sim(vis):
        global goal_pcd, line_mesh, robot_initial_position
        if args.no_clip:
            print("CLIP model is not initialized.")
            return

        text_query = input("Enter your query: ")
        text_queries = [text_query]
        
        text_queries_tokenized = clip_tokenizer(text_queries).to("cuda")
        text_query_ft = clip_model.encode_text(text_queries_tokenized)
        text_query_ft = text_query_ft / text_query_ft.norm(dim=-1, keepdim=True)
        text_query_ft = text_query_ft.squeeze()
        
        # similarities = objects.compute_similarities(text_query_ft)
        objects_clip_fts = objects.get_stacked_values_torch("clip_ft")
        objects_clip_fts = objects_clip_fts.to("cuda")
        similarities = F.cosine_similarity(
            text_query_ft.unsqueeze(0), objects_clip_fts, dim=-1
        )
        max_value = similarities.max()
        min_value = similarities.min()
        normalized_similarities = (similarities - min_value) / (max_value - min_value)
        probs = F.softmax(similarities, dim=0)
        max_prob_idx = torch.argmax(probs)
        similarity_colors = cmap(normalized_similarities.detach().cpu().numpy())[..., :3]
        
        # 打印与输入文本最相似的物体类别
        most_similar_object = objects[max_prob_idx.item()]
        most_similar_object_class = most_similar_object['class_name']
        print(f"The most similar object class to '{text_query}' is: {most_similar_object_class}")
        print(f"The most similar object id is '{most_similar_object['image_idx']}'")
  
        # for i in range(len(objects)):
        #     pcd = pcds[i]
        #     map_colors = np.asarray(pcd.colors)
        #     pcd.colors = o3d.utility.Vector3dVector(
        #         np.tile(
        #             [
        #                 similarity_colors[i, 0].item(),
        #                 similarity_colors[i, 1].item(),
        #                 similarity_colors[i, 2].item()
        #             ], 
        #             (len(pcd.points), 1)
        #         )
        #     )
        for i in range(len(objects)):
            pcd = pcds[i]
            if i == max_prob_idx.item():
                # 上色为红色
                pcd.colors = o3d.utility.Vector3dVector(
                    np.tile(
                        [1, 0, 0],  # 红色
                        (len(pcd.points), 1)
                    )
                )
            vis.update_geometry(pcd)

        vis.poll_events()
        vis.update_renderer()
            # 清理之前的路径
        if robot_initial_position is not None:
            vis.remove_geometry(robot_initial_position)
        if line_mesh is not None:
            for segment in line_mesh.cylinder_segments:
                vis.remove_geometry(segment)
        # 生成从初始位置到最匹配物体的路径
        robot_center = (1.4, - 0.2, -0.8)
        robot_initial_position = create_robot_initial_position(robot_center)
        vis.add_geometry(robot_initial_position)
        print(f"Added robot initial position at {robot_center}")
        
        goal_pcd = objects[max_prob_idx.item()]['pcd']
        # print(f"Target bounding box: {goal_pcd}")

        # 获取3D地图中的障碍物
        obstacles = [obj['pcd'] for obj in objects]

        # 使用A*算法规划路径
        path = a_star(robot_center, goal_pcd, obstacles)
        if not path or len(path) < 2:
            print("No valid path found.")
            return


        # 创建路径的线网格
        path_points = np.array(path)
        path_lines = np.array([[i, i + 1] for i in range(len(path) - 1)])
        # 添加调试信息
        print(f"path_points shape: {path_points.shape}")
        print(f"path_lines shape: {path_lines.shape}")
        path_colors = [[0, 0, 1]] * len(path_lines)  # 蓝色
        line_mesh = LineMesh(points=path_points, lines=path_lines, colors=path_colors, radius=0.05)
        for segment in line_mesh.cylinder_segments:
            vis.add_geometry(segment)
        print("Added path from initial position to target")
        # vis.poll_events()
        # vis.update_geometry(pcd)
            
    def save_view_params(vis):
        param = vis.get_view_control().convert_to_pinhole_camera_parameters()
        o3d.io.write_pinhole_camera_parameters("temp.json", param)
        
    vis.register_key_callback(ord("B"), toggle_bg_pcd)
    vis.register_key_callback(ord("S"), toggle_global_pcd)
    vis.register_key_callback(ord("C"), color_by_class)
    vis.register_key_callback(ord("R"), color_by_rgb)
    vis.register_key_callback(ord("F"), color_by_clip_sim)
    vis.register_key_callback(ord("I"), color_by_instance)
    vis.register_key_callback(ord("V"), save_view_params)
    vis.register_key_callback(ord("G"), toggle_scene_graph)
    vis.register_key_callback(ord("L"), randomize_robot_position)
    
    # Render the scene
    vis.run()
    
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)