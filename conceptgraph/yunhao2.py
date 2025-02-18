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
import re
import distinctipy
import math
from gradslam.structures.pointclouds import Pointclouds
from openai import OpenAI
from conceptgraph.slam.slam_classes import MapObjectList
from conceptgraph.utils.vis import LineMesh
from conceptgraph.slam.utils import filter_objects, merge_objects
import heapq

def adjust_path(path, initial_position):
    adjusted_path = []
    path2d = []
    for coord in path:
        adjusted_coord = (
            coord[0] - initial_position[0],
            coord[1] - initial_position[1],
            # coord[2] - initial_position[2]
        )
        adjusted_path.append(adjusted_coord)
        path2d.append((adjusted_coord[0], adjusted_coord[1]))
    return adjusted_path, path2d


def get_openai_client():
    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY')
    )
    return client

def generate_robot_commands(path2d, initial_direction=0):
    """
    将二维路径点转换为机器人移动和转向指令，支持八方向。
    """
    import math

    if not path2d or len(path2d) < 2:
        return []

    commands = []
    current_direction = initial_direction  
    move_distance = 0.0  
    EPSILON = 1e-6

    # 增加对角方向的角度映射
    direction_angles = {
        'positive_x': 0,
        'positive_y': 90,
        'negative_x': 180,
        'negative_y': 270,
        'diagonal_posx_posy': 45,
        'diagonal_posx_negy': 315,
        'diagonal_negx_posy': 135,
        'diagonal_negx_negy': 225
    }

    def get_movement_direction(p1, p2):
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        angle = math.degrees(math.atan2(dy, dx)) % 360
        # 判断角度并映射到八方向
        if  -22.5 <= angle < 22.5:
            return 'positive_x'
        elif 22.5 <= angle < 67.5:
            return 'diagonal_posx_posy'
        elif 67.5 <= angle < 112.5:
            return 'positive_y'
        elif 112.5 <= angle < 157.5:
            return 'diagonal_negx_posy'
        elif 157.5 <= angle < 202.5:
            return 'negative_x'
        elif 202.5 <= angle < 247.5:
            return 'diagonal_negx_negy'
        elif 247.5 <= angle < 292.5:
            return 'negative_y'
        else:
            return 'diagonal_posx_negy'

    def calculate_turn(current_dir, desired_dir):
        difference = (desired_dir - current_dir) % 360
        # 判断转向（示例：只处理90°、45°等常见转弯）
        if difference == 0:
            return None
        elif difference == 45:
            return "left 45°"
        elif difference == 90:
            return "left 90°"
        elif difference == 135:
            return "left 135°"
        elif difference == 180:
            return "180°"
        elif difference == 225:
            return "right 135°"
        elif difference == 270:
            return "right 90°"
        elif difference == 315:
            return "right 45°"
        else:
            return None

    previous_direction = get_movement_direction(path2d[0], path2d[1])
    desired_angle = direction_angles[previous_direction]

    for i in range(1, len(path2d)):
        p_prev = path2d[i - 1]
        p_curr = path2d[i]
        movement = get_movement_direction(p_prev, p_curr)

        if movement != previous_direction:
            desired_angle = direction_angles[movement]
            turn = calculate_turn(current_direction, desired_angle)
            if move_distance > EPSILON:
                commands.append(f"Move forward {move_distance:.2f}m")
                move_distance = 0.0
            if turn:
                commands.append(f"Turn {turn}")
                # 更新机器人朝向，只考虑最终角度
                current_direction = desired_angle
            previous_direction = movement

        distance = math.hypot(p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
        move_distance += distance

    if move_distance > EPSILON:
        commands.append(f"Move forward {move_distance:.2f}m")
    print(f"Generated commands: {commands}")
    return commands


def detect_turns(path, angle_threshold=0.2):
    turn_points = []
    for i in range(1, len(path) - 1):
        x1, y1 = path[i - 1][:2]  # 取路径点的前两个元素
        x2, y2 = path[i][:2]      # 取路径点的前两个元素
        x3, y3 = path[i + 1][:2]  # 取路径点的前两个元素

        # 计算两段路径之间的角度变化
        angle1 = math.atan2(y2 - y1, x2 - x1)
        angle2 = math.atan2(y3 - y2, x3 - x2)
        
        angle_diff = abs(angle2 - angle1)
        # 如果角度变化超过阈值，则认为发生了转弯
        if angle_diff > angle_threshold:
            turn_points.append(i)  # 保存发生转弯的点
    return turn_points

# 二次贝塞尔曲线
# def quadratic_bezier(P0, P1, P2, num_points=100):
#     t = np.linspace(0, 1, num_points)
#     B = (1 - t) ** 2 * P0 + 2 * (1 - t) * t * P1 + t ** 2 * P2
#     return B
def quadratic_bezier(P0, P1, P2, num_points=20):
    t = np.linspace(0, 1, num_points)  # 在 [0, 1] 范围内生成 num_points 个点
    # 只取x和y坐标，忽略z坐标
    B_x = (1 - t) ** 2 * P0[0] + 2 * (1 - t) * t * P1[0] + t ** 2 * P2[0]
    B_y = (1 - t) ** 2 * P0[1] + 2 * (1 - t) * t * P1[1] + t ** 2 * P2[1]
    
    # 保留原始的z坐标
    B_z = P0[2]  # 假设z坐标是相同的（如果有不同的z值，你可以选择P1[2]或P2[2]）
    
    # 将x, y坐标与z坐标重新合并为一个二维数组
    B = np.vstack((B_x, B_y)).T  # 使得每行是(x, y)对
    B = np.hstack((B, np.full((B.shape[0], 1), B_z)))  # 加回z坐标
    
    return B

# 碰撞检测函数
def is_collision(path, obstacles, vehicle_radius=0.3):
    for point in path:
        x, y, _ = point  # 只检查x, y
        for pcd in obstacles:
            points = np.asarray(pcd.points)
            distances = np.linalg.norm(points[:, :2] - np.array((x, y)), axis=1)
            if np.any(distances < vehicle_radius):  # 检查路径点是否与障碍物碰撞
                return True
    return False

# 平滑路径函数
def smooth_path(path, num_points=10, angle_threshold=0.3):
    smoothed_path = []

    # 检查路径中的转弯
    def detect_turns(path, angle_threshold=0.2):
        turn_points = []
        for i in range(1, len(path) - 1):
            x1, y1 = path[i - 1][:2]  # 只取x和y
            x2, y2 = path[i][:2]
            x3, y3 = path[i + 1][:2]

            # 计算两段路径之间的角度变化
            angle1 = math.atan2(y2 - y1, x2 - x1)
            angle2 = math.atan2(y3 - y2, x3 - x2)
            
            angle_diff = abs(angle2 - angle1)
            # 如果角度变化超过阈值，则认为发生了转弯
            if angle_diff > angle_threshold:
                turn_points.append(i)  # 保存发生转弯的点
        return turn_points

    # 检测转弯点
    turn_points = detect_turns(path, angle_threshold)

    # 逐步对转弯点进行贝塞尔曲线平滑
    for i in range(0, len(turn_points)):
        P0 = path[turn_points[i] - 1]
        P1 = path[turn_points[i]]
        P2 = path[turn_points[i] + 1] if i + 1 < len(path) else path[-1]
        
        # 生成二次贝塞尔曲线
        smoothed_segment = quadratic_bezier(P0, P1, P2, num_points)
        for i in range(0, turn_points[i - 1]):
            smoothed_path.append([path[i]])
        smoothed_path.append(smoothed_segment)
    
    # 添加原始的直行路径部分
    last_index = turn_points[-1] if turn_points else len(path) - 1
    for i in range(last_index + 1, len(path)):
        smoothed_path.append([path[i]])

    # 合并所有平滑路径段
    smoothed_path = np.concatenate(smoothed_path, axis=0)
    
    return smoothed_path

def a_star(start, goal_pcd, obstacles, grid_size=0.5, goal_threshold=0.6):
    import heapq
    import math
    import numpy as np

    # 计算位置距离 + 转向惩罚
    def cost_with_turn_penalty(current, neighbor, turn_penalty=0.3):
        # current: (x, y, dir)
        # neighbor: (nx, ny, ndir)
        pos_dist = np.linalg.norm(np.array(current[:2]) - np.array(neighbor[:2]))
        # 如果方向不同，则增加转向代价
        if current[2] != neighbor[2]:
            return pos_dist + turn_penalty
        return pos_dist

    def is_collision(point, obstacles, threshold=0.07):
        # ...existing code...
        x, y, _ = point
        for pcd in obstacles:
            points = np.asarray(pcd.points)
            mask = (points[:, 2] >= (start[2] - 1.5)) & (points[:, 2] <= start[2])
            filtered_points = points[mask]
            distances = np.linalg.norm(filtered_points[:, :2] - np.array((x, y)), axis=1)
            if np.any(distances < threshold):
                return True
        return False

    def is_goal(point, goal_pcd, threshold):
        # ...existing code...
        x, y, _ = point
        points = np.asarray(goal_pcd.points)
        distances = np.linalg.norm(points[:, :2] - [x, y], axis=1)
        return np.any(distances < threshold)

    # 八方向的离散角度（每45°一个方向）
    directions = [i * 45 for i in range(8)]

    # 构造初始状态(方向先设为0°)，和目标位置中只关心(x,y)
    start_state = (start[0], start[1], 0)
    goal_center = np.mean(np.asarray(goal_pcd.points), axis=0)
    goal_center = (goal_center[0], goal_center[1])

    # f_score,g_score,opened等需要将方向纳入状态
    opened = []
    heapq.heappush(opened, (0, start_state))
    came_from = {}
    g_score = {start_state: 0}
    f_score = {start_state: 0}
    closest_state = start_state
    closest_dist = float('inf')

    while opened:
        _, current = heapq.heappop(opened)
        if is_goal(current, goal_pcd, goal_threshold):
            # 回溯路径并恢复z值
            path = []
            while current in came_from:
                path.append((current[0], current[1], start[2]))
                current = came_from[current]
            path.append((start[0], start[1], start[2]))
            path.reverse()
            print(f"Success to get the goal area\nGenerated path: {path}")
            # return smooth_path(path)
            return path

        for d in directions:
            # 下一步：以方向d移动一个grid_size
            rad = math.radians(d)
            nx = current[0] + grid_size * math.cos(rad)
            ny = current[1] + grid_size * math.sin(rad)
            neighbor = (nx, ny, d)
            if is_collision(neighbor, obstacles):
                continue

            # 计算 g_score
            tentative_g = g_score[current] + cost_with_turn_penalty(current, neighbor, turn_penalty=0.3)
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                # 简单启发=到目标的直线距离
                h = np.linalg.norm(np.array((nx, ny)) - np.array(goal_center))
                f_score[neighbor] = tentative_g + h
                heapq.heappush(opened, (f_score[neighbor], neighbor))

                if h < closest_dist:
                    closest_state = neighbor
                    closest_dist = h

    # 未找到可达路径，回溯离目标最近点
    path = []
    curr = closest_state
    while curr in came_from:
        path.append((curr[0], curr[1], start[2]))
        curr = came_from[curr]
    path.append((start[0], start[1], start[2]))
    path.reverse()
    print("Fail to get the goal area\nReturning fallback path:", path)
    # return smooth_path(path)
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

    result_path = '/data1/haoxuan/concept-graphs/pcd_r_mapping_stride20.pkl.gz'

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
        
    if not args.no_clip:
        print("Initializing CLIP model...")
        clip_model, _, clip_preprocess = open_clip.create_model_and_transforms("ViT-H-14", "laion2b_s32b_b79k")
        clip_model = clip_model.to("cuda")
        clip_tokenizer = open_clip.get_tokenizer("ViT-H-14")
        print("Done initializing CLIP model.")

    cmap = matplotlib.colormaps.get_cmap("turbo")

    if bg_objects is not None:
        print(f"Loaded {len(objects)} objects and {len(bg_objects)} background objects.")
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
    

    # gpt query
    # input query
    text_query = input("Enter your query: ")
    text_queries = [text_query]
    
    # object json
    prospective_json = []
    for prospective_idx in range(len(objects)):
        prospective_object = objects[prospective_idx]
        object_dict = { 
            # "id" : prospective_object['curr_obj_num'],
            "id" : prospective_idx,
            "bbox_extent" : prospective_object['bbox'].extent,
            "bbox_center" : prospective_object['bbox'].center,
            "object_tag" : prospective_object['class_name'],
            "caption" : prospective_object['consolidated_caption'] 
        }
        prospective_json.append(object_dict)
    print(prospective_json)
    
    # prompting
    system_prompt = '''
The input to the model is a 3D scene described in a JSON format. Each entry in the JSON describes one object in the scene, with the following five fields:

1."id": a unique object id; 
2."bbox_extent": extents of the 3D bounding box for the object; 
3."bbox_center": centroid of the 3D bounding box for the object; 
4."object_tag": a brief (but sometimes inaccurate) tag categorizing the object; 
5."caption": a brief caption for the object.

Once you have parsed the JSON and are ready to answer questions about the scene, say "I'm ready".

The user will then begin to ask questions,and the task is to answer various user queries about the 3D scene. 

For each user question, respond with a JSON dictionary with the following fields:

1."inferred_query": your interpretaion of the user query in a succinct form; 
2."relevant_objects": list of relevant object ids for the user query (if applicable); 
3."query_achievable": whether or not the user specified query is achievable using the objects and descriptions provided in the 3Dscene; 
4."final_relevant_objects": A finallist of objects relevant to the user-specified task. As much as possible, sort all objects in this list such that the most relevant object is listed first, followed by the second most relevant, and so on; 
5."explanation": A brief explanation of what the most relevant object(s) is(are), and how they achieve the user-specified task. 

For example, if user asks "I need a quick gift. Help!", then you should respone something like this:
{
inferred_query: Find suitable object for a gift.,
relevant_objects: [0,6,7,23,25,31],
query_achievable: true,
final_relevant_objects: [6],
explanation: The most suitable object for a gift could be the ceramic vase (id 6). Vases are commonly gifted items and this one could potentially be filled with a plantor flower arrangement, making alovely present.     
}
    '''
    user_query_1 = f'''
Here is a list of the JSON dictionary for objects in the scene: {prospective_json}, please read it carefully.
    '''
    user_query_2 = f'''
Here is the user's query: {text_queries}. Please compare it to the list of object-oriented JSON dictionary. 
    
Do not include any other information in your response. Only output a respond with a JSON dictionary with the following fields:
                        
1."inferred_query": your interpretaion of the user query in a succinct form; 
2."relevant_objects": list of relevant object ids for the user query (if applicable); 
3."query_achievable": whether or not the user specified query is achievable using the objects and descriptions provided in the 3Dscene; 
4."final_relevant_objects": A finallist of objects relevant to the user-specified task. As much as possible, sort all objects in this list such that the most relevant object is listed first, followed by the second most relevant, and so on; 
5."explanation": A brief explanation of what the most relevant object(s) is(are), and how they achieve the user-specified task. 

For example, if user asks "I need a quick gift. Help!", then you should respone something like this:
{ '{' }
inferred_query: Find suitable object for a gift.,
relevant_objects: [0,6,7,23,25,31],
query_achievable: true,
final_relevant_objects: [6],
explanation: The most suitable object for a gift could be the ceramic vase (id 6). Vases are commonly gifted items and this one could potentially be filled with a plantor flower arrangement, making alovely present.     
{ '}' }
    '''
    
    vlm_answer = []
    try:
        client = get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_query_1
                },
                {
                    "role": "user",
                    "content": user_query_2
                }
            ]
        )
        
        vlm_answer_str = response.choices[0].message.content
        print(f"vlm_answer_str: {vlm_answer_str}")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
    
    # Parsing
    final_match = re.search(r'\{.*?\}', vlm_answer_str, re.DOTALL)
    final_data = json.loads(final_match.group(0))
    # print(final_data["final_relevant_objects"])
    # final_idx_match = re.search(r'\[(.*?)\]', str(final_data["final_relevant_objects"]), re.DOTALL)
    # # print(final_idx_match.group(1))
    # final_idx = int(final_idx_match.group(1))
    # # print(final_idx)  
    # final_obj = objects[final_idx]

    # 假设 final_data["final_relevant_objects"] 是一个字符串
    final_data_str = str(final_data["final_relevant_objects"])
    # 使用正则表达式匹配第一组数字
    match = re.search(r'\d+', final_data_str)

    # 如果匹配成功，提取第一组数字
    if match:
        final_idx = int(match.group(0))
    else:
        final_idx = None

    print(final_idx)
    # final_obj = objects[final_idx]

    robot_center = (-2.670378774354216e-02, 5.156578799482670e-03, 6.656220113593306e-01)
    robot_initial_position = create_robot_initial_position(robot_center)

    print(f"Added robot initial position at {robot_center}")
    
    goal_pcd = objects[final_idx]['pcd']
    
    # 获取3D地图中的障碍物
    obstacles = [obj['pcd'] for obj in objects]

    # 使用A*算法规划路径
    path = a_star(robot_center, goal_pcd, obstacles)
    if not path or len(path) < 2:
        print("No valid path found.")
        return
    path,path2d = adjust_path(path, robot_center)
    with open('path2d.json', 'w') as f:
        json.dump(path2d, f)
    commands = generate_robot_commands(path2d)
    print("Robot Commands:")
    for cmd in commands:
        print(cmd)


    
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)