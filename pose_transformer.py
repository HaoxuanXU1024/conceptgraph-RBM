import os
import re
import sys
import ast
import shutil
import argparse
import configparser
import numpy as np
from datetime import datetime
from PIL import Image


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str, default='/data1/tianfu/datasets/hkustgz/')
    parser.add_argument('--scene', type=str, default='living_room')
    parser.add_argument('--mode', type=str, default='path_only')

    args = parser.parse_args()
    return args


class PosePrecessor:
    def __init__(self, usr_root, scene_name):
        self.usr_root = usr_root
        self.scene_name = scene_name

        self.poses_path_dir = 'path/camera_parameter/camera_parameter.conf'
        self.color_path_dir = 'path/color_image/'
        self.depth_path_dir = 'path/depth_image/'

        self.poses_gimbal_dir = 'gimbal/camera_parameter/camera_parameter.conf'
        self.color_gimbal_dir = 'gimbal/color_image/'
        self.depth_gimbal_dir = 'gimbal/depth_image/'

        self.L2C_TRANSFORM = np.array([[-0.00859563, -0.999714,   -0.0223111, 0.01927  ],
                                       [-0.0428589,   0.0227041,  -1.00387,   0.699481 ],
                                       [ 0.999075,   -0.00764551, -0.0423244, 0.0611374],
                                       [ 0,           0,           0,         1        ]])

        self.path_info = None
        self.gimbal_info = None


    def load_path_conf(self):
        file_dir = os.path.join(self.usr_root, self.scene_name, self.poses_path_dir)

        path_info = configparser.ConfigParser()
        path_info.read(file_dir)

        self.path_info = path_info


    def load_gimbal_conf(self):
        file_dir = os.path.join(self.usr_root, self.scene_name, self.poses_gimbal_dir)

        gimbal_info = configparser.ConfigParser()
        gimbal_info.read(file_dir)

        self.gimbal_info = gimbal_info


    def transfor_path_coordinate(self):
        color_list = []
        depth_list = []
        poses_list = []

        for section in self.path_info.sections():
            print(f'[INFO] section: {section}')

            if 'rgb_name' in self.path_info[section]:
                color_list_str = self.path_info.get(section, 'rgb_name')
                if isinstance(color_list_str, str):
                    color_list_real = ast.literal_eval(color_list_str)
                    if isinstance(color_list_real, list):
                        color_list.extend(color_list_real)

            if 'depth_name' in self.path_info[section]:
                depth_list_str = self.path_info.get(section, 'depth_name')
                if isinstance(depth_list_str, str):
                    depth_list_real = ast.literal_eval(depth_list_str)
                    if isinstance(depth_list_real, list):
                        depth_list.extend(depth_list_real)

            if 'lpose' in self.path_info[section]:
                '''
                data = self.path_info.get(section, 'lpose')
                # Define array in the local scope
                local_scope = {'array': np.array}
                # Evaluate the string representation of the arrays
                arrays = eval(data, {"__builtins__": None}, local_scope)

                for array in arrays:
                    T_l2w = array
                    T_l2c = self.L2C_TRANSFORM
                    T_c2w = np.dot(T_l2w, np.linalg.inv(T_l2c))
                    poses_list.append(T_c2w)
                '''

                lpose_str = self.path_info.get(section, 'lpose')
                lpose_str_cleaned = re.sub(r'\barray\(', '', lpose_str, flags=re.DOTALL)
                lpose_str_cleaned = re.sub(r'\)', '', lpose_str_cleaned, flags=re.DOTALL)
                lpose_list = ast.literal_eval(lpose_str_cleaned)

                for i, lpose in enumerate(lpose_list):
                    T_wl = np.array(lpose)
                    T_cl = self.L2C_TRANSFORM
                    # T_wc = T_wl · T_lc = T_wl · T_cl^-1
                    T_wc = np.dot(T_wl, np.linalg.inv(T_cl))
                    poses_list.append(T_wc)
                '''
                lpose_str = self.path_info.get(section, 'lpose')
                lpose_str_cleaned = re.sub(r'\barray\(', '', lpose_str, flags=re.DOTALL)
                lpose_str_cleaned = re.sub(r'\)', '', lpose_str_cleaned, flags=re.DOTALL)
                lpose_list = ast.literal_eval(lpose_str_cleaned)

                T_wp = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

                for i, lpose in enumerate(lpose_list):
                    T_pl = np.array(lpose)
                    T_wl = np.dot(T_wp, T_pl)
                    T_cl = self.L2C_TRANSFORM
                    # T_wc = T_wl · T_lc = T_wl · T_cl^-1
                    T_wc = np.dot(T_wl, np.linalg.inv(T_cl))
                    poses_list.append(T_wc)
                    T_wp = T_wl
                '''

        assert len(color_list) == len(depth_list) == len(poses_list)
        print(f'[ASSERT] {len(color_list)} == {len(depth_list)} == {len(poses_list)}')
        return color_list, depth_list, poses_list


    def save_results(self, mode):
        now = datetime.now()
        timestamp_str = now.strftime('%Y%m%d_%H%M%S_%f')
        result_dir = os.path.join(self.usr_root, self.scene_name, 'result', timestamp_str)
        if not os.path.exists(result_dir):
            os.makedirs(result_dir)

        if mode == 'path_only':
            color_list, depth_list, poses_list = self.transfor_path_coordinate()
            self.save_images_path_only(result_dir, color_list, depth_list)
        elif mode == 'path_with_gimbal':
            print("[ERROR] Under Implemention")
            sys.exit(0)
        else:
            print("[ERROR] Invalid Mode")
            sys.exit(0)

        self.save_poses(result_dir, poses_list)


    def save_images_path_only(self, result_dir, color_list, depth_list):
        raw_color_dir = os.path.join(self.usr_root, self.scene_name, self.color_path_dir)
        raw_depth_dir = os.path.join(self.usr_root, self.scene_name, self.depth_path_dir)
        target_dir = os.path.join(result_dir, 'results')
        os.makedirs(target_dir)

        for i, color_name in enumerate(color_list):
            raw = os.path.join(raw_color_dir, color_name)
            target_name = 'frame' + str(i).zfill(6) + '.jpg'  # jpg ?
            traget = os.path.join(target_dir, target_name)

            png_image = Image.open(raw)
            if png_image.mode != 'RGB':
                png_image = png_image.convert('RGB')
            png_image.save(traget, 'JPEG')

        for i, depth_name in enumerate(depth_list):
            raw = os.path.join(raw_depth_dir, depth_name)
            target_name = 'depth' + str(i).zfill(6) + '.png'
            traget = os.path.join(target_dir, target_name)
            shutil.copy(raw, traget)

        print("[INFO] saved all images (path only)")


    def save_poses(self, result_dir, poses_list):
        file_dir = os.path.join(result_dir, 'traj.txt')
        if os.path.exists(file_dir):
            os.remove(file_dir)

        j = -1
        for i, T in enumerate(poses_list):
            line = T.ravel()

            with open(file_dir, 'a') as f:
                np.savetxt(f, line, fmt='%.18e', newline=' ', delimiter=' ', comments='')
                f.close()

            with open(file_dir, 'r') as f:
                content = f.read()
                new_content = content.rstrip()
                f.close()

            with open(file_dir, 'w') as f:
                f.write(new_content)
                f.close()

            with open(file_dir, 'a') as f:
                f.write('\n')
                f.close()

            j = i + 1
        print(f"[INFO] saved all {j} poses")


def main(args):
    pose_processor = PosePrecessor(args.root, args.scene)
    pose_processor.load_path_conf()
    if args.mode == 'path_with_gimbal':
        pose_processor.load_gimbal_conf()
    pose_processor.save_results(args.mode)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
