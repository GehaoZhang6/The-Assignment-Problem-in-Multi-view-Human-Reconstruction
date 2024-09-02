import os
import json
from collections import defaultdict
import torch
from mem_dataset import MemDataset
from geometry import geometry_affinity
from matchSVT import matchSVT
from algorithm import transform_closure, top_down_pose_kernel
import numpy as np
import matplotlib.pyplot as plt


def read_json_files(base_dir):
    info_dict = defaultdict(dict)
    for cam_id in range(1, 9):
        cam_name = f'{cam_id}'
        cam_dir = os.path.join(base_dir, str(cam_id))
        for file_name in os.listdir(cam_dir):
            if file_name.endswith('_keypoints.json'):
                img_id = int(file_name.split('_')[0])
                file_path = os.path.join(cam_dir, file_name)
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    info_dict[cam_name][img_id] = data['people']
    return info_dict


def read_camera_parameters(file_path):
    camera_parameter = {'K': [], 'RT': []}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        num_cameras = len(lines) // 9  # 每个相机参数块有10行
        for i in range(num_cameras):
            start_idx = i * 9
            # 读取内参矩阵 K
            K = np.array([list(map(float, lines[start_idx + 1].split())),
                          list(map(float, lines[start_idx + 2].split())),
                          list(map(float, lines[start_idx + 3].split()))])
            # 读取外参矩阵 RT
            RT = np.array([list(map(float, lines[start_idx + 5].split())),
                           list(map(float, lines[start_idx + 6].split())),
                           list(map(float, lines[start_idx + 7].split()))])
            # 将 K 和 RT 添加到相应的列表中
            camera_parameter['K'].append(K)
            camera_parameter['RT'].append(RT)
    # 转换为 numpy 数组
    camera_parameter['K'] = np.array(camera_parameter['K'])
    camera_parameter['RT'] = np.array(camera_parameter['RT'])
    return camera_parameter




def update(frame,dataset,ax):
    img_id = frame
    dimGroup = dataset.dimGroup[img_id]

    info_list = list()
    for cam_id in dataset.cam_names:
        info_list += dataset.info_dict[cam_id][img_id]

    joint_num = 19
    pose_mat = np.array([i['pose_keypoints_2d'] for i in info_list]).reshape(-1, joint_num, 3)[..., :2]
    geo_affinity_mat = geometry_affinity(pose_mat.copy(), dataset.F.numpy(), dataset.dimGroup[img_id])
    geo_affinity_mat = torch.tensor(geo_affinity_mat)

    W = geo_affinity_mat
    W[torch.isnan(W)] = 0
    match_mat = matchSVT(W, dimGroup)
    rank_A = np.linalg.matrix_rank(match_mat)

    closure_mat = transform_closure(match_mat)
    bin_match = closure_mat[:, torch.nonzero(torch.sum(closure_mat, dim=0) > 1.9).squeeze()]
    bin_match = bin_match.reshape(W.shape[0], -1)

    matched_list = [[] for _ in range(bin_match.shape[1])]
    for sub_imgid, row in enumerate(bin_match.int()):
        if row.sum() != 0:
            pid = row.argmax()
            matched_list[pid].append(sub_imgid)
    matched_list = [np.array(i) for i in matched_list]

    sub_imgid2cam = np.zeros(pose_mat.shape[0], dtype=np.int32)
    for idx, i in enumerate(range(len(dimGroup) - 1)):
        sub_imgid2cam[dimGroup[i]:dimGroup[i + 1]] = idx

    multi_pose3d, chosen_img = top_down_pose_kernel(dataset, geo_affinity_mat, matched_list, pose_mat, sub_imgid2cam)

    data = np.asarray(multi_pose3d)
    print(data.shape)
    np.savez(f"./data/data_{frame:04d}.npz", data=data)
    ax.clear()
    x_range = (-3, 0.5)  # x轴范围
    y_range = (-2, 3)  # y轴范围
    z_range = (-0.25, 1.9)  # z轴范围
    ax.set_xlim(x_range)
    ax.set_ylim(y_range)
    ax.set_zlim(z_range)
    print(data.shape[0])
    for i in range(data.shape[0]):
        x = data[i, 0, :]
        y = data[i, 1, :]
        z = data[i, 2, :]
        ax.scatter(x, y, z, color='blue')

    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Z axis')
    ax.legend()
    plt.savefig(f"./test/frame_{frame:04d}.png")

if __name__ == "__main__" :
    point_dir = 'keypoints'
    info_dict = read_json_files(point_dir)
    # ===========================info_dict里面是字典套字典，camera->image->people最终是一个people的字典===============================
    parameter_path = 'camera.txt'
    camera_parameter = read_camera_parameters(parameter_path)
    dataset = MemDataset(info_dict=info_dict, camera_parameter=camera_parameter)
    # 创建三维图形对象
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    num_frames = len(dataset.dimGroup)
    for frame in range(1,num_frames):
        update(frame, dataset, ax)

#存在的问题：1.帧和帧之间人是怎么匹配出来的2.噪音怎么去除