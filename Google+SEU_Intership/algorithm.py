import cv2
import torch
import numpy as np
from geometry import get_min_reprojection_error


def transform_closure(X_bin):
    """
    Convert binary relation matrix to permutation matrix
    :param X_bin: torch.tensor which is binarized by a threshold
    :return:
    """
    temp = torch.zeros_like(X_bin)
    N = X_bin.shape[0]
    for k in range(N):
        for i in range(N):
            for j in range(N):
                temp[i][j] = X_bin[i, j] or (X_bin[i, k] and X_bin[k, j])
    vis = torch.zeros(N)
    match_mat = torch.zeros_like(X_bin)
    for i, row in enumerate(temp):
        if vis[i]:
            continue
        for j, is_relative in enumerate(row):
            if is_relative:
                vis[j] = 1
                match_mat[j, i] = 1
    return match_mat


def top_down_pose_kernel(dataset, geo_affinity_mat, matched_list, pose_mat, sub_imgid2cam):
    multi_pose3d = list()  # 存储每个人的3D姿态
    chosen_img = list()  # 存储用于3D姿态计算的图像ID

    for person in matched_list:  # 遍历每个匹配的列表（每个人的子图ID集合）
        # 从地理亲和矩阵中提取对应的子图的亲和图
        Graph = geo_affinity_mat[person][:, person].clone().numpy()
        # print(geo_affinity_mat[person][:, person].shape)
        Graph *= (1 - np.eye(Graph.shape[0]))  # 将对角线元素置为0，排除自环

        if len(Graph) < 2:  # 如果节点少于2个，跳过
            continue
        elif len(Graph) > 2:  # 如果节点多于2个
            # 选择具有最小重投影误差的子图ID
            sub_imageid = get_min_reprojection_error(person, dataset, pose_mat, sub_imgid2cam)
        else:  # 如果恰好2个节点
            sub_imageid = person  # 直接使用这两个节点

        # 根据亲和度矩阵对子图ID进行排序，选择前两个子图ID
        _, rank = torch.sort(geo_affinity_mat[sub_imageid][:, sub_imageid].sum(dim=0))

        sub_imageid = sub_imageid[rank[:2]]

        # 获取这两个子图的相机ID
        cam_id_0, cam_id_1 = sub_imgid2cam[sub_imageid[0]], sub_imgid2cam[sub_imageid[1]]

        # 获取对应相机的投影矩阵
        projmat_0, projmat_1 = dataset.P[cam_id_0], dataset.P[cam_id_1]

        # 获取对应子图的2D姿态点
        pose2d_0, pose2d_1 = pose_mat[sub_imageid[0]].T, pose_mat[sub_imageid[1]].T

        # 使用 OpenCV 计算 3D 姿态
        pose3d_homo = cv2.triangulatePoints(projmat_0, projmat_1, pose2d_0, pose2d_1)

        # 将齐次坐标转换为 3D 坐标
        pose3d = pose3d_homo[:3] / (pose3d_homo[3] + 10e-6)

        # 将计算得到的 3D 姿态添加到结果列表中
        multi_pose3d.append(pose3d)
        chosen_img.append(sub_imageid)  # 添加选择的子图ID

    return multi_pose3d, chosen_img

