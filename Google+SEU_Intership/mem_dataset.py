import sys
import os.path as osp

# Config project if not exist
project_path = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
if project_path not in sys.path:
    sys.path.insert(0, project_path)

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from collections import OrderedDict
from glob import glob
import os.path as osp
import json
import torch


class MemDataset(Dataset):
    """
    Datasets in memory to boost performance of whole pipeline
    """

    def __init__(self, info_dict, camera_parameter=None, template_name='Shelf'):
        self.args = dict(arch='resnet50', batch_size=128, camstyle=46,
                         dataset='market', dist_metric='euclidean', dropout=0.5, epochs=50, evaluate=False,
                         features=1024,
                         height=256, logs_dir='logs/market-ide-camstyle-re', lr=0.1, momentum=0.9,
                         output_feature='pool5',
                         print_freq=1, re=0.5, rerank=True, weight_decay=0.0005, width=128, workers=8,
                         resume='logs/market-ide-camstyle-re/checkpoint.pth.tar')
        self.info_dict = info_dict
        self.cam_names = sorted(info_dict.keys())

        self.dimGroup = OrderedDict()

        for img_id in [i for i in self.info_dict[self.cam_names[0]].keys() if i != 'image_data' and i != 'image_path']:
            cnt = 0
            this_dim = [0]
            for cam_id in self.cam_names:
                num_person = len(self.info_dict[cam_id][img_id])
                cnt += num_person
                this_dim.append(cnt)
            self.dimGroup[int(img_id)] = torch.Tensor(this_dim).long()

        self.K = camera_parameter['K'].astype(np.float32)
        self.RT = camera_parameter['RT'].astype(np.float32)
        self.P = self.K @ self.RT
        self.skew_op = lambda x: torch.tensor([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])
        self.fundamental_op = lambda K_0, R_0, T_0, K_1, R_1, T_1: torch.inverse(K_0).t() @ (
                R_0 @ R_1.t()) @ K_1.t() @ self.skew_op(K_1 @ R_1 @ R_0.t() @ (T_0 - R_0 @ R_1.t() @ T_1))
        self.fundamental_RT_op = lambda K_0, RT_0, K_1, RT_1: self.fundamental_op(K_0, RT_0[:, :3], RT_0[:, 3], K_1,
                                                                                  RT_1[:, :3], RT_1[:, 3])
        self.F = torch.zeros(len(self.cam_names), len(self.cam_names), 3, 3)  # NxNx3x3 matrix

        for i in range(len(self.cam_names)):
            for j in range(len(self.cam_names)):
                self.F[i, j] += self.fundamental_RT_op(torch.tensor(self.K[i]),
                                                       torch.tensor(self.RT[i]),
                                                       torch.tensor(self.K[j]), torch.tensor(self.RT[j]))
                if self.F[i, j].sum() == 0:
                    self.F[i, j] += 1e-12  # to avoid nan

    def __getitem__(self, item):
        """
        Get a list of image in multi view at the same time
        :param item:
        :return: images, fnames, pid, cam_id
        """
        img_id = item
        data_by_cam = OrderedDict()
        for cam_id in self.cam_names:
            data_by_cam[cam_id] = [v['cropped_img'] for v in self.info_dict[cam_id][img_id]]
        image = list()
        fname = list()
        pid = list()
        cam_id = list()
        for k, v in data_by_cam.items():
            for i, _ in enumerate(v):
                fname += [f'{k}_{i}']
            # fname += [f'{k}_{i}' for i, _ in enumerate(v)]
            pid += list(range(len(v)))
            cam_id += [k for i in v]
            image += [self.test_transformer(Image.fromarray(np.uint8(i))) for i in v]
        image = torch.stack(image)
        data_batch = (image, fname, pid, cam_id)
        return data_batch

    def __len__(self):
        if len(self.info_dict):
            return len(self.info_dict[self.cam_names[0]])
        else:
            return 0

    def get_unary(self, person, sub_imgid2cam, candidates, img_id):
        def get2Dfrom3D(x, P):
            """get the 2d joint from 3d joint"""
            x4d = np.append(x, 1)
            x2d = np.dot(P, x4d)[0:2] / (np.dot(P, x4d)[2] + 10e-6)  # to avoid np.dot(P, x4d)[2] = 0

            return x2d

        # get the unary of 3D candidates
        joint_num = len(candidates)
        point_num = len(candidates[0])
        unary = np.ones((joint_num, point_num))
        info_list = list()  # This also occur in multi setimator
        for cam_id in self.cam_names:
            info_list += self.info_dict[cam_id][img_id]
        # project the 3d point to each view to get the 2d points

        for pid in person:
            Pi = self.P[sub_imgid2cam[pid]]
            heatmap = info_list[pid]['heatmap_data']
            crop = np.array(info_list[pid]['heatmap_bbox'])
            points_3d = candidates.reshape(-1, 3).T
            points_3d_homo = np.vstack([points_3d, np.ones(points_3d.shape[-1]).reshape(1, -1)])
            points_2d_homo = (Pi @ points_3d_homo).T.reshape(17, -1, 3)
            points_2d = points_2d_homo[..., :2] / (points_2d_homo[..., 2].reshape(17, -1, 1) + 10e-6)
            for joint, heatmap_j in enumerate(heatmap):
                for k_pose, point3d in enumerate(candidates[joint]):
                    point_2d = points_2d[joint, k_pose]
                    point_2d_in_heatmap = point_2d - np.array([crop[0], crop[1]])
                    # point_2d_in_heatmap = (point_2d - np.array ( [crop[0], crop[1]] )) / np.array (
                    #     [crop[2] - crop[0], crop[3] - crop[1]] )
                    if point_2d_in_heatmap[0] > heatmap_j.shape[1] or point_2d_in_heatmap[0] < 0 or point_2d_in_heatmap[
                        1] > heatmap_j.shape[0] or point_2d_in_heatmap[1] < 0:
                        unary_i = 10e-6
                    else:
                        unary_i = heatmap_j[int(point_2d_in_heatmap[1]), int(point_2d_in_heatmap[0])]
                    unary[joint, k_pose] = unary[joint, k_pose] * unary_i

        unary = np.log10(unary)
        return unary

