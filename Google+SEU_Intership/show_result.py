import os
import numpy as np
from scipy.optimize import linear_sum_assignment
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.pyplot as plt

def load_npz_files(folder_path):
    npz_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.npz')])
    data_list = []

    for npz_file in npz_files:
        data = np.load(os.path.join(folder_path, npz_file))['data']  # 修改为实际的键名
        # print(data.shape)
        data_list.append(data)

    return data_list

def plot_pose3d(pose, person_ids, frame):

    _CONNECTION = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                   [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]

    def joint_color(person_idx):

        colors = [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (255, 255, 0),
            (255, 0, 255),
            (0, 255, 255)
        ]
        return colors[person_idx % len(colors)]

    assert (pose.ndim == 3)
    assert (pose.shape[1] == 3)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # Using add_subplot to get 3D axis

    for person_idx in range(pose.shape[0]):  # Loop over each person
        color_idx = person_ids[person_idx]
        for c in _CONNECTION:
            col = '#%02x%02x%02x' % joint_color(color_idx)
            ax.plot([pose[person_idx, 0, c[0]], pose[person_idx, 0, c[1]]],
                    [pose[person_idx, 1, c[0]], pose[person_idx, 1, c[1]]],
                    [pose[person_idx, 2, c[0]], pose[person_idx, 2, c[1]]], c=col)

        for j in range(pose.shape[2]):
            col = '#%02x%02x%02x' % joint_color(color_idx)
            ax.scatter(pose[person_idx, 0, j], pose[person_idx, 1, j], pose[person_idx, 2, j],
                       c=col, marker='o', edgecolor=col)

    x_range = (-3, 2)  # Set x-axis range
    y_range = (-3, 3)  # Set y-axis range
    z_range = (0, 2)  # Set z-axis range
    ax.set_xlim3d(x_range[0], x_range[1])
    ax.set_ylim3d(y_range[0], y_range[1])
    ax.set_zlim3d(z_range[0], z_range[1])
    plt.savefig(f"./data_test/frame_{frame:04d}.png")
    # plt.show()
    return fig

class PoseMatcher:
    def __init__(self, distance_threshold=100):
        self.prev_pose = None
        self.person_ids = []
        self.distance_threshold = distance_threshold  # 距离阈值
        self.connection=[[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8], [7, 9],
                   [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]]

    def compute_joint_distances(self, pose, connection):
        total_distance = 0  # 初始化总距离
        for conn in connection:
            i, j = conn
            # 计算关节 i 和 j 之间的欧氏距离
            dist = np.linalg.norm(pose[:, i] - pose[:, j])
            total_distance += dist  # 将每个距离累加到总距离
        return total_distance  # 返回总距离

    def filter_poses_by_distance(self, current_pose, connection, threshold=6):
        filtered_poses = []
        for i in range(current_pose.shape[0]):
            distances = self.compute_joint_distances(current_pose[i], connection)
            # print('distances:', distances)
            if threshold/2 <= distances <= threshold:
                filtered_poses.append(current_pose[i])  # 只有当总距离小于或等于阈值时，保留该点
            # else:
            #     print(f'Pose {i} discarded due to distance {distances} > {threshold}')
        return np.array(filtered_poses)

    def calculate_distance_matrix(self, pose1, pose2):

        num_persons_prev = pose1.shape[0]
        num_persons_curr = pose2.shape[0]
        dist_matrix = np.zeros((num_persons_prev, num_persons_curr))

        for i in range(num_persons_prev):
            for j in range(num_persons_curr):
                dist_matrix[i, j] = np.sum(np.linalg.norm(pose1[i] - pose2[j], axis=0) ** 2)

        return dist_matrix

    def match_poses(self, current_pose):

        current_pose=self.filter_poses_by_distance(current_pose,self.connection)
        if self.prev_pose is None:
            # 如果没有前一帧数据，直接返回当前帧，并分配初始ID
            self.prev_pose = current_pose
            self.person_ids = list(range(current_pose.shape[0]))
            print(len(self.person_ids))
            return current_pose, self.person_ids

        # 计算前一帧和当前帧的距离矩阵
        dist_matrix = self.calculate_distance_matrix(self.prev_pose, current_pose)

        # 使用距离阈值进行过滤，超过阈值的距离设为无穷大
        # dist_matrix[dist_matrix > self.distance_threshold] = 99999

        # 使用匈牙利算法找到最优匹配
        row_ind, col_ind = linear_sum_assignment(dist_matrix)

        # 根据匹配结果重新排序当前帧的pose和person_ids
        matched_pose = current_pose[col_ind]
        #新出现的下标
        all_indices = set(range(current_pose.shape[0]))
        matched_indices = set(col_ind)
        unmatched_indices = list(all_indices - matched_indices)
        # print('col_ind',col_ind)
        # print('unmatched_indices',unmatched_indices)

        # 更新前一帧的数据
        if current_pose.shape[0] > self.prev_pose.shape[0]:
            self.person_ids = list(range(current_pose.shape[0]))
            self.prev_pose=np.zeros_like(current_pose)
            self.prev_pose[:matched_pose.shape[0],:,:]=matched_pose
            self.prev_pose[matched_pose.shape[0]:, :, :]=current_pose[unmatched_indices]
        elif current_pose.shape[0] < self.prev_pose.shape[0]:
            self.prev_pose[:matched_pose.shape[0],:,:]=matched_pose
        else:
            self.prev_pose = matched_pose

        print(matched_pose.shape)
        return matched_pose, self.person_ids

folder_path = './data'  # 替换为实际的文件夹路径

data_list = load_npz_files(folder_path)
data_list = np.asarray(data_list)
pose_tracker = PoseMatcher()

for i in range(1,data_list.shape[0]):
    matched_frame, person_ids = pose_tracker.match_poses(data_list[i])
    plot_pose3d(matched_frame, person_ids, i)

