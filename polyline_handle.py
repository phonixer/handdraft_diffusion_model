import pickle
import numpy as np
import os
import sys
from utlits_map import get_interested_agents

import matplotlib.pyplot as plt

# 读文件

data_file = '/home/zrg/Code/MTR/data/waymo/processed_scenarios_training/sample_1a37753b91c30e16.pkl'

#读取数据
with open(data_file, 'rb') as f:
    data = pickle.load(f)

print(data.keys())

# dict_keys(['track_infos', 'dynamic_map_infos', 'map_infos', 'scenario_id', 'timestamps_seconds',
#             'current_time_index', 'sdc_track_index', 'objects_of_interest', 'tracks_to_predict'])

# 查看数据，输出数据类型，如果是dict输出key
for key in data.keys():
    print(key, type(data[key]))
    if isinstance(data[key], dict):
        print(data[key].keys())

# 输出结果
# track_infos <class 'dict'>
# dict_keys(['object_id', 'object_type', 'trajs'])
# dynamic_map_infos <class 'dict'>
# dict_keys(['lane_id', 'state', 'stop_point'])
# map_infos <class 'dict'>
# dict_keys(['lane', 'road_line', 'road_edge', 'stop_sign', 'crosswalk', 'speed_bump', 'all_polylines'])
# scenario_id <class 'str'>
# timestamps_seconds <class 'list'>
# current_time_index <class 'int'>
# sdc_track_index <class 'int'>
# objects_of_interest <class 'list'>
# tracks_to_predict <class 'dict'>
# dict_keys(['track_index', 'difficulty', 'object_type'])

# 对track_infos进行查看
track_infos = data['track_infos']
for key in track_infos.keys():
    print(key, type(track_infos[key]))
    if isinstance(track_infos[key], dict):
        print(track_infos[key].keys())

# 输出结果
# dict_keys(['track_index', 'difficulty', 'object_type'])
# object_id <class 'list'>
# object_type <class 'list'>
# trajs <class 'numpy.ndarray'>

for key in track_infos.keys():
    print(key, len(track_infos[key]))






mode = 'train'
infos_path = '/home/zrg/Code/MTR/data/waymo/processed_scenarios_training_infos.pkl'

with open(infos_path, 'rb') as f:
    src_infos = pickle.load(f)
    infos = src_infos

index = 0
info = infos[index]
scene_id = info['scenario_id']
print(scene_id)
print(info.keys())
#dict_keys(['scenario_id', 'timestamps_seconds',
#'current_time_index', 'sdc_track_index', 'objects_of_interest', 'tracks_to_predict'])
info_path = f'/home/zrg/Code/MTR/data/waymo/processed_scenarios_training/sample_{scene_id}.pkl'
print(f'loading {info_path}')
with open(f'/home/zrg/Code/MTR/data/waymo/processed_scenarios_training/sample_{scene_id}.pkl', 'rb') as f:
    info = pickle.load(f)

sdc_track_index = info['sdc_track_index']
current_time_index = info['current_time_index']
timestamps = np.array(info['timestamps_seconds'][:current_time_index + 1], dtype=np.float32)


track_infos = info['track_infos']

track_index_to_predict = np.array(info['tracks_to_predict']['track_index'])
obj_types = np.array(track_infos['object_type'])
obj_ids = np.array(track_infos['object_id'])
obj_trajs_full = track_infos['trajs']  # (num_objects, num_timestamp, 10)
obj_trajs_past = obj_trajs_full[:, :current_time_index + 1]
obj_trajs_future = obj_trajs_full[:, current_time_index + 1:]

print('--'*20)
print(sdc_track_index)          # 99
print(current_time_index)       # 10
print(timestamps)
print(track_index_to_predict)   #  [ 6 26 39 22  0  7  4 20]
print(obj_types)                # 'TYPE_VEHICLE' 'TYPE_CYCLIST' 'TYPE_PEDESTRIAN'
print(obj_ids)                  # 121 122 131 132 133
print(obj_trajs_full.shape)  #(100, 91, 10)
print(obj_trajs_past.shape)  #(100, 11, 10)
print(obj_trajs_future.shape) # (100, 80, 10)
# print(obj_trajs_past[0])
# print(obj_trajs_future[0])


center_objects, track_index_to_predict = get_interested_agents(
    track_index_to_predict=track_index_to_predict,
    obj_trajs_full=obj_trajs_full,
    current_time_index=current_time_index,
    obj_types=obj_types, scene_id=scene_id
)

print(center_objects.shape)  # (8, 10) [cx, cy, cz, dx, dy, dz, heading, vel_x, vel_y, valid]
print(center_objects)

print(len(info['map_infos']['all_polylines']))  # 22240


# map_polylines_data, map_polylines_mask, map_polylines_center =    (
#     center_objects=center_objects, 
#     map_infos=info['map_infos'],
#     center_offset= [30.0, 0]
# )  # (num_center_objects, num_topk_polylines, num_points_each_polyline, 9), (num_center_objects, num_topk_polylines, num_points_each_polyline)
# ret_dict['map_polylines'] = map_polylines_data
# ret_dict['map_polylines_mask'] = (map_polylines_mask > 0)
# ret_dict['map_polylines_center'] = map_polylines_center

num_center_objects = center_objects.shape[0]

print(num_center_objects)  # 8

# plt.plot(center_objects[:, 0], center_objects[:, 1], 'ro')
plt.savefig('center_objects.png')
plt.show()

# obj_trajs_past?
# plt.figure()
for i in range(1, num_center_objects):
    plt.plot(obj_trajs_past[track_index_to_predict[i], :, 0], obj_trajs_past[track_index_to_predict[i], :, 1], 'b', label='Past' if i == 1 else "")
    plt.plot(obj_trajs_future[track_index_to_predict[i], :, 0], obj_trajs_future[track_index_to_predict[i], :, 1], 'r', label='Future' if i == 1 else "")
    plt.plot(center_objects[i, 0], center_objects[i, 1], 'ro', label='Center' if i == 1 else "")

plt.legend()
plt.savefig('obj_trajs_past.png')


## 算map_polylines_data

polylines = np.array(info['map_infos']['all_polylines'])
point_dim = polylines.shape[-1]
print('polylines:', polylines.shape)  # (22240, 7)
print('point_dim:', point_dim)  # 7 

# 超参数
point_sampled_interval = 1     # 采样间隔
vector_break_dist_thresh = 1.0 # 向量断裂距离阈值
num_points_each_polyline = 20  # 每个polyline的点数


# all_polylines (num_points, 7): [x, y, z, dir_x, dir_y, dir_z, global_type]

sampled_points = polylines[::point_sampled_interval]           # 采样点
sampled_points_shift = np.roll(sampled_points, shift=1, axis=0) # 采样点的shift，np.roll是循环移位, axis=0表示按行移位
buffer_points = np.concatenate((sampled_points[:, 0:2], sampled_points_shift[:, 0:2]), axis=-1) # [ed_x, ed_y, st_x, st_y]
buffer_points[0, 2:4] = buffer_points[0, 0:2]
print('sampled_points:', sampled_points.shape)  # (22240, 7)
print('sampled_points_shift:', sampled_points_shift.shape)  # (22240, 7)
print('buffer_points:', buffer_points.shape)  # (22240, 4)
# all_polylines (num_points, 7): [x, y, z, dir_x, dir_y, dir_z, global_type]
break_idxs = (np.linalg.norm(buffer_points[:, 0:2] - buffer_points[:, 2:4], axis=-1) > vector_break_dist_thresh).nonzero()[0]
polyline_list = np.array_split(sampled_points, break_idxs, axis=0)

print('break_idxs:', break_idxs.shape)  # (22240,)
print('polyline_list:', len(polyline_list))  # 22240


# # 给个demo 说明 这几行代码的作用
# sampled_points = np.array([[1, 2, 3, 4, 5, 6, 7], [2, 3, 4, 5, 6, 7, 8], [3, 4, 5, 6, 7, 8, 9], [4, 5, 6, 7, 8, 9, 10]])
# sampled_points_shift = np.roll(sampled_points, shift=1, axis=0)
# buffer_points = np.concatenate((sampled_points[:, 0:2], sampled_points_shift[:, 0:2]), axis=-1)
# buffer_points[0, 2:4] = buffer_points[0, 0:2]
# print('sampled_points:', sampled_points.shape)
# print(sampled_points)
# print(sampled_points_shift)
# print(buffer_points)

ret_polylines = []
ret_polylines_mask = []

def append_single_polyline(new_polyline):
    cur_polyline = np.zeros((num_points_each_polyline, point_dim), dtype=np.float32)
    cur_valid_mask = np.zeros((num_points_each_polyline), dtype=np.int32)
    cur_polyline[:len(new_polyline)] = new_polyline
    cur_valid_mask[:len(new_polyline)] = 1
    ret_polylines.append(cur_polyline)
    ret_polylines_mask.append(cur_valid_mask)


for k in range(len(polyline_list)):
    if polyline_list[k].__len__() <= 0:
        continue
    for idx in range(0, len(polyline_list[k]), num_points_each_polyline):
        append_single_polyline(polyline_list[k][idx: idx + num_points_each_polyline])

ret_polylines = np.stack(ret_polylines, axis=0)
ret_polylines_mask = np.stack(ret_polylines_mask, axis=0)

print('ret_polylines:', ret_polylines.shape)  #(1298, 20, 7)
print('ret_polylines_mask:', ret_polylines_mask.shape)  # (1298, 20)

# ret_polylines = torch.from_numpy(ret_polylines)
# ret_polylines_mask = torch.from_numpy(ret_polylines_mask)

# def generate_batch_polylines_from_map(polylines, point_sampled_interval=1, vector_break_dist_thresh=1.0, num_points_each_polyline=20):

#     ret_polylines = torch.from_numpy(ret_polylines)
#     ret_polylines_mask = torch.from_numpy(ret_polylines_mask)

#     return ret_polylines, ret_polylines_mask

# batch_polylines, batch_polylines_mask = self.generate_batch_polylines_from_map(
#     polylines=polylines.numpy(), point_sampled_interval=self.dataset_cfg.get('POINT_SAMPLED_INTERVAL', 1),
#     vector_break_dist_thresh=self.dataset_cfg.get('VECTOR_BREAK_DIST_THRESH', 1.0),
#     num_points_each_polyline=self.dataset_cfg.get('NUM_POINTS_EACH_POLYLINE', 20),
# )  # (num_polylines, num_points_each_polyline, 7), (num_polylines, num_points_each_polyline)

# polyline_center = batch_polylines[:, :, 0:2].sum(dim=1) / torch.clamp_min(batch_polylines_mask.sum(dim=1).float()[:, None], min=1.0)
# center_offset_rot = torch.from_numpy(np.array(center_offset, dtype=np.float32))[None, :].repeat(num_center_objects, 1)
# center_offset_rot = common_utils.rotate_points_along_z(
#     points=center_offset_rot.view(num_center_objects, 1, 2),
#     angle=center_objects[:, 6]
# ).view(num_center_objects, 2)

# pos_of_map_centers = center_objects[:, 0:2] + center_offset_rot

# dist = (pos_of_map_centers[:, None, :] - polyline_center[None, :, :]).norm(dim=-1)  # (num_center_objects, num_polylines)
# topk_dist, topk_idxs = dist.topk(k=num_of_src_polylines, dim=-1, largest=False)
# map_polylines = batch_polylines[topk_idxs]  # (num_center_objects, num_topk_polylines, num_points_each_polyline, 7)
# map_polylines_mask = batch_polylines_mask[topk_idxs]  # (num_center_objects, num_topk_polylines, num_points_each_polyline)