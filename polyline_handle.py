import pickle
import numpy as np

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
print(sdc_track_index)
print(current_time_index)
print(timestamps)
print(track_index_to_predict)
print(obj_types)
print(obj_ids)
print(obj_trajs_full.shape)
print(obj_trajs_past.shape)
print(obj_trajs_future.shape)
# print(obj_trajs_past[0])
# print(obj_trajs_future[0])
