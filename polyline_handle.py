import pickle
import numpy as np
import os
import sys
import torch
from utlits_map import get_interested_agents

import matplotlib.pyplot as plt
from waymo_types import object_type, lane_type, road_line_type, road_edge_type, signal_state, polyline_type

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


print('info[\'map_infos\']:',info['map_infos'].keys())
# info['map_infos']: dict_keys(['lane', 
# 'road_line', 'road_edge', 'stop_sign', 
# 'crosswalk', 'speed_bump', 'all_polylines'])

print('info[\'map_infos\'][\'lane\']:',len(info['map_infos']['lane']))
print('info[\'map_infos\'][\'road_line\']:',len(info['map_infos']['road_line']))
print('info[\'map_infos\'][\'road_edge\']:',len(info['map_infos']['road_edge']))
print('info[\'map_infos\'][\'stop_sign\']:',len(info['map_infos']['stop_sign']))
print('info[\'map_infos\'][\'crosswalk\']:',len(info['map_infos']['crosswalk']))
print('info[\'map_infos\'][\'speed_bump\']:',len(info['map_infos']['speed_bump']))
print('info[\'map_infos\'][\'all_polylines\']:',len(info['map_infos']['all_polylines']))





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

ret_polylines = torch.from_numpy(ret_polylines)
ret_polylines_mask = torch.from_numpy(ret_polylines_mask)

batch_polylines = ret_polylines
batch_polylines_mask = ret_polylines_mask

center_offset= [30.0, 0]


center_objects = torch.from_numpy(center_objects)
num_of_src_polylines = 50
polyline_center = batch_polylines[:, :, 0:2].sum(dim=1) / torch.clamp_min(batch_polylines_mask.sum(dim=1).float()[:, None], min=1.0)
center_offset_rot = torch.from_numpy(np.array(center_offset, dtype=np.float32))[None, :].repeat(num_center_objects, 1)
# center_offset_rot = common_utils.rotate_points_along_z(
#     points=center_offset_rot.view(num_center_objects, 1, 2),
#     angle=center_objects[:, 6]
# ).view(num_center_objects, 2)
center_offset_rot = center_offset_rot.view(num_center_objects, 2)

# pos_of_map_centers = center_objects[:, 0:2] + center_offset_rot
pos_of_map_centers = center_objects[:, 0:2]


dist = (pos_of_map_centers[:, None, :] - polyline_center[None, :, :]).norm(dim=-1)  # (num_center_objects, num_polylines)
topk_dist, topk_idxs = dist.topk(k=num_of_src_polylines, dim=-1, largest=False)
map_polylines = batch_polylines[topk_idxs]  # (num_center_objects, num_topk_polylines, num_points_each_polyline, 7)
map_polylines_mask = batch_polylines_mask[topk_idxs]  # (num_center_objects, num_topk_polylines, num_points_each_polyline)


print('center_offset_rot:', center_offset_rot.shape)
print('pos_of_map_centers:', pos_of_map_centers.shape)
print('dist:', dist.shape)
print('topk_dist:', topk_dist.shape)
print('topk_idxs:', topk_idxs.shape)
print('map_polylines:', map_polylines.shape)
print('map_polylines_mask:', map_polylines_mask.shape)
#终止程序
# sys.exit(0)


print(map_polylines[:,:,:,6])

# 统计map_polylines[:,:,:,6]的值，有几种
print(np.unique(map_polylines[:,:,:,6].numpy()))


# 获取唯一值
unique_values = np.unique(map_polylines[:, :, :, 6].numpy())
colors = plt.get_cmap('hsv', len(unique_values))

# 创建一个字典，将唯一值映射到颜色
value_to_color = {value: colors(i) for i, value in enumerate(unique_values)}

# 创建一个字典，将唯一值映射到图例标签
value_to_label = {value: f'Value {value}' for value in unique_values}

for i in range(1, num_center_objects):
    plt.plot(center_objects[i, 0], center_objects[i, 1], 'ro', label='Center' if i == 1 else "")
    
    for j in range(num_of_src_polylines):
        # 排除掉（0，0）
        valid_points = (map_polylines[i, j, :, 0] != 0) | (map_polylines[i, j, :, 1] != 0)
        valid_values = map_polylines[i, j, valid_points, -1].numpy()
        
        for value in np.unique(valid_values):
            color = value_to_color[value]  # 获取对应的颜色
            points = map_polylines[i, j, valid_points, :][map_polylines[i, j, valid_points, -1].numpy() == value]
            plt.plot(points[:, 0], points[:, 1], color=color)


# 添加图例
for value in unique_values:
    color = value_to_color[value]
    plt.plot([], [], color = color, label=f'Value {value}')

plt.legend()

# 保存并显示图像
plt.savefig('map_polylines.png')
plt.show()




# obj_trajs_past
# obj_trajs_future
# # 输出形状
print(obj_trajs_past.shape)  # (100, 11, 10)
print(obj_trajs_future.shape)  # (100, 80, 10)

from diffusion_model import Diffusion

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = "cpu"  # cuda
batchsize = 200
act_dim = 20
obs_dim = 22
T = 20
epoch = 20000
lr = 1e-4
# x = torch.randn(256, 2).to(device)  # Batch, action_dim
# state = torch.randn(256, 11).to(device)  # Batch, state_dim

# 生成 x 张量，并转换为浮点类型
x = torch.tensor(obj_trajs_future[track_index_to_predict,::8,:2]).to(device)
# 生成 state 张量，每个 batch 的值都相同，并转换为浮点类型
state = torch.tensor(obj_trajs_past[track_index_to_predict,:,:2]).to(device)
x[0,:,:] = x[3,:,:]
state[0,:,:] = state[3,:,:]

# 将均值变成0 然后归一化
# x 的每个点都减去第一个点
x = (x - x[:, 0:1, :]) / 50 - 0.25
x[1:3,:,1] = x[1:3,:,1] - 0.1
x[2:3,:,0] = x[2:3,:,0] + 0.15

state = (state - state[:, 0:1, :]) / 50 -0.25
# 归一化到均值为0
# x = (x - x.mean(dim=1, keepdim=True)) / 50
# state = (state - state.mean(dim=1, keepdim=True)) / 50




# # 复制 x 和 state，使其形状为 (100, 10, 2)
x = x.repeat(25, 1, 1)
state = state.repeat(25, 1, 1)

# Reshape x and state to (100,)
x = x.view(batchsize, -1)
state = state.view(batchsize, -1)
print('---------')
print(x.shape)
print(state.shape)








model = Diffusion(loss_type='l2', obs_dim=obs_dim, act_dim=act_dim, hidden_dim=800, device=device, T=T)
result = model(state)  # Sample result

loss = model.loss(x, state)

# print(f"action: {result};loss: {loss.item()}")
import matplotlib.pyplot as plt
import torch.optim as optim


# T = 200

# Check if training is required
train_model = True
model_path = 'polylinediffusion_model.pth'

if os.path.exists(model_path):
    print(f"Loading model from {model_path}")
    model.load_state_dict(torch.load(model_path))
    train_model = False
else:
    print("Model not found, training a new model")
# 训练模型
from torch.optim.lr_scheduler import LambdaLR
if train_model:
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # Define the learning rate scheduler
    model.train()
    for i in range(epoch):
        loss = model.loss(x, state)
        loss.backward()
        if i % 100 == 0:
            print(f'epoch: {i}, loss: {loss.item()}')
        optimizer.step()
        optimizer.zero_grad()

    
    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

# Evaluate the model
model.eval()
result, diffusion_steps = model(state)
print('result', result.shape)




# 训练结束后绘制扩散过程的图像
state_test = state
x_test = x
print(state_test)
print(state.shape)
print(state_test.shape)
print(x_test)
print(x_test.shape)

action, diffusion_steps = model.sample(state_test)

# 算下loss
loss = model.loss(x_test, state_test)
print(f"action: {action};loss: {loss.item()}")
# 输出真值
print(x_test)
print(len(diffusion_steps))

# 恢复形状
x_test = x_test.view(batchsize, -1, 2)
print(x_test.shape)
action = action.view(batchsize, -1, 2)
print(action.shape)
state_test = state_test.view(batchsize, -1, 2)
# Calculate endpoint error
endpoints_gt = x_test[:, -1, :]  # Ground truth endpoints
endpoints_pred = action[:, -1, :]  # Predicted endpoints

# 放到cpu上
endpoints_gt = endpoints_gt.cpu().detach().numpy()
endpoints_pred = endpoints_pred.cpu().detach().numpy()
x_test = x_test.cpu().detach().numpy()
action = action.cpu().detach().numpy()  

endpoint_error = np.linalg.norm(endpoints_pred - endpoints_gt, axis=-1)
total_endpoint_error = endpoint_error.sum()
average_endpoint_error = total_endpoint_error / np.prod(endpoints_gt.shape[:-1])
percentage_endpoint_error = (total_endpoint_error / np.linalg.norm(endpoints_gt, axis=-1).sum()) * 100


print(f"Total Endpoint Error: {total_endpoint_error}")
print(f"Average Endpoint Error: {average_endpoint_error}")
print(f"Percentage Endpoint Error: {percentage_endpoint_error}%")
# 计算平均所有点误差
point_error = np.linalg.norm(action - x_test, axis=-1)
total_point_error = point_error.sum()
average_point_error = total_point_error / np.prod(x_test.shape[:-1])
percentage_point_error = (total_point_error / np.linalg.norm(x_test, axis=-1).sum()) * 100


print(f"Total Point Error: {total_point_error}")
print(f"Average Point Error: {average_point_error}")
print(f"Percentage Point Error: {percentage_point_error}%")


# Plotting the results
plt.figure(figsize=(15, 5))
num_plots_per_row = 5
polt_item = [0,3,4,5,6]
num_rows = (len(polt_item) + num_plots_per_row - 1) // num_plots_per_row

fig, axes = plt.subplots(num_rows, num_plots_per_row, figsize=(20, 5 * num_rows))
axes = axes.flatten()
for i in range(len(polt_item)):
    ax = axes[i]

    valid_points = (abs(action[polt_item[i],  :, 0]) < 2) & (abs(action[polt_item[i],  :, 1]) < 2)
    points_action = action[polt_item[i],  valid_points, :]
    points_x_test = x_test[polt_item[i],  valid_points, :]
    ax.plot(points_action[:, 0], points_action[:, 1], 'r', label='Predicted' if i == 0 else "")
    ax.plot(points_x_test[:, 0], points_x_test[:, 1], 'g', label='Ground Truth' if i == 0 else "")
    ax.set_title(f'Center Object {i}')
    ax.legend()

# plt.title('Predicted vs Ground Truth Polylines')

plt.tight_layout()
plt.savefig('predicted_vs_ground_truth_polylines.png')
plt.show()





# Plotting the results
plt.figure(figsize=(15, 5))
num_plots_per_row = 5
polt_item = [0,3,4,5,6]
num_rows = (len(polt_item) + num_plots_per_row - 1) // num_plots_per_row

fig, axes = plt.subplots(num_rows, num_plots_per_row, figsize=(20, 5 * num_rows))
axes = axes.flatten()
for i in range(len(polt_item)):
    ax = axes[i]

    valid_points = (abs(action[polt_item[i],  :, 0]) < 2) & (abs(action[polt_item[i],  :, 1]) < 2)
    points_action = action[polt_item[i],  valid_points, :]
    points_x_test = x_test[polt_item[i],  valid_points, :]
    ax.plot(points_action[:, 0], points_action[:, 1], 'r', label='Predicted' if i == 0 else "")
    ax.plot(points_x_test[:, 0], points_x_test[:, 1], 'g', label='Ground Truth' if i == 0 else "")
    ax.set_title(f'Center Object {i}')
    ax.legend()

# plt.title('Predicted vs Ground Truth Polylines')

plt.tight_layout()
plt.savefig('predicted_vs_ground_truth_polylines.png')
plt.show()

# Plotting endpoint errors
plt.figure(figsize=(15, 5))
plt.plot(endpoint_error.flatten(), 'bo-', label='Endpoint Error')
plt.title('Endpoint Error for Each Polyline')
plt.xlabel('Polyline Index')
plt.ylabel('Error')
plt.legend()
plt.tight_layout()
plt.savefig('endpoint_error.png')
plt.show()



# 画降噪过程
# Plotting the results
plt.figure(figsize=(15, 5))
num_plots_per_row = 5
polt_item = [0,3,4,5,6]
num_rows = (len(polt_item) + num_plots_per_row - 1) // num_plots_per_row

fig, axes = plt.subplots(num_rows, num_plots_per_row, figsize=(20, 5 * num_rows))
axes = axes.flatten()

num_steps = len(diffusion_steps)
steps_to_plot = [int(i * num_steps / 10) for i in range(10)] + [num_steps - 1]

for i in range(len(polt_item)):
    ax = axes[i]
    for j in range(num_steps):
        step = diffusion_steps[j].view(batchsize, -1, 2).cpu().detach().numpy()
        valid_points = (abs(step[polt_item[i], :, 0]) < 2) & (abs(step[polt_item[i], :, 1]) < 2)
        points = step[polt_item[i], valid_points, :]
        # 画点
        ax.plot(points[:, 0], points[:, 1], 'ro', alpha=j / num_steps)

    valid_points = (abs(action[polt_item[i],  :, 0]) < 2) & (abs(action[polt_item[i],  :, 1]) < 2)
    points_action = action[polt_item[i],  valid_points, :]
    points_x_test = x_test[polt_item[i],  valid_points, :]
    ax.plot(points_action[:, 0], points_action[:, 1], 'r', label='Predicted' if i == 0 else "")
    ax.plot(points_x_test[:, 0], points_x_test[:, 1], 'g', label='Ground Truth' if i == 0 else "")
    ax.set_title(f'Center Object {i}')
    ax.legend()

# plt.title('Predicted vs Ground Truth Polylines')

plt.tight_layout()
plt.savefig('Diffusion5polylines.png')
plt.show()

# Plotting endpoint errors
plt.figure(figsize=(15, 5))
plt.plot(endpoint_error.flatten(), 'bo-', label='Endpoint Error')
plt.title('Endpoint Error for Each Polyline')
plt.xlabel('Polyline Index')
plt.ylabel('Error')
plt.legend()
plt.tight_layout()
plt.savefig('endpoint_error.png')
plt.show()


polt_item = [0,3,4,5,6]
# 计算polt_item的各种误差
# 放到cpu上


endpoints_gt = endpoints_gt[polt_item]
endpoints_pred = endpoints_pred[polt_item]
x_test = x_test[polt_item]
action = action[polt_item]

endpoint_error = np.linalg.norm(endpoints_pred - endpoints_gt, axis=-1)
total_endpoint_error = endpoint_error.sum()
average_endpoint_error = total_endpoint_error / np.prod(endpoints_gt.shape[:-1])
percentage_endpoint_error = (total_endpoint_error / np.linalg.norm(endpoints_gt, axis=-1).sum()) * 100


print(f"Total Endpoint Error: {total_endpoint_error}")
print(f"Average Endpoint Error: {average_endpoint_error}")
print(f"Percentage Endpoint Error: {percentage_endpoint_error}%")
# 计算平均所有点误差
point_error = np.linalg.norm(action - x_test, axis=-1)
total_point_error = point_error.sum()
average_point_error = total_point_error / np.prod(x_test.shape[:-1])
percentage_point_error = (total_point_error / np.linalg.norm(x_test, axis=-1).sum()) * 100


print(f"Total Point Error: {total_point_error}")
print(f"Average Point Error: {average_point_error}")
print(f"Percentage Point Error: {percentage_point_error}%")



num_plots_per_row = 11
num_rows = (len(steps_to_plot) + num_plots_per_row - 1) // num_plots_per_row

fig, axes = plt.subplots(num_rows, num_plots_per_row, figsize=(40, 5 * num_rows))
axes = axes.flatten()

for i in range(len(steps_to_plot)):

    step_idx = steps_to_plot[i]


    ax = axes[i]


    j = 0

    step = diffusion_steps[step_idx].view(batchsize, -1, 2).cpu().detach().numpy()
    valid_points = (abs(action[j,  :, 0]) < 2) & (abs(action[j,  :, 1]) < 2)
    points_action = action[j,  valid_points, :]
    points_x_test = x_test[j,  valid_points, :]
    ax.plot(points_action[:, 0], points_action[:, 1], 'ro', label='Predicted' if i == 0 else "")
    ax.plot(points_x_test[:, 0], points_x_test[:, 1], 'g', label='Ground Truth' if i == 0 else "")

    ax.scatter(step[j, :, 0], step[j, :, 1], label=f'Step {step_idx}')
    ax.set_title(f'Step {step_idx}')
    ax.legend()




plt.tight_layout()
plt.savefig('1mapDIffusionprocesscenter_objects_comparison.png')
plt.show()


steps_to_plot = range(15, 20)
num_plots_per_row = 5
num_rows = (len(steps_to_plot) + num_plots_per_row - 1) // num_plots_per_row

fig, axes = plt.subplots(num_rows, num_plots_per_row, figsize=(40, 5 * num_rows))
axes = axes.flatten()

for i in range(len(steps_to_plot)):

    step_idx = steps_to_plot[i]


    ax = axes[i]


    j = 0

    step = diffusion_steps[step_idx].view(batchsize, -1, 2).cpu().detach().numpy()
    valid_points = (abs(action[j,  :, 0]) < 2) & (abs(action[j,  :, 1]) < 2)
    points_action = action[j,  valid_points, :]
    points_x_test = x_test[j,  valid_points, :]
    ax.plot(points_action[:, 0], points_action[:, 1], 'ro', label='Predicted' if i == 0 else "")
    ax.plot(points_x_test[:, 0], points_x_test[:, 1], 'g', label='Ground Truth' if i == 0 else "")
    ax.plot(points_x_test[:, 0], points_x_test[:, 1], 'go', label='Ground Truth' if i == 0 else "")


    ax.scatter(step[j, :, 0], step[j, :, 1], label=f'Step {step_idx}')
    ax.set_title(f'Step {step_idx}')
    ax.legend()




plt.tight_layout()
plt.savefig('1mapDIffusionprocesscenter_objects_comparison2.png')
plt.show()