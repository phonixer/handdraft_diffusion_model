import pickle
import numpy as np
import os
import sys
import torch
from utlits_map import get_interested_agents
from utils import polyline_encoder
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
    # plt.plot(center_objects[i, 0], center_objects[i, 1], 'ro', label='Center' if i == 1 else "")

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
num_points_each_polyline = 5  # 每个polyline的点数


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
num_of_src_polylines = 8
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

# for value in unique_values:
#     color = value_to_color[value]
#     plt.plot([], [], color = 'g', label=f'Value {value}')

plt.legend()

# 保存并显示图像
plt.savefig('map_polylines.png')
plt.show()




# obj_trajs_past
# obj_trajs_future
# # 输出形状
print(obj_trajs_past.shape)  # (100, 11, 10)
print(obj_trajs_future.shape)  # (100, 80, 10)

from diffusion_model_map import Diffusion

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# device = "cpu"  # cuda

act_dim = num_of_src_polylines * num_points_each_polyline * 2
obs_dim = num_of_src_polylines * num_points_each_polyline * 2


print('map_polylines.shape', map_polylines.shape)

# print(map_polylines([map_polylines_mask]).shape)
print('map_polylines_mask', map_polylines_mask.shape)
# 将掩码扩展到与 map_polylines 相同的形状
mask_expanded = map_polylines_mask.unsqueeze(-1).expand_as(map_polylines)

# 使用掩码过滤 map_polylines 中的数据
filtered_map_polylines = map_polylines * mask_expanded

print('filtered_map_polylines.shape', filtered_map_polylines.shape)


# 生成 x 张量，并转换为浮点类型
x = map_polylines[:,:,:,:2] 
# 生成 state 张量，每个 batch 的值都相同，并转换为浮点类型
state = map_polylines[:,:,:,:2]
mask_expanded = map_polylines_mask.unsqueeze(-1).expand_as(x)



# 归一化 变到 -1, 1
# valid_points = (x[:, :, :, 0] != 0) | (x[:, :, :, 1] != 0)
# print(valid_points.shape)
# print(mask_expanded.shape)
# mask_expanded = map_polylines_mask | valid_points

mask_expanded = map_polylines_mask
mask_expanded = mask_expanded.unsqueeze(-1).expand_as(x)
print(mask_expanded.shape)

batch_size, num_polylines, num_points_each_polyline, _ = x.shape
# 每组polyline 减去 center object的位置
# print(center_objects[:, None, None, 0:2].shape) 
x = x - center_objects[:, None, None, 0:2] #实现了每组polyline 减去 center object的位置
x[2,:,:,:] = x[4,:,:,:]
x_0 = x[:,:,:, 0][map_polylines_mask == 1]
x_1 = x[:,:,:, 1][map_polylines_mask == 1]
print(x_0.shape)
# 对 x[:, :, :, 0] 进行归一化
x_min_0 = x_0.min()
x_max_0 = x_0.max()
x[:, :, :, 0] = (x[:, :, :, 0] - x_min_0) / (x_max_0 - x_min_0)  - 0.5

# 对 x[:, :, :, 1] 进行归一化
x_min_1 = x_1.min()
x_max_1 = x_1.max()
x[:, :, :, 1] = (x[:, :, :, 1] - x_min_1) / (x_max_1 - x_min_1)  - 0.5

state_min = state[mask_expanded == 1].min()
state_max = state[mask_expanded == 1].max()
state = (state - state_min) / (state_max - state_min) * 2 - 1


# 画图
x.shape
num_center_objects, num_of_src_polylines, num_points_each_polyline, _ = x.shape
print(x.shape)
plt.figure(figsize=(15, 5))
for i in range(0, num_center_objects):
    # plt.plot(center_objects[i, 0], center_objects[i, 1], 'ro', label='Center' if i == 1 else "")
    
    for j in range(num_of_src_polylines):
        # 排除掉（0，0）
        valid_points = (abs(x[i, j, :, 0]) < 2) | (abs(x[i, j, :, 1]) < 2)
        for value in np.unique(valid_values):
            points = x[i, j, valid_points, :]
            plt.plot(points[:, 0], points[:, 1])

plt.savefig('x_state.png')



# # 复制 x 和 state，使其形状为 (100, 10, 2)
x = x.repeat(40, 1, 1, 1)
state = state.repeat(40, 1, 1, 1)
map_polylines_mask = map_polylines_mask.repeat(40, 1, 1)
# # Reshape x and state to (100,)
# x = x.view(batchsize, -1)
# state = state.view(batchsize, -1)
print('x.shape', x.shape)
print(state.shape)




device = 'cuda' if torch.cuda.is_available() else 'cpu'

lr = 4e-5
epoch = 10000
obs_dim = 11
batch_size = 320
num_polylines = num_of_src_polylines
num_points_each_polylines = num_points_each_polyline
in_channels = 2
hidden_dim = 256
T = 10
loss_type = 'l2'
beta_schedule = 'linear'
clip_denoised = True
predict_epsilon = False
t_dim = 16
num_layers = 3
num_pre_layers = 1
out_channels = 10
mlp_hidden_dim = 1024
train_model = True

mlp_out_dim = num_polylines * num_points_each_polylines * in_channels
act_dim = num_polylines * num_points_each_polylines * in_channels

# polylines = torch.randn(batch_size, num_polylines, num_points_each_polylines, in_channels).to(device)
# polylines_mask = torch.randint(0, 2, (batch_size, num_polylines, num_points_each_polylines)).bool().to(device)
# polylines = map_polylines
polylines_mask = map_polylines_mask.bool().to(device)
x = x.to(device)
state = {'polylines': x, 'polylines_mask': polylines_mask}
x = x.view(batch_size, -1)
print('state[\'polylines\'].shape, state[\'polylines_mask\'].shape', state['polylines'].shape, state['polylines_mask'].shape)
# print(x.shape)
# print(map_polylines_mask.shape)
# assert x.shape == (batch_size, num_polylines, num_points_each_polylines, in_channels)
# assert map_polylines_mask.shape == (batch_size, num_polylines, num_points_each_polylines)
# # x = torch.randn(batch_size, act_dim).to(device)
# state = {'polylines': x.to(device), 'polylines_mask': map_polylines_mask.bool().to(device)}

model = Diffusion(
    loss_type=loss_type,
    beta_schedule=beta_schedule,
    clip_denoised=clip_denoised,
    predict_epsilon=predict_epsilon,
    obs_dim=obs_dim,
    act_dim=act_dim,
    hidden_dim=hidden_dim,
    device=device,
    T=T,
    t_dim=t_dim,
    num_polylines=num_polylines,
    num_points_each_polylines=num_points_each_polylines,
    in_channels=in_channels,
    num_layers=num_layers,
    num_pre_layers=num_pre_layers,
    out_channels=out_channels,
    mlp_hidden_dim=mlp_hidden_dim,
    mlp_out_dim=mlp_out_dim
)
print(model)
result, diffusion_steps = model(state)



# model = Diffusion(loss_type='l2', obs_dim=obs_dim, act_dim=act_dim, hidden_dim= 2 * act_dim, device=device, T=T)
# result = model(state)  # Sample result

loss = model.loss(x, state)

print(f"action: {result};loss: {loss.item()}")
import matplotlib.pyplot as plt
import torch.optim as optim

# print(sum(state['polylines'].view(batch_size, -1) - x))
# exit(0)

optimizer = optim.Adam(model.parameters(), lr=lr)
# T = 200

# Check if training is required
model_path = 'diffusion_model.pth'

if os.path.exists(model_path) and train_model == False:
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
    def lr_lambda(epoch):
        return 0.5 ** (epoch // 10000)

    scheduler = LambdaLR(optimizer, lr_lambda)
    model.train()
    for i in range(epoch):
        loss = model.loss(x, state)
        loss.backward()
        if i % 1000 == 0:
            print(f'epoch: {i}, loss: {loss.item()}')
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
    
    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

# Evaluate the model
model.eval()
result, diffusion_steps = model(state)
print('result', result.shape)
loss = model.loss(x, state)
# 训练结束后绘制扩散过程的图像
action = result
x_test = x
print(f"action: {action.shape}; loss: {loss.item()}")

# Output ground truth shape
print('x_test:', x_test.shape)
print(len(diffusion_steps))

# Restore shapes
x_test = x_test.view(-1, num_polylines, num_points_each_polyline, in_channels)
action = action.view(-1, num_polylines, num_points_each_polyline, in_channels)
print(x_test.shape)
print(action.shape)

# Plot diffusion process
num_steps = len(diffusion_steps)
steps_to_plot = [int(i * num_steps / 10) for i in range(10)] + [num_steps - 1]
# # 逆归一化
x_test[:, :, :, 0] = (x_test[:, :, :, 0] + 0.5) * (x_max_0 - x_min_0) + x_min_0
x_test[:, :, :, 1] = (x_test[:, :, :, 1] + 0.5) * (x_max_1 - x_min_1) + x_min_1

action[:, :, :, 0] = (action[:, :, :, 0] + 0.5) * (x_max_0 - x_min_0) + x_min_0
action[:, :, :, 1] = (action[:, :, :, 1] + 0.5) * (x_max_1 - x_min_1) + x_min_1
# # 加上center object的位置
# x_test = x_test + center_objects[:, None, None, 0:2]
# action = action + center_objects[:, None, None, 0:2]

x_test = x_test.cpu().detach().numpy()
action = action.cpu().detach().numpy()

# Calculate errors
total_error = np.linalg.norm(action - x_test, axis=-1).sum()
average_error = total_error / np.prod(x_test.shape[:-1])
percentage_error = (total_error / np.linalg.norm(x_test, axis=-1).sum()) * 100

print(f"Total Error: {total_error}")
print(f"Average Error: {average_error}")
print(f"Percentage Error: {percentage_error}%")





plt.figure(figsize=(15, 5))

for i in range(0, num_center_objects):
    for j in range(num_of_src_polylines):
        # 排除掉（0，0）
        valid_points = (abs(action[i, j, :, 0]) < 2) & (abs(action[i, j, :, 1]) < 2)
        points = action[i, j, valid_points, :]
        plt.plot(points[:, 0], points[:, 1], label=f'Action {i}' if j == 0 else "")
plt.savefig('action_plot.png')
plt.show()

plt.figure(figsize=(15, 5))
for step_idx in steps_to_plot:
    step = diffusion_steps[step_idx].cpu().detach().numpy().reshape(-1, 2)
    # print('step', step.shape)
    plt.scatter(step[:, 0], step[:, 1], label=f'Step {step_idx}')

for i in range(0, num_center_objects):
    for j in range(num_of_src_polylines):
        # 排除掉（0，0）
        valid_points = (abs(x_test[i, j, :, 0]) < 2) & (abs(x_test[i, j, :, 1]) < 2)
        points = x_test[i, j, valid_points, :]
        plt.plot(points[:, 0], points[:, 1], label=f'x_test {i}' if j == 0 else "")
masked_x_test = x_test[polylines_mask.cpu().detach().numpy()]
masked_action = action[polylines_mask.cpu().detach().numpy()]

# 考虑polyline mask
plt.scatter(masked_x_test[:, 0], masked_x_test[:, 1], label='Ground Truth', color='g')
plt.scatter(masked_action[:, 0], masked_action[:, 1], label='Predicted', color='r')

plt.title('Diffusion Process')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.tight_layout()
plt.savefig('diffusion.png')
plt.show()


# Plotting each num_center_objects separately
num_plots_per_row = 4
num_rows = (num_center_objects + num_plots_per_row - 1) // num_plots_per_row

fig, axes = plt.subplots(num_rows, num_plots_per_row, figsize=(20, 5 * num_rows))
axes = axes.flatten()

for i in range(num_center_objects):
    ax = axes[i]
    for j in range(num_of_src_polylines):
        valid_points = (abs(action[i, j, :, 0]) < 2) & (abs(action[i, j, :, 1]) < 2)
        points_action = action[i, j, valid_points, :]
        points_x_test = x_test[i, j, valid_points, :]
        ax.plot(points_action[:, 0], points_action[:, 1], 'r', label='Predicted' if j == 0 else "")
        ax.plot(points_x_test[:, 0], points_x_test[:, 1], 'g', label='Ground Truth' if j == 0 else "")
    ax.set_title(f'Center Object {i}')
    ax.legend()

# Hide any unused subplots
for i in range(num_center_objects, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.savefig('center_objects_comparison.png')
plt.show()

for step_idx in range(T):
    step = diffusion_steps[step_idx].view(batch_size, num_polylines, num_points_each_polylines, in_channels)
    # 逆归一化
    step[:,:, :, 0] = (step[:,:, :, 0] + 0.5) * (x_max_0 - x_min_0) + x_min_0
    step[:,:, :, 1] = (step[:,:, :, 1] + 0.5) * (x_max_1 - x_min_1) + x_min_1



print('diffusion_steps',diffusion_steps[0].shape)
# Plotting each num_center_objects separately
num_plots_per_row = 4
num_rows = (num_center_objects + num_plots_per_row - 1) // num_plots_per_row

fig, axes = plt.subplots(num_rows, num_plots_per_row, figsize=(20, 5 * num_rows))
axes = axes.flatten()

for i in range(num_center_objects):
    ax = axes[i]
    # # 画center object 和轨迹
    # # 减去第一个点，得到相对位置
    # obj_trajs_full = obj_trajs_full - obj_trajs_full[:, 0:1, :]
    # obj_trajs_past = obj_trajs_past - obj_trajs_past[:, 0:1, :]
    # obj_trajs_future = obj_trajs_future - obj_trajs_future[:, 0:1, :]
    
    # ax.plot(obj_trajs_past[track_index_to_predict[i], :, 0], obj_trajs_past[track_index_to_predict[i], :, 1], 'b', label='Past' if i == 1 else "")
    # ax.plot(obj_trajs_future[track_index_to_predict[i], :, 0], obj_trajs_future[track_index_to_predict[i], :, 1], 'r', label='Future' if i == 1 else "")
    # # ax.plot(center_objects[i, 0], center_objects[i, 1], 'ro', label='Center' if i == 1 else "")

    for step_idx in steps_to_plot:
        step = diffusion_steps[step_idx].view(batch_size, num_polylines, num_points_each_polylines, in_channels)

        step = step[i,:,:, :].cpu().detach().numpy().reshape(-1, 2)
        # print('step', step.shape)
        ax.scatter(step[:, 0], step[:, 1], label=f'Step {step_idx}')
    for j in range(num_of_src_polylines):


        valid_points = (abs(action[i, j, :, 0]) < 2) & (abs(action[i, j, :, 1]) < 2)
        points_action = action[i, j, valid_points, :]
        points_x_test = x_test[i, j, valid_points, :]
        ax.plot(points_action[:, 0], points_action[:, 1], 'r', label='Predicted' if j == 0 else "")
        ax.plot(points_x_test[:, 0], points_x_test[:, 1], 'g', label='Ground Truth' if j == 0 else "")
    ax.set_title(f'Center Object {i}')
    ax.legend()

# Hide any unused subplots
for i in range(num_center_objects, len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.savefig('DIffusionprocesscenter_objects_comparison.png')
plt.show()







num_plots_per_row = 11
num_rows = (len(steps_to_plot) + num_plots_per_row - 1) // num_plots_per_row

fig, axes = plt.subplots(num_rows, num_plots_per_row, figsize=(40, 5 * num_rows))
axes = axes.flatten()

for i in range(len(steps_to_plot)):
    ax = axes[i]

    step_idx = steps_to_plot[i]
    step = diffusion_steps[step_idx].view(batch_size, num_polylines, num_points_each_polylines, in_channels)


    step = step[0,:,:, :].cpu().detach().numpy().reshape(-1, 2)
    # print('step', step.shape)
    ax.scatter(step[:, 0], step[:, 1], label=f'Step {step_idx}')
    for j in range(num_of_src_polylines):

        valid_points = (abs(action[0, j, :, 0]) < 2) & (abs(action[0, j, :, 1]) < 2)
        points_action = action[0, j, :, :]
        points_x_test = x_test[0, j, :, :]
        ax.plot(points_action[:, 0], points_action[:, 1], 'r', label='Predicted' if j == 0 else "")
        ax.plot(points_x_test[:, 0], points_x_test[:, 1], 'g', label='Ground Truth' if j == 0 else "")
    ax.set_title(f'Diffusion Step {step_idx}')
    # ax.legend()



plt.tight_layout()
plt.savefig('1DIffusionprocesscenter_objects_comparison.png')
plt.show()


num_plots_per_row = 5
steps_to_plot = range(T-5,T)
num_rows = (len(steps_to_plot) + num_plots_per_row - 1) // num_plots_per_row

fig, axes = plt.subplots(num_rows, num_plots_per_row, figsize=(20, 5 * num_rows))
axes = axes.flatten()

for i in range(len(steps_to_plot)):
    ax = axes[i]

    step_idx = steps_to_plot[i]
    step = diffusion_steps[step_idx].view(batch_size, num_polylines, num_points_each_polylines, in_channels)


    step = step[0,:,:, :].cpu().detach().numpy().reshape(-1, 2)
    # print('step', step.shape)
    ax.scatter(step[:, 0], step[:, 1], label=f'Step {step_idx}')
    for j in range(num_of_src_polylines):

        valid_points = (abs(action[0, j, :, 0]) < 2) & (abs(action[0, j, :, 1]) < 2)
        points_action = action[0, j, :, :]
        points_x_test = x_test[0, j, :, :]
        ax.plot(points_action[:, 0], points_action[:, 1], 'r', label='Predicted' if j == 0 else "")
        ax.plot(points_x_test[:, 0], points_x_test[:, 1], 'g', label='Ground Truth' if j == 0 else "")
    ax.set_title(f'Diffusion Step {step_idx}')
    # ax.legend()



plt.tight_layout()
plt.savefig('1DIffusionprocesscenter_objects_comparison2.png')
plt.show()




# 画降噪过程
# Plotting the results
plt.figure(figsize=(15, 5))
num_plots_per_row = 5
polt_item = [0,2,3,4,6]
num_rows = (len(polt_item) + num_plots_per_row - 1) // num_plots_per_row

fig, axes = plt.subplots(num_rows, num_plots_per_row, figsize=(20, 5 * num_rows))
axes = axes.flatten()

num_steps = len(diffusion_steps)
steps_to_plot = [int(i * num_steps / 10) for i in range(10)] + [num_steps - 1]
print('steps_to_plot', steps_to_plot)
for i in range(len(polt_item)):
    ax = axes[i]
    for j in range(len(steps_to_plot)):
        step = diffusion_steps[steps_to_plot[j]].view(batch_size, num_polylines, num_points_each_polyline, in_channels)
        step = step[polt_item[i],:,:, :].cpu().detach().numpy().reshape(-1, 2)

        # 画点
        # ax.plot(points[:, 0], points[:, 1], 'ro', alpha=j / num_steps)
        ax.plot(step[:, 0], step[:, 1],'ro', label=f'Step {steps_to_plot[j]}', alpha=steps_to_plot[j] / num_steps)

    
    for j in range(num_of_src_polylines):
        valid_points = (abs(action[polt_item[i], j, :, 0]) < 2) & (abs(action[polt_item[i], j, :, 1]) < 2)
        points_action = action[polt_item[i], j, valid_points, :]
        points_x_test = x_test[polt_item[i], j, valid_points, :]
        ax.plot(points_action[:, 0], points_action[:, 1], 'r', label='Predicted' if j == 0 else "")
        ax.plot(points_x_test[:, 0], points_x_test[:, 1], 'g', label='Ground Truth' if j ==0 else "")


    ax.set_title(f'Center Object {i}')


# plt.title('Predicted vs Ground Truth Polylines')
plt.legend()
plt.tight_layout()
plt.savefig('Diffusion5polylinesMAP.png')
plt.show()


polt_item = [0,2,3,4,6]
#截取
x_test = x_test[polt_item]
action = action[polt_item]

# Calculate errors
total_error = np.linalg.norm(action - x_test, axis=-1).sum()
average_error = total_error / np.prod(x_test.shape[:-1])
percentage_error = (total_error / np.linalg.norm(x_test, axis=-1).sum()) * 100

print(f"Total Error: {total_error}")
print(f"Average Error: {average_error}")
print(f"Percentage Error: {percentage_error}%")


