# 自学扩散模型
太有意思了
![alt text](image.png)


第一步看科华的代码有没有地图

## 2024 1121
看一下MID的数据处理是怎么处理的

### Process_data
这个代码的主要功能是处理和预处理轨迹数据，并将其保存为 .pkl 文件，以便后续用于训练和评估轨迹预测模型。以下是代码的详细解释：

"""from environment import Environment, Scene, Node, derivative_of"""

定义了一些新的数据结构

为每个数据源和数据类别创建一个 Environment 对象，并设置注意力半径。

遍历原始数据文件夹中的所有 .txt 文件，读取并处理数据。

将数据转换为 Scene 和 Node 对象，并计算速度和加速度。

对训练数据进行数据增强（旋转）

将处理后的场景添加到环境中，并使用 dill 序列化保存为 .pkl 文件。



##### pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])
![alt text](image-1.png)

是 Pandas 库中的一个方法，用于创建一个多级索引（MultiIndex），其级别是输入序列的笛卡尔积。多级索引允许在一个 DataFrame 中使用多个索引级别，从而可以更灵活地表示和操作数据。

> 举例
假设集合A={a,b}，集合B={0,1,2}，则两个集合的笛卡尔积为{(a,0),(a,1),(a,2),(b,0),(b,1),(b,2)}。类似的例子有，如果A表示某学校学生的集合，B表示该学校所有课程的集合，则A与B的笛卡尔积表示所有可能的选课情况。
> 应用
在数据库中，笛卡尔积常用于描述两个表之间所有可能的配对情况。当在查询中连接两个表时，如果没有指定适当的连接条件，就可能产生笛卡尔积，这通常会导致非常庞大的结果集。


## Training 
For example, train with 8 GPUs: 
```
cd tools

bash scripts/dist_train.sh 8 --cfg_file cfgs/waymo/mtr+100_percent_data.yaml --batch_size 80 --epochs 30 --extra_tag my_first_exp
```
Actually, during the training process, the evaluation results will be logged to the log file under `output/waymo/mtr+100_percent_data/my_first_exp/log_train_xxxx.txt`

## Testing
For example, test with 8 GPUs: 
```
cd tools
bash scripts/dist_test.sh 8 --cfg_file cfgs/waymo/mtr+100_percent_data.yaml --ckpt ../output/waymo/mtr+100_percent_data/my_first_exp/ckpt/checkpoint_epoch_30.pth --extra_tag my_first_exp --batch_size 80 
```


waymo 数据集的表头
dict_keys(['track_infos', 'dynamic_map_infos', 'map_infos', 'scenario_id', 'timestamps_seconds', 'current_time_index', 'sdc_track_index', 'objects_of_interest', 'tracks_to_predict'])

检查 HDMap 数据：

如果配置中没有设置 WITHOUT_HDMAP 或者其值为 False，则继续处理 HDMap 数据。
检查 info['map_infos']['all_polylines'] 是否为空。如果为空，则用形状为 (2, 7) 的零数组填充，并打印警告信息。
创建地图数据：

调用 self.create_map_data_for_center_objects 方法，为中心对象创建地图数据。该方法返回三个值：map_polylines_data、map_polylines_mask 和 map_polylines_center。
map_polylines_data：形状为 (num_center_objects, num_topk_polylines, num_points_each_polyline, 9) 的数组，表示中心对象的地图折线数据。
map_polylines_mask：形状为 (num_center_objects, num_topk_polylines, num_points_each_polyline) 的布尔掩码数组。
map_polylines_center：中心对象的地图折线中心数据。
更新返回字典：

将 map_polylines_data、map_polylines_mask 和 map_polylines_center 添加到返回字典 ret_dict 中。
map_polylines_mask 被转换为布尔值，表示哪些折线是有效的。
总结来说，这段代码在处理 Waymo 数据集时，检查并处理 HDMap 数据，为中心对象创建地图折线数据，并将这些数据添加到返回字典中。