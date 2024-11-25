# utils_map.py
import numpy as np

def get_interested_agents(track_index_to_predict, obj_trajs_full, current_time_index, obj_types, scene_id):
    center_objects_list = []
    track_index_to_predict_selected = []

    for k in range(len(track_index_to_predict)):
        obj_idx = track_index_to_predict[k]

        assert obj_trajs_full[obj_idx, current_time_index, -1] > 0, f'obj_idx={obj_idx}, scene_id={scene_id}'

        center_objects_list.append(obj_trajs_full[obj_idx, current_time_index])
        track_index_to_predict_selected.append(obj_idx)

    center_objects = np.stack(center_objects_list, axis=0)  # (num_center_objects, num_attrs)
    track_index_to_predict = np.array(track_index_to_predict_selected)
    return center_objects, track_index_to_predict




# 写个测试用例
if __name__ == "__main__":
    track_index_to_predict = np.array([0, 1, 2, 3, 4])
    obj_trajs_full = np.random.rand(5, 10, 10)
    current_time_index = 9
    obj_types = np.array(['TYPE_VEHICLE', 'TYPE_CYCLIST', 'TYPE_PEDESTRIAN', 'TYPE_VEHICLE', 'TYPE_CYCLIST'])
    scene_id = 0
    center_objects, track_index_to_predict = get_interested_agents(track_index_to_predict, obj_trajs_full, current_time_index, obj_types, scene_id)
    print(center_objects.shape)  # (5, 10)
    print(center_objects)
    print(track_index_to_predict)  # [0 1 2 3 4]