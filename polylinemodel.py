import numpy as np
import torch
import torch.nn as nn


from utils.transformer import transformer_encoder_layer, position_encoding_utils
from utils import polyline_encoder





class PolyLineEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        config.NUM_INPUT_ATTR_AGENT = 0
        config.NUM_CHANNEL_IN_MLP_AGENT = 256 # hidden_dim
        config.NUM_LAYER_IN_MLP_AGENT = 3
        config.D_MODEL = 256
        config.NUM_INPUT_ATTR_MAP = 1
        config.NUM_CHANNEL_IN_MLP_MAP = 64
        config.NUM_LAYER_IN_MLP_MAP = 5
        config.NUM_LAYER_IN_PRE_MLP_MAP = 3

        self.model_cfg = config

        # build polyline encoders
        self.agent_polyline_encoder = self.build_polyline_encoder(
            # in_channels, hidden_dim, num_layers=3, num_pre_layers=1

            in_channels=self.model_cfg.NUM_INPUT_ATTR_AGENT + 1,
            hidden_dim=self.model_cfg.NUM_CHANNEL_IN_MLP_AGENT,
            num_layers=self.model_cfg.NUM_LAYER_IN_MLP_AGENT,
            out_channels=self.model_cfg.D_MODEL
        )
        self.map_polyline_encoder = self.build_polyline_encoder(
            in_channels=self.model_cfg.NUM_INPUT_ATTR_MAP,
            hidden_dim=self.model_cfg.NUM_CHANNEL_IN_MLP_MAP,
            num_layers=self.model_cfg.NUM_LAYER_IN_MLP_MAP,
            num_pre_layers=self.model_cfg.NUM_LAYER_IN_PRE_MLP_MAP,
            out_channels=self.model_cfg.D_MODEL
        )

        # # build transformer encoder layers
        # self.use_local_attn =  False
        # self_attn_layers = []
        # for _ in range(self.model_cfg.NUM_ATTN_LAYERS):
        #     self_attn_layers.append(self.build_transformer_encoder_layer(
        #         d_model=self.model_cfg.D_MODEL,
        #         nhead=self.model_cfg.NUM_ATTN_HEAD,
        #         dropout= 0.1,
        #         normalize_before=False,
        #         use_local_attn=self.use_local_attn
        #     ))

        # self.self_attn_layers = nn.ModuleList(self_attn_layers)
        self.num_out_channels = self.model_cfg.D_MODEL


        """
    def __init__(self, in_channels, hidden_dim, num_layers=3, num_pre_layers=1, out_channels=None):

        Args:
            polylines (batch_size, num_polylines, num_points_each_polylines, C):
            polylines_mask (batch_size, num_polylines, num_points_each_polylines):

        Returns:
        """
    def build_polyline_encoder(self, in_channels, hidden_dim, num_layers, num_pre_layers=1, out_channels=None):
        ret_polyline_encoder = polyline_encoder.PointNetPolylineEncoder(
            in_channels=in_channels,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_pre_layers=num_pre_layers,
            out_channels=out_channels
        )
        return ret_polyline_encoder
    

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
              input_dict:
        """
        input_dict = batch_dict['input_dict']
        obj_trajs, obj_trajs_mask = input_dict['obj_trajs'].cuda(), input_dict['obj_trajs_mask'].cuda() 
        map_polylines, map_polylines_mask = input_dict['map_polylines'].cuda(), input_dict['map_polylines_mask'].cuda() 

        obj_trajs_last_pos = input_dict['obj_trajs_last_pos'].cuda() 
        map_polylines_center = input_dict['map_polylines_center'].cuda() 
        track_index_to_predict = input_dict['track_index_to_predict']

        assert obj_trajs_mask.dtype == torch.bool and map_polylines_mask.dtype == torch.bool

        num_center_objects, num_objects, num_timestamps, _ = obj_trajs.shape
        num_polylines = map_polylines.shape[1]

        # apply polyline encoder
        obj_trajs_in = torch.cat((obj_trajs, obj_trajs_mask[:, :, :, None].type_as(obj_trajs)), dim=-1)
        obj_polylines_feature = self.agent_polyline_encoder(obj_trajs_in, obj_trajs_mask)  # (num_center_objects, num_objects, C)
        map_polylines_feature = self.map_polyline_encoder(map_polylines, map_polylines_mask)  # (num_center_objects, num_polylines, C)

        # # apply self-attn
        # obj_valid_mask = (obj_trajs_mask.sum(dim=-1) > 0)  # (num_center_objects, num_objects)
        # map_valid_mask = (map_polylines_mask.sum(dim=-1) > 0)  # (num_center_objects, num_polylines)

        # global_token_feature = torch.cat((obj_polylines_feature, map_polylines_feature), dim=1) 
        # global_token_mask = torch.cat((obj_valid_mask, map_valid_mask), dim=1) 
        # global_token_pos = torch.cat((obj_trajs_last_pos, map_polylines_center), dim=1) 

        # if self.use_local_attn:
        #     global_token_feature = self.apply_local_attn(
        #         x=global_token_feature, x_mask=global_token_mask, x_pos=global_token_pos,
        #         num_of_neighbors=self.model_cfg.NUM_OF_ATTN_NEIGHBORS
        #     )
        # else:
        #     global_token_feature = self.apply_global_attn(
        #         x=global_token_feature, x_mask=global_token_mask, x_pos=global_token_pos
        #     )

        # obj_polylines_feature = global_token_feature[:, :num_objects]
        # map_polylines_feature = global_token_feature[:, num_objects:]
        assert map_polylines_feature.shape[1] == num_polylines

        # organize return features
        center_objects_feature = obj_polylines_feature[torch.arange(num_center_objects), track_index_to_predict]

        batch_dict['center_objects_feature'] = center_objects_feature
        batch_dict['obj_feature'] = obj_polylines_feature
        batch_dict['map_feature'] = map_polylines_feature
        # batch_dict['obj_mask'] = obj_valid_mask
        # batch_dict['map_mask'] = map_valid_mask
        batch_dict['obj_pos'] = obj_trajs_last_pos
        batch_dict['map_pos'] = map_polylines_center

        return batch_dict



# 当main时，执行下面代码

if __name__ == '__main__':
        # Define a dummy configuration class
    class Config:
        def __init__(self):
            self.NUM_INPUT_ATTR_AGENT = 0
            self.NUM_CHANNEL_IN_MLP_AGENT = 256
            self.NUM_LAYER_IN_MLP_AGENT = 3
            self.D_MODEL = 256
            self.NUM_INPUT_ATTR_MAP = 1
            self.NUM_CHANNEL_IN_MLP_MAP = 64
            self.NUM_LAYER_IN_MLP_MAP = 5
            self.NUM_LAYER_IN_PRE_MLP_MAP = 3
            self.NUM_ATTN_LAYERS = 2
            self.NUM_ATTN_HEAD = 8
            self.DROPOUT_OF_ATTN = 0.1
            self.USE_LOCAL_ATTN = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = Config()
    # Create a dummy batch_dict
    batch_size = 10
    batch_dict = {
        'input_dict': {
            'obj_trajs': torch.randn(batch_size, config.NUM_INPUT_ATTR_AGENT + 1, 20, 5),  # (num_center_objects, num_objects, num_timestamps, C)
            'obj_trajs_mask': torch.ones(batch_size,  config.NUM_INPUT_ATTR_AGENT + 1, 20, dtype=torch.bool),
            'map_polylines': torch.randn(batch_size,  config.NUM_INPUT_ATTR_MAP, 20, 7),  # (num_center_objects, num_polylines, num_points_each_polylines, C)
            'map_polylines_mask': torch.ones(batch_size, config.NUM_INPUT_ATTR_MAP, 20, dtype=torch.bool),
            'obj_trajs_last_pos': torch.randn(batch_size, config.NUM_INPUT_ATTR_AGENT+1, 2),  # (num_center_objects, num_objects, 2)
            'map_polylines_center': torch.randn(batch_size, config.NUM_INPUT_ATTR_MAP, 2),  # (num_center_objects, num_polylines, 2)
            'track_index_to_predict': torch.tensor([0, 1])  # (num_center_objects)
        }
    }
            #    polylines (batch_size, num_polylines, num_points_each_polylines, C):
            # polylines_mask (batch_size, num_polylines, num_points_each_polylines):

    # Initialize the PolyLineEncoder with the configuration
    
    polyline_encoder = PolyLineEncoder(config).to(device)

    # Forward pass
    output_dict = polyline_encoder(batch_dict)

    # Print the output
    print(output_dict['center_objects_feature'])
    print(output_dict['obj_feature'])
    print(output_dict['obj_pos'])
    print(output_dict['map_pos'])