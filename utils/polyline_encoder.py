import torch
import torch.nn as nn
# from utils import common_layers
import common_layers

class PointNetPolylineEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layers=3, num_pre_layers=1, out_channels=None):
        super().__init__()
        self.pre_mlps = common_layers.build_mlps(
            c_in=in_channels,
            mlp_channels=[hidden_dim] * num_pre_layers,
            ret_before_act=False
        )
        self.mlps = common_layers.build_mlps(
            c_in=hidden_dim * 2,
            mlp_channels=[hidden_dim] * (num_layers - num_pre_layers),
            ret_before_act=False
        )
        
        if out_channels is not None:
            self.out_mlps = common_layers.build_mlps(
                c_in=hidden_dim, mlp_channels=[hidden_dim, out_channels], 
                ret_before_act=True, without_norm=True
            )
        else:
            self.out_mlps = None 

    def forward(self, polylines, polylines_mask):
        batch_size, num_polylines,  num_points_each_polylines, C = polylines.shape

        # pre-mlp
        polylines_feature_valid = self.pre_mlps(polylines[polylines_mask])  # (N, C)
        polylines_feature = polylines.new_zeros(batch_size, num_polylines,  num_points_each_polylines, polylines_feature_valid.shape[-1])
        polylines_feature[polylines_mask] = polylines_feature_valid

        # get global feature
        pooled_feature = polylines_feature.max(dim=2)[0]
        polylines_feature = torch.cat((polylines_feature, pooled_feature[:, :, None, :].repeat(1, 1, num_points_each_polylines, 1)), dim=-1)

        # mlp
        polylines_feature_valid = self.mlps(polylines_feature[polylines_mask])
        feature_buffers = polylines_feature.new_zeros(batch_size, num_polylines, num_points_each_polylines, polylines_feature_valid.shape[-1])
        feature_buffers[polylines_mask] = polylines_feature_valid

        # max-pooling
        feature_buffers = feature_buffers.max(dim=2)[0]  # (batch_size, num_polylines, C)
        
        # out-mlp 
        if self.out_mlps is not None:
            valid_mask = (polylines_mask.sum(dim=-1) > 0)
            feature_buffers_valid = self.out_mlps(feature_buffers[valid_mask])  # (N, C)
            feature_buffers = feature_buffers.new_zeros(batch_size, num_polylines, feature_buffers_valid.shape[-1])
            feature_buffers[valid_mask] = feature_buffers_valid
        return feature_buffers

class MLPWithPolylineEncoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_layers, num_pre_layers, out_channels, mlp_hidden_dim, mlp_out_dim):
        super().__init__()
        self.encoder = PointNetPolylineEncoder(in_channels, hidden_dim, num_layers, num_pre_layers, out_channels)

        mlp_in_dim = 8 * out_channels
        print("mlp_in_dim:", mlp_in_dim)
        self.mlp = nn.Sequential(
            nn.Linear(mlp_in_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(mlp_hidden_dim, mlp_out_dim)
        )

    def forward(self, polylines, polylines_mask):
        encoded_features = self.encoder(polylines, polylines_mask)

        print("encoded_features Shape:", encoded_features.shape)

        encoded_features = encoded_features.reshape(encoded_features.shape[0], -1)
        print("encoded_features Shape: reshape", encoded_features.shape)

        output = self.mlp(encoded_features)
        print("output Shape no reshape:", output.shape)
        output = output.reshape(polylines.shape[0], polylines.shape[1],-1)
        print("output Shape reshape:", output.shape)
        return output

    def compute_loss(self, output, target, mask):
        loss_fn = nn.MSELoss(reduction='none')
        loss = loss_fn(output, target)
        mask = mask.unsqueeze(-1).expand_as(loss)  # Ensure mask has the same shape as loss
        loss = loss * mask
        return loss.mean()

if __name__ == "__main__":

    batch_size = 100
    num_polylines = 8
    num_points_each_polylines = 20
    in_channels = 2
    hidden_dim = 64
    num_layers = 3
    num_pre_layers = 1
    out_channels = 10   # emb的维度
    mlp_hidden_dim = 256
    mlp_out_dim = num_polylines * 40 # 20 * 2
    model = MLPWithPolylineEncoder(in_channels, hidden_dim, num_layers, num_pre_layers, out_channels, mlp_hidden_dim, mlp_out_dim)

    # Create polylines with y-coordinates as horizontal lines
    polylines = torch.zeros(batch_size, num_polylines, num_points_each_polylines, in_channels)
    for i in range(num_polylines):
        polylines[0, i, :, 0] = torch.linspace(0, 10, num_points_each_polylines)  # x-coordinates
        polylines[0, i, :, 1] = i  # y-coordinates

    # Create a mask to remove a specific line (e.g., the 4th line)
    polylines_mask = torch.ones(batch_size, num_polylines, num_points_each_polylines).bool()
    polylines_mask[0, 3, :] = False  # Remove the 4th line

    x = torch.randn(batch_size, num_polylines, 40)  # Example input for MLP
    target = polylines.reshape(batch_size, num_polylines, -1)  # Example target for loss calculation
    print("target Shape:", target.shape)

    output = model(polylines, polylines_mask)
    loss = model.compute_loss(output, target, polylines_mask.sum(dim=-1) > 0)
    print(model)

    print("Output Shape:", output.shape)
    print("Loss:", loss.item())


        # Apply the mask to remove the specific line
    masked_polylines = polylines.clone()
    masked_polylines[~polylines_mask] = float('nan')  # Set masked points to NaN for plotting


    import matplotlib.pyplot as plt
    # Plot the polylines before applying the mask
    plt.figure(figsize=(10, 5))
    for i in range(num_polylines):
        plt.plot(polylines[0, i, :, 0], polylines[0, i, :, 1], label=f'Polyline {i+1}')
    plt.title('Polylines Before Applying Mask')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.savefig('before.png')
    # Plot the polylines after applying the mask
    plt.figure(figsize=(10, 5))
    for i in range(num_polylines):
        plt.plot(masked_polylines[0, i, :, 0], masked_polylines[0, i, :, 1], label=f'Polyline {i+1}')
    plt.title('Polylines After Applying Mask')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.savefig('after.png')



    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 1000
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(polylines, polylines_mask)
        loss = model.compute_loss(output, target, polylines_mask.sum(dim=-1) > 0)
        loss.backward()
        optimizer.step()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}')

    # Plot the model output vs target for each polyline
    print("Output Shape:", output.shape)

    target = target.reshape(batch_size, num_polylines, num_points_each_polylines, in_channels)
    output = output.reshape(batch_size, num_polylines, num_points_each_polylines, in_channels)
    # 考虑mask
    target[~polylines_mask] = float('nan')
    output[~polylines_mask] = float('nan')
    plt.figure(figsize=(10, 5))
    for i in range(num_polylines):
        plt.plot(target[0, i, :, 0].detach().numpy(), target[0, i, :, 1].detach().numpy(), label=f'Polyline {i+1} Target')
        plt.plot(output[0, i, :, 0].detach().numpy(), output[0, i, :, 1].detach().numpy() + 0.1, label=f'Polyline {i+1} Output')
    plt.title(f'Epoch {epoch+1} - Model Output vs Target')
    plt.xlabel('Point Index')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(f'output_vs_target_epoch_{epoch+1}.png')
    plt.close()