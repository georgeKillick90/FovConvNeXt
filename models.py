from .retina import *
from .layers import Conv, DWise, ConvV2, ExpCentroid1d, AvgPool
from .tessellations import *
from einops import repeat
from torchvision.models.convnext import ConvNeXt, CNBlockConfig
import torch.nn as nn
import torch.nn.functional as F
import torch


class Block(nn.Module):
    def __init__(self, in_channels, vertices, k, sigma, max_ord):
        super().__init__()
        
        self.dwise = ConvV2(in_channels,
                            in_channels,
                            vertices,
                            vertices,
                            sigma,
                            max_ord,
                            groups=in_channels)
        
        self.bn = nn.BatchNorm1d(in_channels)
        self.up = nn.Conv1d(in_channels, in_channels * 4, 1)
        self.down = nn.Conv1d(in_channels * 4, in_channels, 1)
        self.act = nn.GELU()
        
    def forward(self, x):
        identity = x
        x = self.dwise(x)
        x = self.bn(x)
        x = self.up(x)
        x = self.act(x)
        x = self.down(x)
        x = x + identity
        return x

class DownLayer(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 in_vertices,
                 out_vertices,
                 k,
                 sigma,
                 max_ord,
                 patch_conv=False):
        
        super().__init__()
        
        self.pool = AvgPool(in_vertices, out_vertices, k)
        self.poolconv = nn.Conv1d(in_channels, out_channels, 1)

        if in_vertices.shape[0] == out_vertices.shape[1]:
            self.conv = nn.Conv1d(in_channels, out_channels, 1)

        elif patch_conv:
            self.conv = Conv(in_channels, out_channels, in_vertices, out_vertices, k, sigma, max_ord)

        else:
            self.conv = ConvV2(in_channels, out_channels, in_vertices, out_vertices, k, sigma, max_ord)

        self.bn = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        identity = self.poolconv(self.pool(x))
        x = self.conv(x)
        x = self.bn(x)
        x = x + identity
        return x

class ConvNext(nn.Module):
    def __init__(self,
                 radius,
                 spatial_dims,
                 width,
                 depth,
                 block_sigma,
                 block_max_ord,
                 patch_sigma,
                 patch_max_ord,
                 ds_sigma,
                 ds_max_ord,
                 n_classes
                ):
        
        super().__init__()
        
        self.block_sigma = block_sigma
        self.block_max_ord = block_max_ord
        self.ds_sigma = ds_sigma
        self.ds_max_ord = ds_max_ord
        
        # get vertices for input image and featuremaps (i.e pixel locs)
        f_verts = self.get_feature_vertices(spatial_dims, radius)
        
        # samples uniform input to get foveated image
        self.retina = LightRetina(f_verts[0], padding_mode='border')
        
        # patchify style graph conv
        self.patchconv = nn.Sequential(Conv(3, width[0], f_verts[0], f_verts[1], 16, patch_sigma, patch_max_ord, False),
                                       nn.BatchNorm1d(width[0]))
        
        self.stage1 = self.make_layer(depth[0], width[0], width[1], f_verts[1], f_verts[2])
        self.stage2 = self.make_layer(depth[1], width[1], width[2], f_verts[2], f_verts[3])
        self.stage3 = self.make_layer(depth[2], width[2], width[3], f_verts[3], f_verts[4])
        self.stage4 = self.make_layer(depth[3], width[3], width[3], f_verts[4], f_verts[4])
        
        self.where_stage = nn.Sequential(nn.Conv1d(width[3], 1, 1),
                                         ExpCentroid1d(f_verts[4].T, 1.0))
        
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Linear(width[3], n_classes)
    
    def forward(self, x, fixations=None):
        x = self.retina(x, fixations)
        x = self.patchconv(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        what = self.stage4(x)
        where = self.where_stage(x.clone().detach())
        
        what = self.pool(what).view(x.shape[0], -1)
        what = self.fc(what)
        
        return where, what
    
    def make_layer(self, depth, width_in, width_out, dim_in, dim_out, downsample=True):
        blocks = []
        
        for _ in range(depth):
            blocks.append(Block(width_in, dim_in, 49, self.block_sigma, self.block_max_ord))
        
        if downsample:
            blocks.append(DownLayer(width_in, width_out, dim_in, dim_out, 4, self.ds_sigma, self.ds_max_ord, True))
        
        return nn.Sequential(*blocks)
        
    def get_feature_vertices(self, spatial_dims, radius):
        ts = []
        for dim in spatial_dims:
            print(dim)
            ts.append(log_fibonacci(dim, radius, 0.05, True, True))
        
        return ts

class ActiveCNN(nn.Module):
    def __init__(self, net, n_fixations):
        super().__init__()
        
        self.net = net
        self.n_fixations = n_fixations
    
    def forward(self, x):
        
        fix_history = []
        
        if self.training:
            fixations = torch.rand(x.shape[0], 2).to(x.device) * 0.3
        
        else:
            fixations = torch.zeros(x.shape[0], 2).to(x.device)
        
        predictions = []
        
        for i in range(0, self.n_fixations):
            fix_history.append(fixations)
            where, what = self.net(x, fixations)
            predictions.append(what)
            fixations = where + fixations
        
        predictions = torch.stack(predictions, dim=1)
        
        predictions = torch.mean(predictions, dim=1)
        
        return predictions


def make_model(n_fixations,
			   n_classes,
			   radius,
			   block_sigma, 
			   block_max_ord, 
			   patch_sigma, 
			   patch_max_ord, 
			   ds_sigma, 
			   ds_max_ord):
	
	dims = [112**2, 28**2, 14**2, 7**2, 7**2]
	width = [40, 80, 160, 320]
	depth = [2, 2, 6, 2]

	net = ConvNext(radius, dims, width, depth, block_sigma, block_max_ord, patch_sigma, patch_max_ord, ds_sigma, ds_max_ord, n_classes)

	return ActiveCNN(net, n_fixations)
	

