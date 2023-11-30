import torch
import torch.nn as nn
from .bases import *
from scipy.spatial import KDTree
from .tessellations import *
from einops import rearrange, repeat
#import opt_einsum as oe
import torch.nn.functional as F
import numpy as np


### Non Conv Layers ###

class Centroid1d(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, coords):
        out = torch.sum(x * coords[None,:])
        out = out / torch.sum(x, dim=-1)
        return out
    
class ExpCentroid1d(nn.Module):
    def __init__(self, coords, temperature):
        super().__init__()
        self.coords = nn.Parameter(coords[None,:], False)
        self.temp = temperature
    
    def forward(self, logits):
        out = torch.softmax(logits / self.temp, dim=-1) * self.coords
        out = torch.sum(out, dim=-1)
        return out

class MaxPool(nn.Module):
    def __init__(self, in_locs, out_locs, k):
        super(). __init__()
        dists, idx = self.knn(in_locs, out_locs, k)
        self.register_buffer('idx', idx)

    def forward(self, x):
        x = x[:,:,self.idx]
        x = torch.max(x, dim=-1)[0]
        return x

    def knn(self, points, query_points, k):
        tree = KDTree(points)
        dists, idx = tree.query(query_points, k)
        dists = torch.tensor(dists, dtype=torch.float32)
        idx = torch.tensor(idx, dtype=torch.long)
        return dists, idx

class AvgPool(nn.Module):
    def __init__(self, in_locs, out_locs, k):
        super(). __init__()
        dists, idx = self.knn(in_locs, out_locs, k)
        self.register_buffer('idx', idx)

    def forward(self, x):
        x = x[:,:,self.idx]
        x = torch.mean(x, dim=-1)
        return x

    def knn(self, points, query_points, k):
        tree = KDTree(points)
        dists, idx = tree.query(query_points, k)
        dists = torch.tensor(dists, dtype=torch.float32)
        idx = torch.tensor(idx, dtype=torch.long)
        return dists, idx

### Conv Layers ###


class GConv(nn.Module):
    
    def __init__(self, l1, l2, k, sigma, max_ord):
        
        super().__init__()
        self.k = k
        self.l1 = l1
        self.l2 = l2
        self.sigma = torch.tensor([sigma]).float()
        
        self.basis = GaussianBasis2d(max_ord)
        patch_coords, idx = self.compute_support()
        filter_weights = self.compute_filter_weights(patch_coords)
        
        self.n_filters = filter_weights.shape[0]
        
        ### Parameters and Buffers
        self.filter_weights = nn.Parameter(filter_weights, False)
        self.register_buffer('idx', idx)
        
        self.patches = patch_coords
        
    def vis_patches(self):
        ### Helper method to visualize delta offsets for all patches
        patches = self.patches.clone().detach()
        plt.figure(figsize=(10,10))
        plt.scatter(patches[...,0], patches[...,1])
        plt.show()
    
    def forward(self, x):
        x = x[:,:,self.idx]
        x = torch.einsum('bcnk,fnk->bcfn', x, self.filter_weights)#.contiguous()
        return x
        
    def compute_support(self):

        ### KNN ###
        tree = KDTree(self.l1)
        dists, idx = tree.query(self.l2, k=self.k)
        dists = torch.tensor(dists, dtype=torch.float32)
        idx = torch.tensor(idx, dtype=torch.long)
        
        ### Extracting and Normalizing Patches
        patches = self.l1[idx]

        patches = patches - self.l2[:,None,:]
        
        ### 1:7 for stability 
        patches = patches / torch.mean(dists[:,1:7], dim=-1)[:,None,None]
        
        return patches, idx
    
    def compute_filter_weights(self, patch_coords):
        x, y = patch_coords[...,0], patch_coords[...,1]
        weights = self.basis(x, y, self.sigma)
        ### Filter Normalization
        # mean centre AC Filters
        weights[1:] = weights[1:] - torch.mean(weights[1:], dim=-1, keepdims=True)
        
        # l2 normalize
        weights[1:] = weights[1:] / torch.sqrt(torch.sum(weights[1:]**2, dim=-1, keepdims=True))
        
        # l1 normalize dc filter i.e. gaussian filter
        weights[0] = weights[0] / torch.sum(weights[0], dim=-1, keepdims=True)
        return weights

class DWise(GConv):
    
    def __init__(self, in_channels, l1, l2, k, sigma, max_ord, use_bias=True):
        
        super().__init__(l1, l2, k, sigma, max_ord)

        self.conv1x1 = nn.Conv1d(self.n_filters * in_channels,
                                  in_channels,
                                  1,
                                  groups=in_channels,
                                  bias=use_bias)

        weights = self.conv1x1.weight.clone().detach()
        self.weights = nn.Parameter(weights, True)


    def forward(self, x):
        x = x[:,:,self.idx]
        filters = torch.einsum('fnk,cf1->cnk', self.filter_weights, self.weights)
        x = torch.einsum('bcnk,cnk->bcn', x, filters)

        return x

class Conv(GConv):
    def __init__(self, in_channels, out_channels, l1, l2, k, sigma, max_ord, use_bias=True):
        super().__init__(l1, l2, k, sigma, max_ord)
        self.conv1x1 = nn.Conv1d(self.n_filters * in_channels , out_channels, 1, bias=use_bias)

    def forward(self, x):
        x = x[:,:,self.idx]
        x = torch.einsum('bcnk,fnk->bfcn', x, self.filter_weights)
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        x = self.conv1x1(x)

        return x

### These layers are the same as above but implement convolution without extracting patches
### i.e. each filter is the size of the image - in practice this is a lot quicker than the patch
### method because the indexing method plus creation of the new tensor is the bottleneck
### in the previous layers (e.g. Line 167 "x = x[:,:,self.idx]")
### For the paper - we report flops on the patch method, in the future we will be optimizing the
### patch convolution methods and the following convolutions will be deprecated

class GConvV2(nn.Module):
    
    def __init__(self, l1, l2, sigma, max_ord):
        
        super().__init__()

        self.l1 = l1
        self.l2 = l2
        self.sigma = torch.tensor([sigma]).float()
        
        self.basis = GaussianBasis2d(max_ord)
        self.weights = nn.Parameter(self.compute_filter_weights(l2, l2).contiguous(), False)
                
    def forward(self, x):
        identity = x
        x = torch.einsum('bcm,fmn->bcfn', x, self.weights)#.contiguous()
        return x
    
    def compute_filter_weights(self, l1, l2):
        support = l1.unsqueeze(0) - l2.unsqueeze(1)
        
        x, y = support.permute(2, 0, 1)
        
        r = torch.sqrt(x**2 + y**2)
        
        local_support = torch.sort(r, dim=1)[0][:,1:6]
        local_support = torch.mean(local_support, dim=-1)[None, :]
        x = x / local_support
        y = y / local_support

        weights = self.basis(x, y, self.sigma)
        
        ### Filter Normalization
        # mean centre AC Filters
        weights[1:] = weights[1:] - torch.mean(weights[1:], dim=-2, keepdims=True)
        
        # l2 normalize
        weights[1:] = weights[1:] / torch.sqrt(torch.sum(weights[1:]**2, dim=-2, keepdims=True))
        
        # l1 normalize dc filter i.e. gaussian filter
        weights[0] = weights[0] / torch.sum(weights[0], dim=-2, keepdims=True)
        return weights
    
class ConvV2(GConvV2):
    
    def __init__(self, in_channels, out_channels, l1, l2, sigma, max_ord, groups=1):
        
        super().__init__(l1, l2, sigma, max_ord)

        self.conv = nn.Conv1d(in_channels * ((max_ord+1)**2),
                              out_channels,
                              1,
                              groups=groups)

        nn.init.trunc_normal_(self.conv.weight, std=0.02)
        
    def forward(self, x):
        x = torch.einsum('bcm,fnm->bcfn', x, self.weights)#.contiguous()
        x = x.reshape(x.shape[0], -1, x.shape[-1])
        x = self.conv(x)
        return x