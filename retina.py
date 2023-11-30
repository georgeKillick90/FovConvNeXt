import torch
import torch.nn as nn
import torch.nn.functional as F
import einops
from .tessellations import *
import matplotlib.pyplot as plt
#from torch_scatter import scatter
import numpy as np
from scipy.spatial import KDTree
from torch import nn
from .utils import *
from .bases import *


class LightRetina(nn.Module):
	def __init__(self, tessellation, padding_mode='zeros'):
		super().__init__()
		tessellation = tessellation
		assert len(tessellation.shape) == 2, "Expected tessellation to be a 2 dimensional tensor"
		assert tessellation.shape[1] == 2, "Expected tessellation to be of shape [N, 2]"
		tess = einops.rearrange(tessellation, 'n c -> 1 1 n c')
		self.tess = nn.Parameter(tess, False)
		self.padding_mode = padding_mode

	def forward(self, x, fixations=None):
		if fixations is not None:
			fixations = einops.rearrange(fixations, 'b c -> b 1 1 c')
			coords = self.tess + fixations
		else:
			coords = einops.repeat(self.tess, '1 1 n c -> (b 1) 1 n c', b=x.shape[0])

		x = F.grid_sample(x, coords, padding_mode=self.padding_mode)
		x = einops.rearrange(x, 'b c h w -> b c (h w)')
		return x