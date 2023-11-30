import torch
import matplotlib.pyplot as plt
import numpy as np
from skimage.data import astronaut
from torchvision.transforms.functional import resize

def cart2pol(points):
	x = points[0]
	y = points[1]
	r = torch.sqrt(x**2 + y**2)
	theta = torch.atan2(y, x)
	return torch.stack((r, theta), dim=0)

def pol2cart(points):
	r = points[0]
	theta = points[1]
	x = r * torch.cos(theta)
	y = r * torch.sin(theta)
	return torch.stack((x, y), dim=0)

def scatter_2d(points, figsize=(7,7), s=5, c=None):
	plt.figure(figsize=figsize)
	plt.scatter(points[0], points[1], s=s, c=c)
	plt.show()
	return

def example_img(out_size=(224,224)):
	img = torch.tensor(astronaut() / 255., dtype=torch.float32)
	img = img.permute(2,0,1).unsqueeze(0)
	img = resize(img, out_size)
	return img

def make_grid(H, W):
	x = torch.linspace(-1, 1, W)
	y = torch.linspace(-1, 1, H)
	x,y = torch.meshgrid(x, y, indexing='xy')
	return torch.stack((x,y), dim=0)

def make_polar_grid(R, S):
	grid = torch.tensor(np.mgrid[1:R+1, 1:S+1], dtype=torch.float32)
	grid[1] *= (1/(S)) * np.pi * 2
	return grid 