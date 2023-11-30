import torch.nn as nn
import torch
import numpy as np

class GaussianBasis2d(nn.Module):
    def __init__(self, max_ord):
        super().__init__()
        self.max_ord = max_ord
        
    def forward(self, x, y, sigma):
        basis = []
        for x_ord in range(0, self.max_ord+1):
            for y_ord in range(0, self.max_ord+1):
                base = self.derivative2d(x, y, sigma, x_ord, y_ord)
                basis.append(base)
        return torch.stack(basis, dim=0)
                   
    def gaussian1d(self, x, sigma):
        scaling = 1/(np.sqrt(2*np.pi) * sigma)
        exponent = -(x**2)/(2*sigma**2)
        return scaling * torch.e ** exponent
    
    def derivative1d(self, x, sigma, m):
        sign = (-1)**m
        scale = 1/(torch.sqrt(sigma)**m)
        herm = self.hermite(x/(sigma * np.sqrt(2)), m)
        window = self.gaussian1d(x, sigma)
        return sign * scale * herm * window
    
    def derivative2d(self, x, y, sigma, x_ord, y_ord):
        xder = self.derivative1d(x, sigma, x_ord)
        yder = self.derivative1d(y, sigma, y_ord)
        return xder * yder

    def hermite(self, x, ord):
        """ Computes hermite polynomials at x for a given order """
        if ord < 0 or ord > 5:
            raise ValueError("ord must be in the range 0 - 5 inclusive.")
        if ord == 0:
            return x * 0 + 1
        elif ord == 1:
            return 2*x
        elif ord == 2:
            return 4*x**2 - 2
        elif ord == 3:
            return 8*x**3 - 12*x
        elif ord == 4:
            return 16*x**4 - 48*x**2 + 12
        elif ord == 5:
            return 32*x**5 - 160*x**3 + 120*x
