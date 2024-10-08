import torch 
import torch.nn as nn 
import torch.nn.functional as F

_eps = 1e-7

def ones_pad(x):
    _pad = torch.ones_like(x[:, :1, :, :])
    x = torch.cat([_pad, x], dim=1)
    return x

def power_attention(x, R):
    # power attention processing
    x_m = torch.cat([x.mean(dim=(2,3)), x.std(dim=(2,3))], dim=1)
    x_m = x_m.view(x_m.shape[0], x_m.shape[1], 1, 1)
    x_R = torch.ones_like(x_m)[:,0,:,:].unsqueeze(1) * R.item() 
    x_m = torch.cat([x_R, x_m], dim=1)
    
    return x_m

class MLP(nn.Module):
    """
    A simple channel-wise attention implementation.
    """
    def __init__(self, i_channels : int = 3, h_channels : int = 16, o_channels : int = 3):
        super(MLP, self).__init__()
        self.att_a = nn.Conv2d(i_channels, h_channels, 1)
        self.att_b = nn.Conv2d(h_channels, h_channels, 1)
        self.att_c = nn.Conv2d(h_channels, o_channels, 1)
        
    def forward(self, x):
        view_shape = x.shape[0], 1, 1, 1
        x = self.att_a(x)
        x = (x - x.mean(dim=(1,2,3)).view(view_shape)) / (x.std(dim=(1,2,3)).view(view_shape) + _eps)
        x = F.leaky_relu(x)

        x = self.att_b(x)
        x = (x - x.mean(dim=(1,2,3)).view(view_shape)) / (x.std(dim=(1,2,3)).view(view_shape) + _eps)
        x = F.leaky_relu(x)

        x = self.att_c(x)

        return x