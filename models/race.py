import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3,3), stride=1, padding=1, padding_mode='reflect', dilation=1, bias=True, use_act=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, padding_mode=padding_mode, dilation=dilation, bias=bias)
        self.use_act = use_act

    def forward(self, x):
        x = self.conv(x)
        if self.use_act:
            x = F.leaky_relu(x)
        return x

class ACE(nn.Module):
    
    def __init__(self):
        super(ACE, self).__init__()
        scale_factor = 2
        self.M2D = nn.MaxPool2d((scale_factor, scale_factor)) 
        self.L1 = ConvAct(2, 32, kernel_size=(3,3), stride=1, padding=1, bias=True)
        self.L2 = ConvAct(33, 32, kernel_size=(3,3), stride=1, padding=4, padding_mode='reflect', dilation=4, bias=True)
        self.L3 = ConvAct(33, 32, kernel_size=(3,3), stride=1, padding=8, padding_mode='reflect', dilation=8, bias=True)
        self.L4 = ConvAct(33, 32, kernel_size=(3,3), stride=1, padding=16, padding_mode='reflect', dilation=16, bias=True)
        
        self.L5 = nn.PixelShuffle(scale_factor)
        self.L6a = ConvAct(8, 32, kernel_size=(3,3), stride=1, padding=1, padding_mode='reflect', bias=True)
        self.L6b = ConvAct(32, 1, kernel_size=(1,1), stride=1, padding=0, bias=True, use_act=False)
        
    def forward(self, x, R = 0.81):
        # First, check if the size is divisible by 2. If not, resize.
        orig_size = x.shape[2], x.shape[3]
        is_resized = False
        if orig_size[0] % 2 != 0 or orig_size[1] % 2 != 0:
            is_resized = True
            x = F.interpolate(x, size=(int((orig_size[0] / 2) * 2), int((orig_size[1] / 2) * 2)))

        R_channel_s1 = torch.unsqueeze(x[:,0,:,:].detach().clone(), 1) * 0. + R
        x = torch.cat([x, R_channel_s1], dim=1)
        x = self.L1(x)
        x = self.M2D(x)
        
        R_channel_s2 = torch.unsqueeze(x[:,0,:,:].detach().clone(), 1) * 0. + R
        x = torch.cat([x, R_channel_s2], dim=1)
        x = self.L2(x)
        
        x = torch.cat([x, R_channel_s2], dim=1)
        x = self.L3(x)
        
        x = torch.cat([x, R_channel_s2], dim=1)
        x = self.L4(x)
        
        x = self.L5(x)
        
        x = self.L6a(x)
        x = self.L6b(x)

        # To original size
        if is_resized:
            x = F.interpolate(x, size=orig_size)

        return torch.clip(x, min=0., max=1.)

class RACE(nn.Module):
    
    def __init__(self):
        super(RACE, self).__init__()
        scale_factor = 2
        self.M2D = nn.MaxPool2d((scale_factor, scale_factor)) 
        self.L1 = ConvAct(2, 32, kernel_size=(3,3), stride=1, padding=1, bias=True)
        self.L2 = ConvAct(33, 32, kernel_size=(3,3), stride=1, padding=4, padding_mode='reflect', dilation=4, bias=True)
        self.L3 = ConvAct(33, 32, kernel_size=(3,3), stride=1, padding=8, padding_mode='reflect', dilation=8, bias=True)
        self.L4 = ConvAct(33, 32, kernel_size=(3,3), stride=1, padding=16, padding_mode='reflect', dilation=16, bias=True)
        
        self.L5 = nn.PixelShuffle(scale_factor)
        self.L6a = ConvAct(8, 32, kernel_size=(3,3), stride=1, padding=1, padding_mode='reflect', bias=True)
        self.L6b = ConvAct(32, 1, kernel_size=(1,1), stride=1, padding=0, bias=True, use_act=False)
        
    def forward(self, x, R = 0.81):
        # First, check if the size is divisible by 2. If not, resize.
        orig_size = x.shape[2], x.shape[3]
        is_resized = False
        if orig_size[0] % 2 != 0 or orig_size[1] % 2 != 0:
            is_resized = True
            x = F.interpolate(x, size=(int((orig_size[0] / 2) * 2), int((orig_size[1] / 2) * 2)))

        x_inp = x.clone()
        R_channel_s1 = torch.unsqueeze(x[:,0,:,:].detach().clone(), 1) * 0. + R
        x = torch.cat([x, R_channel_s1], dim=1)
        x = self.L1(x)
        x = self.M2D(x)
        
        R_channel_s2 = torch.unsqueeze(x[:,0,:,:].detach().clone(), 1) * 0. + R
        x = torch.cat([x, R_channel_s2], dim=1)
        x = self.L2(x)
        
        x = torch.cat([x, R_channel_s2], dim=1)
        x = self.L3(x)
        
        x = torch.cat([x, R_channel_s2], dim=1)
        x = self.L4(x)
        
        x = self.L5(x)
        
        x = self.L6a(x)
        x = self.L6b(x)

        # To original size
        if is_resized:
            x = F.interpolate(x, size=orig_size)

        x = x + x_inp
        
        return torch.clip(x, min=0., max=1.)        