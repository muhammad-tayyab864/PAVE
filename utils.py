import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from variable_luminance import rgb_to_ycbcr, ycbcr_to_rgb
from kornia.filters.filter import filter2d
from kornia.filters.kernels import get_gaussian_kernel2d

_eps = 1e-7
_eps_log = 1.

def decompose_imgs(imgs, luminance_const = [0.206, 0.339, 0.454]):
    """
    RGB to YCbCr, returns luminance and chrominance.
    """
    _ycbcr = rgb_to_ycbcr(imgs, luminance_const=luminance_const)
    _y = _ycbcr[:,:1]
    _cbcr = _ycbcr[:,1:]
    return _y, _cbcr

def compose_imgs(y_imgs, cbcr_imgs, luminance_const = [0.206, 0.339, 0.454]):
    """
    YCbCr to RGB, returns RGB.
    """
    _ycbcr = torch.cat((y_imgs, cbcr_imgs), dim=1)
    rgb_imgs = ycbcr_to_rgb(_ycbcr, luminance_const=luminance_const)
    return rgb_imgs

def R_imgs(refs, imgs, gamma = 1.):
    """
    Power reduction (R) computed on images.
    """
    div_factor = (imgs.abs() ** gamma).sum(dim=(2,3)) / ((refs.abs() ** gamma).sum(dim=(2,3)) + _eps)
    return (1. - div_factor)

def R_sclr(power_k, gamma = 1.):
    """
    Power reduction (R) computed on scalar value (k).
    """
    return 1. - (power_k ** gamma)

def contrast_loss_G(output, target, R, log_ratio=True):
    """
    Global contrast loss.
    """
    w_std = F.relu(1. - 2 * R)
    _c0 = torch.abs(target.mean(dim=(2,3)) - output.mean(dim=(2,3)))
    if log_ratio:
        _c1 = -1 * (1. - w_std) * torch.log((output.var(dim=(2,3)) + 7e-2) / (target.var(dim=(2,3)) + 7e-2))
    else:
        _c1 = -1 * (1. - w_std) * (output.var(dim=(2,3)) / (target.var(dim=(2,3)) + 1e-1))
    _c2 = w_std * torch.abs(target.var(dim=(2,3)) - output.var(dim=(2,3)))
    _c_loss = _c0 + _c1 + _c2

    return _c_loss 

def contrast_loss_L(output, target, R, window_size=(11,11), log_ratio=True):
    """Adapted from https://kornia.readthedocs.io/en/latest/_modules/kornia/metrics/ssim.html#ssim"""
    """
    Local contrast loss.
    """
    
    # prepare kernel
    kernel = get_gaussian_kernel2d((window_size[0], window_size[1]), (1.5, 1.5))

    # compute local mean per channel
    output_mu = filter2d(output, kernel=kernel)
    target_mu = filter2d(target, kernel=kernel)

    output_mu_sq = output_mu ** 2
    target_mu_sq = target_mu ** 2

    # compute local sigma per channel
    # use ReLU to avoid numerical error - variance equal to zero
    output_sigma_sq = F.relu(filter2d(output ** 2, kernel=kernel) - output_mu_sq)
    target_sigma_sq = F.relu(filter2d(target ** 2, kernel=kernel) - target_mu_sq)

    # weighting factor
    w_std = F.relu(1. - 2 * R).view(-1,1,1,1)

    # compute contrast loss 
    _c0 = torch.abs(output_mu - target_mu)
    if log_ratio:
        _c1 = -1 * (1 - w_std) * torch.log((output_sigma_sq + 7e-2) / (target_sigma_sq + 7e-2))
    else:
        _c1 = -1 * (1 - w_std) * (output_sigma_sq / (target_sigma_sq + 1e-1))
    _c2 =  w_std * torch.abs(target_sigma_sq - output_sigma_sq)
    _c_loss = _c0 + _c1 + _c2

    return _c_loss.mean(dim=(2,3))

def instance_norm(x):
    """
    Instance normalization.
    """
    view_shape = x.shape[0], x.shape[1], 1, 1
    return (x - x.mean(dim=(2,3)).view(view_shape)) / (x.std(dim=(2,3)).view(view_shape) + _eps)

def EME(img, kernel_size=11, padding=5):
    """
    Measure of enhancement
    """
    assert(torch.sum(img < 0) == 0)
    _max_pool = F.max_pool2d(img, kernel_size=kernel_size, padding=padding, stride=kernel_size)
    _min_pool = F.max_pool2d(-img, kernel_size=kernel_size, padding=padding, stride=kernel_size) * -1
    _eme = 20 * (torch.log10((_max_pool + 1e-2) / (_min_pool + 1e-2)))
    return _eme.mean()

def save_img(img, save_name, normalize=False):
    """
    Save numpy image.
    """
    if normalize:
        print(img.max(), img.min(), img.mean())
        range = img.max(axis=(0,1)) - img.min(axis=(0,1))
        img = (img - img.min(axis=(0,1))) / (range + 1e-7)
    img = (img * 255).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_name, img)

def histogram_img(img, title=None):
    """
    https://stackoverflow.com/questions/55659784/plot-multiple-rgb-images-and-histogram-side-by-side-in-a-grid
    """
    plt.figure(figsize=(16,6))
    ax1 = plt.subplot(1, 2, 1)
    ax2 = plt.subplot(1, 2, 2)

    colors = ('b','g','r')
    histograms = []

    for i in range(3):
        hist = cv2.calcHist([img], [i], None, [256], [1,255])
        histograms.append(hist)
        ax1.plot(hist, color=colors[i])

    # tmp_img = cv2.bitwise_and(img, img, mask=mask)
    # ax2.imshow(cv2.cvtColor(tmp_img,cv2.COLOR_BGR2RGB))
    ax2.imshow(img)
    ax2.grid(False)
    ax2.axis('off')    

    if title is not None:
        plt.title =  title

    plt.show()

def hpf(x):
    """ 
    High-pass filter.
    """
    hpf_kernel = (torch.tensor([[[[-1,-2,-1],[-2,12,-2],[-1,-2,-1]]]])) * (1./9) # (torch.tensor([[[[-1,-1,-1],[-1,8,-1],[-1,-1,-1]]]])) * (1./9) # (torch.tensor([[[[-1,-2,-1],[-2,12,-2],[-1,-2,-1]]]])) * (1./9)
    hpf_kernel = hpf_kernel.cuda() if x.is_cuda else hpf_kernel
    x = F.conv2d(x, hpf_kernel, padding=1)
    return x