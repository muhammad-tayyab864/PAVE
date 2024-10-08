"""
Source:
https://github.com/kornia/kornia/blob/master/kornia/metrics/ssim.py
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.filters import filter2d, get_gaussian_kernel2d
from kornia.filters.filter import _compute_padding

_eps = 1.

def _crop(img: torch.Tensor, cropping_shape: List[int]) -> torch.Tensor:
    """Crop out the part of "valid" convolution area."""
    return torch.nn.functional.pad(
        img, (-cropping_shape[2], -cropping_shape[3], -cropping_shape[0], -cropping_shape[1])
    )

def adjustable_ssim(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int,
    max_val: float = 1.0,
    eps: float = 1e-12,
    padding: str = 'same',
    luminance_factor: float = .25
) -> torch.Tensor:
    r"""Function that computes the Structural Similarity (SSIM) index map between two images.

    Measures the (SSIM) index between each element in the input `x` and target `y`.

    The index can be described as:

    .. math::

      \text{SSIM}(x, y) = \frac{(2\mu_x\mu_y+c_1)(2\sigma_{xy}+c_2)}
      {(\mu_x^2+\mu_y^2+c_1)(\sigma_x^2+\sigma_y^2+c_2)}

    where:
      - :math:`c_1=(k_1 L)^2` and :math:`c_2=(k_2 L)^2` are two variables to
        stabilize the division with weak denominator.
      - :math:`L` is the dynamic range of the pixel-values (typically this is
        :math:`2^{\#\text{bits per pixel}}-1`).

    Args:
        img1: the first input image with shape :math:`(B, C, H, W)`.
        img2: the second input image with shape :math:`(B, C, H, W)`.
        window_size: the size of the gaussian kernel to smooth the images.
        max_val: the dynamic range of the images.
        eps: Small value for numerically stability when dividing.
        padding: ``'same'`` | ``'valid'``. Whether to only use the "valid" convolution
         area to compute SSIM to match the MATLAB implementation of original SSIM paper.
        luminance_factor: the importance of luminance factor.

    Returns:
       The ssim index map with shape :math:`(B, C, H, W)`.

    Examples:
        >>> input1 = torch.rand(1, 4, 5, 5)
        >>> input2 = torch.rand(1, 4, 5, 5)
        >>> ssim_map = ssim(input1, input2, 5)  # 1x4x5x5
    """
    if not isinstance(img1, torch.Tensor):
        raise TypeError(f"Input img1 type is not a torch.Tensor. Got {type(img1)}")

    if not isinstance(img2, torch.Tensor):
        raise TypeError(f"Input img2 type is not a torch.Tensor. Got {type(img2)}")

    if not isinstance(max_val, float):
        raise TypeError(f"Input max_val type is not a float. Got {type(max_val)}")

    if not len(img1.shape) == 4:
        raise ValueError(f"Invalid img1 shape, we expect BxCxHxW. Got: {img1.shape}")

    if not len(img2.shape) == 4:
        raise ValueError(f"Invalid img2 shape, we expect BxCxHxW. Got: {img2.shape}")

    if not img1.shape == img2.shape:
        raise ValueError(f"img1 and img2 shapes must be the same. Got: {img1.shape} and {img2.shape}")

    # prepare kernel
    kernel: torch.Tensor = get_gaussian_kernel2d((window_size, window_size), (1.5, 1.5)).unsqueeze(0)

    # compute coefficients
    C1: float = (0.01 * max_val) ** 2
    C2: float = (0.03 * max_val) ** 2

    # compute local mean per channel
    mu1: torch.Tensor = filter2d(img1, kernel)
    mu2: torch.Tensor = filter2d(img2, kernel)

    cropping_shape: List[int] = []
    if padding == 'valid':
        height, width = kernel.shape[-2:]
        cropping_shape = _compute_padding([height, width])
        mu1 = _crop(mu1, cropping_shape)
        mu2 = _crop(mu2, cropping_shape)
    elif padding == 'same':
        pass

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    mu_img1_sq = filter2d(img1**2, kernel)
    mu_img2_sq = filter2d(img2**2, kernel)
    mu_img1_img2 = filter2d(img1 * img2, kernel)

    if padding == 'valid':
        mu_img1_sq = _crop(mu_img1_sq, cropping_shape)
        mu_img2_sq = _crop(mu_img2_sq, cropping_shape)
        mu_img1_img2 = _crop(mu_img1_img2, cropping_shape)
    elif padding == 'same':
        pass

    # compute local sigma per channel
    sigma1_sq = mu_img1_sq - mu1_sq
    sigma2_sq = mu_img2_sq - mu2_sq
    sigma12 = mu_img1_img2 - mu1_mu2

    # compute the similarity index map
    num: torch.Tensor = ((2.0 * mu1_mu2 + C1) ** luminance_factor) * (2.0 * sigma12 + C2)
    den: torch.Tensor = ((mu1_sq + mu2_sq + C1) ** luminance_factor) * (sigma1_sq + sigma2_sq + C2)

    return num / (den + eps)

def adjustable_ssim_loss(img1, img2, window_size=11, reduction="mean", luminance_factor: float = .25):
    ssim_map: torch.Tensor = adjustable_ssim(img1, img2, window_size=window_size, luminance_factor=luminance_factor)

    # compute and reduce the loss
    loss = torch.clamp((1.0 - ssim_map) / 2, min=0, max=1)

    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)
    elif reduction == "none":
        pass
    return loss

def ssim_map(
    img1: torch.Tensor,
    img2: torch.Tensor,
    window_size: int,
    max_val: float = 1.0,
    eps: float = 1e-12,
    padding: str = 'same',
) -> torch.Tensor:
    r"""Function that computes the Structural Similarity (SSIM) index map between two images.

    Measures the (SSIM) index between each element in the input `x` and target `y`.

    The index can be described as:

    .. math::

      \text{SSIM}(x, y) = \frac{(2\mu_x\mu_y+c_1)(2\sigma_{xy}+c_2)}
      {(\mu_x^2+\mu_y^2+c_1)(\sigma_x^2+\sigma_y^2+c_2)}

    where:
      - :math:`c_1=(k_1 L)^2` and :math:`c_2=(k_2 L)^2` are two variables to
        stabilize the division with weak denominator.
      - :math:`L` is the dynamic range of the pixel-values (typically this is
        :math:`2^{\#\text{bits per pixel}}-1`).

    Args:
        img1: the first input image with shape :math:`(B, C, H, W)`.
        img2: the second input image with shape :math:`(B, C, H, W)`.
        window_size: the size of the gaussian kernel to smooth the images.
        max_val: the dynamic range of the images.
        eps: Small value for numerically stability when dividing.
        padding: ``'same'`` | ``'valid'``. Whether to only use the "valid" convolution
         area to compute SSIM to match the MATLAB implementation of original SSIM paper.

    Returns:
       The ssim index map with shape :math:`(B, C, H, W)`.

    Examples:
        >>> input1 = torch.rand(1, 4, 5, 5)
        >>> input2 = torch.rand(1, 4, 5, 5)
        >>> ssim_map = ssim(input1, input2, 5)  # 1x4x5x5
    """
    if not isinstance(img1, torch.Tensor):
        raise TypeError(f"Input img1 type is not a torch.Tensor. Got {type(img1)}")

    if not isinstance(img2, torch.Tensor):
        raise TypeError(f"Input img2 type is not a torch.Tensor. Got {type(img2)}")

    if not isinstance(max_val, float):
        raise TypeError(f"Input max_val type is not a float. Got {type(max_val)}")

    if not len(img1.shape) == 4:
        raise ValueError(f"Invalid img1 shape, we expect BxCxHxW. Got: {img1.shape}")

    if not len(img2.shape) == 4:
        raise ValueError(f"Invalid img2 shape, we expect BxCxHxW. Got: {img2.shape}")

    if not img1.shape == img2.shape:
        raise ValueError(f"img1 and img2 shapes must be the same. Got: {img1.shape} and {img2.shape}")

    # prepare kernel
    # kernel: torch.Tensor = get_gaussian_kernel2d((window_size, window_size), (1.5, 1.5)).unsqueeze(0)
    kernel: torch.Tensor = get_gaussian_kernel2d((window_size, window_size), (1.5, 1.5))

    # compute coefficients
    C1: float = (0.01 * max_val) ** 2
    C2: float = (0.03 * max_val) ** 2

    # compute local mean per channel
    mu1: torch.Tensor = filter2d(img1, kernel)
    mu2: torch.Tensor = filter2d(img2, kernel)

    cropping_shape: List[int] = []
    if padding == 'valid':
        height, width = kernel.shape[-2:]
        cropping_shape = _compute_padding([height, width])
        mu1 = _crop(mu1, cropping_shape)
        mu2 = _crop(mu2, cropping_shape)
    elif padding == 'same':
        pass

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    mu_img1_sq = filter2d(img1**2, kernel)
    mu_img2_sq = filter2d(img2**2, kernel)
    mu_img1_img2 = filter2d(img1 * img2, kernel)

    if padding == 'valid':
        mu_img1_sq = _crop(mu_img1_sq, cropping_shape)
        mu_img2_sq = _crop(mu_img2_sq, cropping_shape)
        mu_img1_img2 = _crop(mu_img1_img2, cropping_shape)
    elif padding == 'same':
        pass

    # compute local sigma per channel
    sigma1_sq = mu_img1_sq - mu1_sq
    sigma2_sq = mu_img2_sq - mu2_sq
    sigma12 = mu_img1_img2 - mu1_mu2

    # compute the similarity index map
    num: torch.Tensor = (2.0 * mu1_mu2 + C1) * (2.0 * sigma12 + C2)
    den: torch.Tensor = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map: torch.Tensor = num / (den + eps)

    return ssim_map.mean(dim=1).unsqueeze(1)


def pos_similarity_ratio(target_a, target_b, reference):
    cmap_a = F.relu(ssim_map(target_a, reference, window_size=11))
    cmap_b = F.relu(ssim_map(target_b, reference, window_size=11))

    map_diff = torch.log(cmap_a + _eps) - torch.log(cmap_b + _eps)
    map_diff = F.relu(map_diff)
    return blur(map_diff)


def blur(img):
    # kernel = get_gaussian_kernel2d((41, 41), (4.5, 4.5)).unsqueeze(0) # get_gaussian_kernel2d((9, 9), (4.5, 4.5)).unsqueeze(0) # get_gaussian_kernel2d((41, 41), (4.5, 4.5)).unsqueeze(0)
    kernel = get_gaussian_kernel2d((41, 41), (4.5, 4.5))
    b_img = filter2d(img, kernel)
    return b_img
