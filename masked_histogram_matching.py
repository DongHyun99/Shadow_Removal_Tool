#-*- coding: utf-8 -*-

import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def match_cumulative_distribution_function(source, template, mask, channel=1):
    """mask 부분의 바깥과 안쪽에 대한 누적분포함수를 비교하여 적용하는 함수

    Args:
        source (np.array): mask 안쪽 부분에 대한 영상
        template (np.array): mask 바깥쪽 부분에 대한 영상
        mask (np.array): mask 이미지
        channel (int): 이미지의 채널수, 흑백이아닌 컬러 이미지도 channel이 1여도 상관없음. Defaults to 1.

    Returns:
        np.array : 매칭이 완료된 이미지
    """
    
    _, src_unique_indices, src_counts = np.unique(source.ravel(), return_inverse=True, return_counts=True)
    tmpl_values, tmpl_counts = np.unique(template.ravel(), return_counts=True)
    
    src_counts[0] = src_counts[0] - len(mask[mask==0]) * channel
    tmpl_counts[0] = tmpl_counts[0] - len(mask[mask!=0]) * channel
    
    source_size = source.size - len(mask[mask==0]) * channel
    template_size = template.size - len(mask[mask!=0]) * channel

    src_quantiles = np.cumsum(src_counts) / source_size
    tmpl_quantiles = np.cumsum(tmpl_counts) / template_size
    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
    return interp_a_values[src_unique_indices].reshape(source.shape)


def masked_histogram_matching(image, reference, mask, multichannel=False):
    """match_cumulative_distribution_function을 사용하여 masked histogram matching을 함. channel에 따라 다르게 작용.

    Args:
        image (np.array)): mask 안쪽 부분에 대한 영상
        reference (np.array): mask 바깥쪽 부분에 대한 영상
        mask (np.array): mask 이미지
        multichannel (bool): 이미지의 채널수를 어떻게 할 것인지를 입력받음. Defaults to False.

    Raises:
        ValueError: image와 reference 사이의 채널 수가 같지 않다면 ValueError를 띄움.

    Returns:
        np.array : 매칭이 완료된 이미지
    """
    
    if image.ndim != reference.ndim:
        raise ValueError('Image and reference must have the same number'
                         'of channels.')

    matched = np.empty(image.shape, dtype=image.dtype)
    
    if multichannel:
        for channel in range(image.shape[-1]):
            matched_channel = match_cumulative_distribution_function(image[..., channel], reference[..., channel], mask)
            matched[..., channel] = matched_channel
    
    else:
        matched = match_cumulative_distribution_function(image, reference, mask, channel=3)
    
    return matched


img = cv2.imread(os.path.join('save', 'Masked-Histogram Maching result', '2_2.png'))
mask = cv2.imread(os.path.join('save', 'Masked-Histogram Maching result', '2_mask.png'), cv2.IMREAD_GRAYSCALE)

if img is not None and mask is not None:
    print('img & mask True')
else:
    print('img or mask is None')

mask[mask>0] = 255
mask = np.expand_dims(mask, axis=2)

masked = mask // 255 * img
none_masked = (255-mask) // 255 * img

mhm = masked_histogram_matching(masked, none_masked, mask)
img = mask // 255 * mhm + none_masked

cv2.imwrite('post-processing.png', img)