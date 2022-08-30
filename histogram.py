import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from skimage.exposure import histogram_matching

def _match_cumulative_cdf(source, template, mask, channel=1):
    
    _, src_unique_indices, src_counts = np.unique(source.ravel(), return_inverse=True, return_counts=True)
    tmpl_values, tmpl_counts = np.unique(template.ravel(), return_counts=True)
    
    src_counts[0] = src_counts[0] - len(mask[mask==0]) * channel
    tmpl_counts[0] = tmpl_counts[0] - len(mask[mask!=0]) * channel
    
    source_size = source.size - len(mask[mask==0]) * channel
    template_size = template.size - len(mask[mask!=0]) * channel
    
    # calculate normalized quantiles for each array
    src_quantiles = np.cumsum(src_counts) / source_size
    tmpl_quantiles = np.cumsum(tmpl_counts) / template_size
    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)
    return interp_a_values[src_unique_indices].reshape(source.shape), src_quantiles, tmpl_quantiles


def masked_histogram_matching(image, reference, mask, multichannel=False):
    
    if image.ndim != reference.ndim:
        raise ValueError('Image and reference must have the same number '
                         'of channels.')


    matched = np.empty(image.shape, dtype=image.dtype)
    
    if multichannel:
        for channel in range(image.shape[-1]):
            matched_channel, src_quantiles, tmpl_quantiles = _match_cumulative_cdf(image[..., channel],
                                                    reference[..., channel], mask)
            y1_d = np.r_[0, np.diff(src_quantiles)]/0.1
            y2_d = np.r_[0, np.diff(tmpl_quantiles)]/0.1
            plt.plot(src_quantiles)
            plt.plot(tmpl_quantiles)
            plt.plot(y1_d)
            plt.plot(y2_d)
            plt.legend(['masked', 'none_masked', 'masked_diff', 'none_masked_diff'])
            plt.savefig(f'img{channel}.png')
            plt.clf()
            src_zero = [0 for x in range(0, 256-src_quantiles.size)]
            tmpl_zero = [0 for x in range(0, 256-tmpl_quantiles.size)]

            src_value = np.concatenate((src_quantiles, src_zero))
            tmpl_value = np.concatenate((tmpl_quantiles, tmpl_zero))
            print(f'value: {np.sum(np.abs(src_value - tmpl_value)) / 256}')

            matched[..., channel] = matched_channel
    
    else:
        matched, src_quantiles, tmpl_quantiles = _match_cumulative_cdf(image, reference, mask, channel=3)
        y1_d = np.r_[0, np.diff(src_quantiles)]/0.1
        y2_d = np.r_[0, np.diff(tmpl_quantiles)]/0.1
        plt.plot(src_quantiles)
        plt.plot(tmpl_quantiles)
        plt.plot(y1_d)
        plt.plot(y2_d)
        plt.legend(['masked', 'none_masked', 'masked_diff', 'none_masked_diff'])
        plt.savefig('img.png')
        plt.clf()
        src_zero = [0 for x in range(0, 256-src_quantiles.size)]
        tmpl_zero = [0 for x in range(0, 256-tmpl_quantiles.size)]

        src_value = np.concatenate((src_quantiles, src_zero))
        tmpl_value = np.concatenate((tmpl_quantiles, tmpl_zero))
        print(f'value: {np.sum(np.abs(src_value - tmpl_value)) / 256}')

    
    return matched

img = cv2.imread(os.path.join('save', 'Masked-Histogram Maching result', '3_2.png'))
mask = cv2.imread(os.path.join('save', 'Masked-Histogram Maching result', '3_mask.png'), cv2.IMREAD_GRAYSCALE)

if img is not None and mask is not None:
    print('img & mask True')
else:
    print('img or mask is None')

mask[mask>0] = 255
mask = np.expand_dims(mask, axis=2)

masked = mask // 255 * img
none_masked = (255-mask) // 255 * img

masked = masked_histogram_matching(masked, none_masked, mask, multichannel=False)
img = mask // 255 * masked + none_masked
cv2.imwrite('post-processing.png', img)