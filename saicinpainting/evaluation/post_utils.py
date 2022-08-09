import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
    
def _match_cumulative_cdf(source, template, mask):
    src_values, src_unique_indices, src_counts = np.unique(source.ravel(),
                                                           return_inverse=True,
                                                           return_counts=True)
    tmpl_values, tmpl_counts = np.unique(template.ravel(), return_counts=True)
    
    
    src_counts[0] = src_counts[0] - len(mask[mask==0]) * 3
    tmpl_counts[0] = tmpl_counts[0] - len(mask[mask!=0]) * 3
    
    source_size = source.size - len(mask[mask==0]) * 3
    template_size = template.size - len(mask[mask!=0]) * 3
    
    # calculate normalized quantiles for each array
    src_quantiles = np.cumsum(src_counts) / source_size
    tmpl_quantiles = np.cumsum(tmpl_counts) / template_size

    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)

    return interp_a_values[src_unique_indices].reshape(source.shape)

def _match_cumulative_cdf_mul(source, template, mask):
    src_values, src_unique_indices, src_counts = np.unique(source.ravel(),
                                                           return_inverse=True,
                                                           return_counts=True)
    tmpl_values, tmpl_counts = np.unique(template.ravel(), return_counts=True)
    
    
    src_counts[0] = src_counts[0] - len(mask[mask==0])
    tmpl_counts[0] = tmpl_counts[0] - len(mask[mask!=0])
    
    source_size = source.size - len(mask[mask==0])
    template_size = template.size - len(mask[mask!=0])
    
    # calculate normalized quantiles for each array
    src_quantiles = np.cumsum(src_counts) / source_size
    tmpl_quantiles = np.cumsum(tmpl_counts) / template_size

    interp_a_values = np.interp(src_quantiles, tmpl_quantiles, tmpl_values)

    return interp_a_values[src_unique_indices].reshape(source.shape)

def match_histograms(image, reference, mask, *, multichannel=False):
    if image.ndim != reference.ndim:
        raise ValueError('Image and reference must have the same number '
                         'of channels.')

    if multichannel:
        if image.shape[-1] != reference.shape[-1]:
            raise ValueError('Number of channels in the input image and '
                             'reference image must match!')

        matched = np.empty(image.shape, dtype=image.dtype)
        for channel in range(image.shape[-1]):
            matched_channel = _match_cumulative_cdf_mul(image[..., channel],
                                                    reference[..., channel], mask)
            matched[..., channel] = matched_channel
    else:
        matched = _match_cumulative_cdf(image, reference, mask)

    return matched