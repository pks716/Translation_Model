import cv2  
import numpy as np
from slicer_parameters import slice_size, interpolation_method #, padding_size


"""
Define Preprocessing steps here. Write numpy slice funtions. Expected function type:

def f_name(slice):
    BOC
    return processed_slice
"""
"""
Updated preprocessing to:
1. Pad to 608x608 (preserves anatomy aspect ratio)
2. Resize to 256x256 (for model compatibility)
"""

def place_holder_fn(ele):
    # Apply this logic on ct and mri
    return ele

def resize(slice):
    return cv2.resize(slice, slice_size, interpolation=interpolation_method)

# def pad_to_square(slice, target_size=padding_size):
#     """
#     Pad slice to target_size x target_size, preserving aspect ratio
#     """
#     h, w = slice.shape
    
#     # Calculate padding needed
#     pad_h = (target_size - h) // 2
#     pad_w = (target_size - w) // 2
    
#     # Pad with reflection for better boundary handling
#     padded = np.pad(slice, 
#                    ((pad_h, target_size - h - pad_h), 
#                     (pad_w, target_size - w - pad_w)), 
#                    mode='reflect')
    
#     return padded


# pre_processing_order = [(pad_to_square, ['ct','mri','mask']),
pre_processing_order= [ (resize,['ct','mri']), (place_holder_fn, ['ct', 'mri']) ]

# pre_processing_order.append((rescale_to_float16,['ct', 'mri', 'mask']))

for method, targets in pre_processing_order:
    for ele in targets:
        if ele not in ['ct', 'mri', 'mask']:
            raise ValueError('Incorrect modalities specified in pre_processing_order, pre_processing.py')