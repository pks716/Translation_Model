# Load Nify
# Create slices based on pre_processing

"""
DO NOT EDIT. CONTROL FROM SLICER PARAMETERS.

All the output scans will be saved in the destination folder as
    patient_name
        ct
            -1.npy
            -2.npy
            ...
        mr
            -1.npy
            -2.npy
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nib
import argparse
from multiprocessing import Pool

from slicer_parameters import view, view_index_hh, destination_directory, data_directory, full_dataset, all_slices, per_sample_normalize
from pre_processing import pre_processing_order

view_key, view_index = False, 4
for key, item in view.items():
    if item:
        view_key = key
        view_index = view_index_hh[view_key]

if not view_key:
    raise ValueError("Select a view from slicer_parameters.py")

path = 'data_original/train'
path2 = 'data_fin/val'


path = data_directory
destination = destination_directory

num_cores = 24


def fetch_scan_paths(parent_foler):
    patients = os.listdir(parent_foler)
    patients = sorted(patients)
    patients_paths = []
    for patient in patients:
        if patient == '.DS_Store':
            continue
        if patient == 'overview':
            continue
        patients_paths.append(f"{parent_foler}/{patient}")
    
    return patients_paths

def read_nifti_to_numpy(file_path):
    try:
        nifti_image = nib.load(file_path)
        numpy_array = nifti_image.get_fdata()
        return numpy_array

    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def populate(scan_path,
            destination_directory):
    # print(scan_path.split('/'))
    patient_name = scan_path.split('/')[-1]

    print("Splitting ", patient_name, "  scan.")
    
    if not os.path.exists(f'{destination_directory}/{patient_name}/mr/processed'):
        os.makedirs(f'{destination_directory}/{patient_name}/mr/processed')
    if not os.path.exists(f'{destination_directory}/{patient_name}/ct/processed'):
        os.makedirs(f'{destination_directory}/{patient_name}/ct/processed')
    # if not os.path.exists(f'{destination_directory}/{patient_name}/mask/processed'):
    #     os.makedirs(f'{destination_directory}/{patient_name}/mask/processed')
        
    if not os.path.exists(f'{destination_directory}/{patient_name}/mr/original'):
        os.makedirs(f'{destination_directory}/{patient_name}/mr/original')
    if not os.path.exists(f'{destination_directory}/{patient_name}/ct/original'):
        os.makedirs(f'{destination_directory}/{patient_name}/ct/original')
    # if not os.path.exists(f'{destination_directory}/{patient_name}/mask/original'):
    #     os.makedirs(f'{destination_directory}/{patient_name}/mask/original')
        
    def read_and_convert(scan_path):
        print("Loading", scan_path)
        mr_vol = read_nifti_to_numpy(f'{scan_path}/mr.nii.gz')
        ct_vol = read_nifti_to_numpy(f'{scan_path }/ct.nii.gz')
        # mask_vol = read_nifti_to_numpy(f'{scan_path }/mask.nii.gz')

        # print(mr_vol, ct_vol)
        if mr_vol.shape != ct_vol.shape:
            print("Volume size mismatch at index--", scan_path)

        "train/{patient_name}/mri.---"
        if view_index > 2:
            raise ValueError("Set view in slicer_parameters as axial, coronal or sagittal")


        mr_max, mr_min = np.max(mr_vol), np.min(mr_vol)
        ct_vol = np.clip(ct_vol, -1024, 3000)
        ct_max, ct_min = np.max(ct_vol), np.min(ct_vol)
            
            
        
        for l in range(mr_vol.shape[view_index]):
            if not all_slices:
                if l < mr_vol.shape[view_index]//2:
                    continue
                if l > (mr_vol.shape[view_index]//2 + 5):
                    break
                
            if view_key == 'axial':                
                mr_slice = mr_vol[:,:,l]
                ct_slice = ct_vol[:,:,l]
                # mask = mask_vol[:,:,l]
            elif view_key == 'coronal':
                mr_slice = mr_vol[:,l,:]
                ct_slice = ct_vol[:,l,:]
                # mask = mask_vol[:,l,:]
            elif view_key == 'saggital':
                mr_slice = mr_vol[l,:,:]
                ct_slice = ct_vol[l,:,:]
                # mask = mask_vol[l,:,:]
            else:
                raise ValueError("Set view in slicer_parameters as axial, coronal or saggital")
            
            if per_sample_normalize:
                mr_slice = (mr_slice-mr_min)/(mr_max-mr_min)
                ct_slice = (ct_slice-ct_min)/(ct_max-ct_min)
            
            # ct_slice_processed, mr_slice_processed, mask_processed = np.copy(ct_slice), np.copy(mr_slice), np.copy(mask)
            ct_slice_processed, mr_slice_processed = np.copy(ct_slice), np.copy(mr_slice)

            # This is for extra pre-processing steps on training data. The original data should remain untouched.
            for method, target in pre_processing_order:
                if 'ct' in target:
                    ct_slice_processed = method(ct_slice_processed)
                if 'mri' in target:
                    mr_slice_processed = method(mr_slice_processed)
                # if 'mask' in target:
                #     mask_processed = method(mask_processed)
                # raise ValueError("Incorrect Key in pre_processing_methods. Expected ordered pair as (function_name, [targets..]). targer can me 'mri and/or 'ct")
            np.savez_compressed(f'{destination_directory}/{patient_name}/mr/processed/{patient_name}_processed_{view_key}_{l}.npz', mr_slice_processed)
            np.savez_compressed(f'{destination_directory}/{patient_name}/ct/processed/{patient_name}_processed_{view_key}_{l}.npz', ct_slice_processed)
            # np.savez_compressed(f'{destination_directory}/{patient_name}/mask/processed/{patient_name}_processed_{view_key}_{l}.npz', mask_processed)
                
            np.savez_compressed(f'{destination_directory}/{patient_name}/mr/original/{patient_name}_original_{view_key}_{l}.npz', mr_slice)
            np.savez_compressed(f'{destination_directory}/{patient_name}/ct/original/{patient_name}_original_{view_key}_{l}.npz', ct_slice)
            # np.savez_compressed(f'{destination_directory}/{patient_name}/mask/original/{patient_name}_original_{view_key}_{l}.npz', mask)
            print("Saved to", f"{destination_directory}/{patient_name}")

            
            

    read_and_convert(scan_path)

mp_args = []

for paths in fetch_scan_paths(path):
    mp_args.append((paths,destination))

if __name__ == "__main__":
    if  not full_dataset:
        mp_args = mp_args[:10]
    with Pool(num_cores) as pool_fn:
        pool_fn.starmap(populate, mp_args)