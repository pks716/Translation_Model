# import data loader

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import os
import pandas as pd
from collections import deque
import pickle
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import pandas as pd
from training_hyperparameters import *
from splits_only_test import SPLITS
from train_dataloader import train_dataset
from test_dataloader import test_dataset
from ESAU_net import ESAU
from PTN_model2D import PTN_local
from residual_transformers import ResViT
from transformer_configs import get_resvit_b16_config, get_resvit_l16_config
from swin_unet_v2 import SwinTransformerSys
from swinunet import SwinUNet
from CGNet_arch import CascadedGaze
from unet import UNet
from training_helpers import post_processing_order
from tqdm import tqdm
from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure
from torchmetrics.image.psnr import PeakSignalNoiseRatio
from torchvision.utils import make_grid, save_image
from torchvision.models.feature_extraction import create_feature_extractor
import torchvision
import json
from resvit_conditional import ConditionalResViTWrapper 

DEVICE = HP['DEVICE']
# DEVICE = 'cpu'
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
psnr = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)

# Model loading function
def load_checkpoint_eval(model, checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

# Metrics calculation
def calculate_val_metrics(output, target):
    mse_loss = nn.MSELoss()(output, target)
    psnr = 10 * torch.log10(1 / mse_loss)
    ssim_value = ssim(output, target, data_range=1.0, size_average=True)
    return psnr.item(), ssim_value.item()

# Configuration
MODEL_CHECKPOINT_PATH = '/home/pks/Desktop/Peeyush/Project/pelvis/MRI_CT_models/MRI_CT_Models-main/sessions/esau-atlas/SPLIT_1/model_weights/psnr/epoch_49__split_1__.pth'
OUTPUT_BASE_PATH = f"/home/pks/Desktop/Peeyush/Project/pelvis/MRI_CT_models/MRI_CT_Models-main/sessions/rectal_output/{EXPERIMENT_NAME}"
config = get_resvit_b16_config()
for key, splitc in SPLITS.items():
    print(f"Processing SPLIT ---------- {key} ----------")
    
    # Initialize model
    model = ESAU("Converter",in_channels=1,n_channels=HP['model_params']['num_channels'],out_channels=1).to(DEVICE)
    # model = ResViT(config=config, input_dim=1, img_size=256, output_dim=1, vis=False).to(DEVICE)
    # model = UNet(in_channels=1, num_classes=1, base_channels=32, bilinear=True).to(DEVICE)
    # model = SwinUNet(256, 256, 1, 128, 1, 4, 2).to(DEVICE)
    # model = PTN_local(img_size=[256,256]).to(DEVICE)
#     model = CascadedGaze(
#     img_channel=1,               
#     width=16,                    
#     middle_blk_num=5,           
#     enc_blk_nums=[1, 1, 2, 3],   
#     dec_blk_nums=[1, 1, 1, 1],   
#     GCE_CONVS_nums=[3, 3, 2, 2]  
# ).to(DEVICE)
    
#     model = SwinTransformerSys(
#     img_size=256,
#     patch_size=4,
#     in_chans=1,
#     num_classes=1,
#     embed_dim=192,           # 2x capacity: 96 → 192
#     depths=[2, 2, 8, 2],     # Deeper bottleneck
#     depths_decoder=[2, 8, 2, 2],
#     num_heads=[6, 12, 24, 48],
#     window_size=8,
#     mlp_ratio=6.,            # Larger MLP: 4 → 6
#     drop_path_rate=0.3
# ).to(DEVICE)

    
    # Load the trained model
    model = load_checkpoint_eval(model, MODEL_CHECKPOINT_PATH, DEVICE)
    model.eval()
    
    # Create output directories
    split_output_path = f"{OUTPUT_BASE_PATH}/SPLIT_{key}"
    os.makedirs(f"{split_output_path}/individual_samples", exist_ok=True)
    os.makedirs(f"{split_output_path}/validation_grids", exist_ok=True)
    
    # Store per-patient metrics
    patient_metrics = {'PSNR': [], 'SSIM': [], 'MSE': [], 'MAE': []}
    all_validation_results = []
    
    with torch.no_grad():
        splitc['validation'] = [ele for ele in splitc['validation'] if '.DS_Store' not in ele]
        validation_loop = tqdm(splitc['validation'], desc=f"Split {key} Validation", leave=True)
        
        for val_scan_index, scan in enumerate(validation_loop):
            validation_DS = test_dataset(scan, HP['training_type'])
            validation_loader = DataLoader(validation_DS, batch_size=1, shuffle=False, 
                                         num_workers=4, pin_memory=True, persistent_workers=True,
                                         prefetch_factor=2)
            
            scan_name_val = scan.split('/')[-1]
            print(f"Processing scan: {scan_name_val}")
            
            # Per-patient accumulation
            patient_psnr_total = 0.0
            patient_ssim_total = 0.0
            patient_mse_total = 0.0
            patient_mae_total = 0.0
            patient_slice_count = 0
            
            # Storage for this scan's results
            scan_results = []
            combined_grids = []
            
            for idx, (ct, mr, mask) in enumerate(validation_loader):
                # CT2MR
                source_real = ct.to(DEVICE).unsqueeze(1)  # Input CT
                target_real = mr.to(DEVICE).unsqueeze(1)  # Ground truth MR
                mask = mask.to(DEVICE).unsqueeze(1)
                
                # Generate fake MR from CT
                target_fake = model(source_real)

                #mask
                target_real = target_real * mask
                source_real = source_real * mask
                target_fake = target_fake * mask

                #skip blank slices inf psnr issue
                mask_pixels = (mask > 0.5).sum().item()
                target_nonzero = (target_real.abs() > 1e-6).sum().item()
                fake_nonzero = (target_fake.abs() > 1e-6).sum().item()
                if mask_pixels < 100 or target_nonzero < 100 or fake_nonzero < 100:
                        print(f"Skipping blank slice {idx}: mask_pixels={mask_pixels}, target_nonzero={target_nonzero}, fake_nonzero={fake_nonzero}")
                        continue


                # Apply post-processing
                for fn in post_processing_order:
                    target_fake = fn(target_fake)
                
                # Resize to match target
                target_size = (target_real.shape[2], target_real.shape[3])
                target_fake = F.interpolate(target_fake, size=target_size, 
                                          mode=HP['inference_interpolation_mode'], 
                                          align_corners=HP['inference_interpolation_allign_cornors']).to(DEVICE)
                
                target_fake = torch.clamp(target_fake, 0, 1)
                
                # Calculate metrics for this batch
                batch_size = source_real.shape[0]
                patient_psnr_total += psnr(target_real, target_fake).mean().item() * batch_size
                patient_ssim_total += ssim(target_real, target_fake).mean().item() * batch_size
                patient_mse_total += F.mse_loss(target_real, target_fake).item() * batch_size
                patient_mae_total += F.l1_loss(target_real, target_fake).item() * batch_size
                patient_slice_count += batch_size
                
                # Create visualization grids for each sample in the batch
                for sample_idx in range(batch_size):
                    # Extract individual samples
                    ct_sample = source_real[sample_idx].squeeze(0)  # Remove channel dim for visualization
                    mr_real_sample = target_real[sample_idx].squeeze(0)
                    mr_fake_sample = target_fake[sample_idx].squeeze(0)


                    
                    # Create side-by-side comparison: Input CT | Generated MR | Ground Truth MR
                    # combined_sample = torch.cat([ct_sample, mr_fake_sample, mr_real_sample], dim=-1)
                    # Resize CT to match MR dimensions
                    target_h, target_w = mr_real_sample.shape
                    if ct_sample.shape != (target_h, target_w):
                        ct_sample = F.interpolate(ct_sample.unsqueeze(0).unsqueeze(0), 
                                                size=(target_h, target_w), 
                                                mode='bilinear', align_corners=False).squeeze()
                        
                    ct_sample_masked = ct_sample * mask[sample_idx].squeeze(0)
                    mr_fake_sample_masked = mr_fake_sample * mask[sample_idx].squeeze(0)
                    mr_real_sample_masked = mr_real_sample * mask[sample_idx].squeeze(0)

                    combined_sample = torch.cat([ct_sample_masked, mr_fake_sample_masked, mr_real_sample_masked], dim=-1)

                    # Save individual sample
                    slice_num = idx * batch_size + sample_idx
                    sample_filename = f"{split_output_path}/individual_samples/{scan_name_val}_slice_{slice_num:03d}.png"
                    save_image(combined_sample, sample_filename)
                    
                    # Store for batch grid
                    scan_results.append(combined_sample)
                
                # Create batch grid for overview
                # grid_ct = make_grid(source_real.squeeze(1), nrow=batch_size, padding=2, normalize=True)
                # Resize source to match target dimensions
                batch_h, batch_w = target_real.shape[2], target_real.shape[3]
                source_resized = F.interpolate(source_real, size=(batch_h, batch_w), mode='bilinear', align_corners=False)
                grid_ct = make_grid(source_resized.squeeze(1), nrow=batch_size, padding=2, normalize=True)
                grid_fake = make_grid(target_fake.squeeze(1), nrow=batch_size, padding=2, normalize=True)
                grid_real = make_grid(target_real.squeeze(1), nrow=batch_size, padding=2, normalize=True)
                
                # Combine all grids: CT | Generated | Real
                combined_grid = torch.cat([grid_ct, grid_fake, grid_real], dim=1)
                combined_grids.append(combined_grid)
            
            # Calculate per-patient averages
            patient_avg_psnr = patient_psnr_total / patient_slice_count
            patient_avg_ssim = patient_ssim_total / patient_slice_count
            patient_avg_mse = patient_mse_total / patient_slice_count
            patient_avg_mae = patient_mae_total / patient_slice_count
            
            # Store patient metrics
            patient_metrics['PSNR'].append(patient_avg_psnr)
            patient_metrics['SSIM'].append(patient_avg_ssim)
            patient_metrics['MSE'].append(patient_avg_mse)
            patient_metrics['MAE'].append(patient_avg_mae)
            
            # Save patient results
            patient_result = {
                'scan_name': scan_name_val,
                'split': key,
                'avg_psnr': patient_avg_psnr,
                'avg_ssim': patient_avg_ssim,
                'avg_mse': patient_avg_mse,
                'avg_mae': patient_avg_mae,
                'num_slices': patient_slice_count
            }
            all_validation_results.append(patient_result)
            
            # Save scan overview grid
            if combined_grids:
                scan_overview_grid = torch.cat(combined_grids, dim=2)
                scan_grid_filename = f"{split_output_path}/validation_grids/{scan_name_val}_overview.png"
                save_image(scan_overview_grid, scan_grid_filename)
            
            print(f"Scan {scan_name_val}: PSNR={patient_avg_psnr:.4f}, SSIM={patient_avg_ssim:.4f}")
    
    # Calculate overall split metrics
    split_metrics = {
        'split': key,
        'avg_psnr': np.mean(patient_metrics['PSNR']),
        'avg_ssim': np.mean(patient_metrics['SSIM']),
        'avg_mse': np.mean(patient_metrics['MSE']),
        'avg_mae': np.mean(patient_metrics['MAE']),
        'std_psnr': np.std(patient_metrics['PSNR']),
        'std_ssim': np.std(patient_metrics['SSIM']),
        'num_patients': len(splitc['validation']),
        'total_slices': sum([r['num_slices'] for r in all_validation_results])
    }
    
    # Save results
    with open(f"{split_output_path}/split_metrics.json", "w") as f:
        json.dump(split_metrics, f, indent=2)
    
    with open(f"{split_output_path}/patient_results.json", "w") as f:
        json.dump(all_validation_results, f, indent=2)
    
    # Save as CSV for easy analysis
    df_patients = pd.DataFrame(all_validation_results)
    df_patients.to_csv(f"{split_output_path}/patient_results.csv", index=False)
    
    print(f'Split {key} validation completed.')
    print(f'Average PSNR: {split_metrics["avg_psnr"]:.4f} ± {split_metrics["std_psnr"]:.4f}')
    print(f'Average SSIM: {split_metrics["avg_ssim"]:.4f} ± {split_metrics["std_ssim"]:.4f}')
    print(f'Results saved to: {split_output_path}')
    print("-" * 60)

print("Validation complete for all splits!")