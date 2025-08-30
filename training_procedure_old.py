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
from splits_only_val import SPLITS
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

if wand_db_boolean:
    import wandb

DEVICE = HP['DEVICE']
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
psnr = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
'''
CT will always be the first argument returnned by the dataloader

'''


base_path = f"sessions/{EXPERIMENT_NAME}"

# Boilerplate model loading and saving
def save_checkpoint(model, optimizer, scheduler, epoch, path):
    if '/' in path:
        name = path.split('/')[-1]
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)       # Create directories if they don't exist
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict()
    }, path)

def load_checkpoint(model, optimizer, scheduler, checkpoint, device=None):
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint.get('model_state_dict', {}))
    else:
        ValueError("Check point does not contain hey for model_state_dict")
    if 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint.get('optimizer_state_dict',{}))
    else:
        ValueError("Check point does not contain key for optimizer_state_dict")
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint.get('scheduler_state_dict',{}))
    else:
        ValueError("Check point does not contain key for scheduler_state_dict")
        
def model_restore_state(check_point_path):
    model_ckpt_name = check_point_path.split('/')[-1]
    epoch = model_ckpt_name.split('__')[0]
    splitd = model_ckpt_name.split('__')[1]
    print(epoch)
    print(splitd)
    
    
    return int(epoch.split("_")[1]), int(splitd.split("_")[1])
    
    

def load_checkpoint_eval(model, checkpoint_path, devvicee):
    checkpoint = torch.load(checkpoint_path, map_location=devvicee)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model



def convert_to_dict(x):
    if isinstance(x, torch.Tensor):
        return {'level_0': x}
    return x

def define_pretrained(model_name):
    if model_name == 'vgg19':
        pretrained = torchvision.models.vgg19(weights=torchvision.models.vgg.VGG19_Weights.IMAGENET1K_V1)
    elif model_name == 'resnet152':
        pretrained = torchvision.models.resnet152(weights=torchvision.models.resnet.ResNet152_Weights)
    return pretrained

class PerceptionLoss2D(nn.Module):
    def __init__(
        self,
        feature_extractor,
        loss_fn=nn.L1Loss(),
        channel_dim: int = 3,
        normalize_mean: list = [0.485, 0.456, 0.406],
        normalize_std: list = [0.229, 0.224, 0.225],
        separate_channel: bool = True,
        base_weight: float = 1.0,
    ):
        super().__init__()
        self.loss_fn = loss_fn
        self.feature_extractor = feature_extractor
        self.feature_extractor.eval()
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.channel_dim = channel_dim
        
        if normalize_mean is not None:
            c = len(normalize_mean)
            self.normalize_mean = torch.Tensor(normalize_mean).view(1, c, 1, 1)
            self.normalize_std = torch.Tensor(normalize_std).view(1, c, 1, 1)
        else:
            self.normalize_mean = normalize_mean
            self.normalize_std = normalize_std
        
        self.separate_channel = separate_channel
        self.base_weight = base_weight
    
    def forward(self, out, target, return_record=False):
        if isinstance(out, dict):
            out = out['level_0']
        if isinstance(target, dict):
            target = target['level_0']
            
        if self.normalize_mean is not None:
            device = out.device
            self.normalize_mean = self.normalize_mean.to(device)
            self.normalize_std = self.normalize_std.to(device)
            out = (out - self.normalize_mean) / (self.normalize_std + 1e-5)
            target = (target - self.normalize_mean) / (self.normalize_std + 1e-5)
        
        # Handle channel dimension for 2D images
        b, c, h, w = out.shape
        if c != self.channel_dim or self.separate_channel:
            # Convert grayscale to RGB by repeating channels
            if c == 1:
                out = out.repeat(1, self.channel_dim, 1, 1)
                target = target.repeat(1, self.channel_dim, 1, 1)
        
        # Extract features
        o_features = self.feature_extractor(out)
        t_features = self.feature_extractor(target)
        
        # Calculate perceptual loss
        loss = 0
        for key in o_features.keys():
            loss += self.loss_fn(o_features[key], t_features[key]) / self.base_weight
        
        loss_record = loss.item()
        
        if return_record:
            return loss, loss_record
        else:
            return loss
        
class CombinedLoss(nn.Module):
    def __init__(self, l1_weight=1.0, perceptual_weight=0.1, device='cuda'):
        super(CombinedLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        
        # Create VGG19 feature extractor
        model_name = 'vgg19'
        return_nodes = ['features.35']  # Single layer for reduced complexity
        
        pretrained = define_pretrained(model_name).eval()
        feature_extractor = create_feature_extractor(pretrained, return_nodes)
        feature_extractor.to(device)
        
        self.perceptual_loss = PerceptionLoss2D(
            feature_extractor=feature_extractor,
            loss_fn=nn.L1Loss(),
            channel_dim=3,
            separate_channel=True,
            base_weight=1.0  # Single layer so base_weight = 1
        )
        
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        
    def forward(self, pred, target):
        l1 = self.l1_loss(pred, target)
        perceptual = self.perceptual_loss(pred, target)
        
        total_loss = self.l1_weight * l1 + self.perceptual_weight * perceptual
        return total_loss
    
# With hybrid loss that combines both:
class HybridLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7):
        super().__init__()
        self.alpha = alpha  # Weight for classification
        self.beta = beta    # Weight for regression (L1 only, skip perceptual for now)
        self.ce = nn.CrossEntropyLoss()
        self.l1 = nn.L1Loss()  # Use simple L1 instead of combined loss
    
    def forward(self, pred_logits, target):
        # Classification loss
        class_targets = intensity_to_class(target.squeeze(1))  # [B, H, W]
        class_loss = self.ce(pred_logits, class_targets)
        
        # Convert predictions back to continuous for regression loss
        class_predictions = torch.softmax(pred_logits, dim=1)
        continuous_pred = torch.sum(class_predictions * torch.arange(1000, device=pred_logits.device).view(1, -1, 1, 1) / 999.0, dim=1, keepdim=True)
        
        # Simple L1 loss (both should be [B, 1, H, W])
        regression_loss = self.l1(continuous_pred, target)
        
        return self.alpha * class_loss + self.beta * regression_loss

criterion = HybridLoss(alpha=0.5, beta=0.5)
# criterion = CombinedLoss(l1_weight=1.0, perceptual_weight=0.1, device=DEVICE)
# criterion = nn.CrossEntropyLoss()

# Metrics calculation
def calculate_val_metrics(output, target):
    mse_loss = nn.MSELoss()(output, target)
    psnr = 10 * torch.log10(1 / mse_loss)
    ssim_value = ssim(output, target, data_range=1.0, size_average=True)
    return psnr.item(), ssim_value.item()

def intensity_to_class(intensity_tensor, num_classes=1000):
    """Convert continuous intensities [0,1] to class indices [0, num_classes-1]"""
    # Scale from [0,1] to [0, num_classes-1] 
    class_indices = (intensity_tensor * (num_classes - 1)).long()
    class_indices = torch.clamp(class_indices, 0, num_classes - 1)
    return class_indices

def class_to_intensity(class_tensor, num_classes=1000):
    """Convert class indices back to continuous intensities"""
    return class_tensor.float() / (num_classes - 1)

split_resolved, epoch_resolved = True, True, 
if continue_path:
    split_resolved, epoch_resolved = False, False, 

for key, splitc in SPLITS.items():
    config = get_resvit_b16_config()
    # conditional_wrapper = ConditionalResViTWrapper(config, DEVICE, conditioning_mode='fusion_gate')
    # optimizer = conditional_wrapper.setup_optimizer(lr=HP['learning_rate'])
    # model = ESAU("Converter",in_channels=1,n_channels=HP['model_params']['num_channels'],out_channels=1).to(DEVICE)
    # model = ResViT(config=config, input_dim=1, img_size=256, output_dim=1, vis=False).to(DEVICE)
    model = UNet(in_channels=1, num_classes=1000, base_channels=32, bilinear=True).to(DEVICE)
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

    # model.load_state_dict(torch.load('/home/pks/Desktop/Peeyush/Project/pelvis/sessions/Unet_base_CT2MR/SPLIT_1/model_weights/psnr/epoch_99__split_1__.pth', map_location=DEVICE)['model_state_dict'], strict=False)
    checkpoint = torch.load('/home/pks/Desktop/Peeyush/Project/pelvis/sessions/Unet_base_CT2MR/SPLIT_1/model_weights/psnr/epoch_99__split_1__.pth', map_location=DEVICE)['model_state_dict']
    checkpoint.pop('outc.conv.weight', None)
    checkpoint.pop('outc.conv.bias', None)
    model.load_state_dict(checkpoint, strict=False)

    optimizer = optim.Adam(model.parameters(), lr=HP['learning_rate'])
    scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    
    best_models_psnr = deque(maxlen=3)
    best_models_ssim = deque(maxlen=3)
    
    

    if not split_resolved:
        e, spll =  model_restore_state(continue_path)
        if key < spll :
            continue
        load_checkpoint(model, optimizer, scheduler, continue_path)
                    
        top_location = ""
        for ele in continue_path.split('/'):
            if ele == "ssim" or ele == "psnr":
                break
            top_location = f"{top_location}/{ele}"
        
        with open(f"{top_location}/Top3PSNR.pkl", "rb") as f:
            best_models_psnr = pickle.load(f)
        with open(f"{top_location}/Top3SSIM.pkl", "rb") as f:
            best_models_ssim = pickle.load(f)
            
        split_resolved = True
    if wand_db_boolean:
        run = wandb.init(project=PROJECT, 
                        name = f"{EXPERIMENT_NAME} Split-{key}",
                        config= HP,
                        resume="allow"
                        )

    print("Processing SPLIT ---------- " , key," ---------- ")
    split_base_path = f"{base_path}/SPLIT_{key}"
    os.makedirs(split_base_path,exist_ok=True)
    trr = train_dataset(splitc['train'])
    train_dataloader = DataLoader(trr, batch_size=HP['batch_size'], shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True,
                            prefetch_factor=4)

    print(f"Split {key} - Train: {len(splitc['train'])}, Validation: {len(splitc['validation'])}")

    for epoch in range(HP['epochs']):
        if not epoch_resolved:
            if epoch < e:
                continue
            epoch_resolved = True
        
        ######################################

        model.train()
        train_dataloader = tqdm(train_dataloader, desc=f"Split {key} Training Epoch {epoch+1}/{HP['epochs']}", leave=True)
        # Example: Fetch one batch
        # for idx, (ct_batch, mr_batch , mask) in enumerate(train_dataloader):
        #     ct_batch, mr_batch, mask = ct_batch.to(DEVICE), mr_batch.to(DEVICE), mask.to(DEVICE)
        for idx, (ct_batch, mr_batch) in enumerate(train_dataloader):
            ct_batch, mr_batch = ct_batch.to(DEVICE), mr_batch.to(DEVICE)
            ct_batch = ct_batch.unsqueeze(1)
            mr_batch = mr_batch.unsqueeze(1)
            
            mr_fake = model(ct_batch)
            optimizer.zero_grad()
            
            # # NEW: Convert MR to class labels
            # mr_labels = intensity_to_class(mr_batch.squeeze(1), num_classes=1000)  # Remove channel dim for CrossEntropy
            # # loss =  criterion(mr_fake, mr_batch)

            # # Convert MR intensities to class labels
            # # mr_labels = intensity_to_class(mr_batch.squeeze(1), num_classes=1000)  # Remove channel dim: [B, H, W]
            # loss = criterion(mr_fake, mr_labels)

            loss = criterion(mr_fake, mr_batch) 
            
            loss.backward()
            optimizer.step()
            train_dataloader.set_postfix(batch_loss=loss.item())
                
        scheduler.step()
        
        if wand_db_boolean:
            wandb.log({
                "epoch": epoch,  # Explicitly log the epoch
                "Train_loss" : loss,
                'tag' : "Training Loss",
                'step' : epoch

            })
            
        
        
        ##########################################################################################
        
        model.eval()
        
        # ADD: Store per-patient metrics
        patient_metrics = {'PSNR': [], 'SSIM': [], 'MSE': [], 'MAE': []}
        
        validation_metrics_epoch = {
            'step' : epoch,
            'ValidationPSNR': 0,
            'ValidationSSIM': 0,
            'ValidationMSE': 0,
            'ValidationMAE': 0,
            'epoch' : epoch
            }
        
        with torch.no_grad():
            model_path_val = f"{split_base_path}/model_weights"
            os.makedirs(f"{model_path_val}/ssim", exist_ok=True)
            os.makedirs(f"{model_path_val}/psnr", exist_ok=True)
            val_index= 0
            splitc['validation'] = [ele for ele in splitc['validation'] if '.DS_Store' not in ele]
            validation_loop = tqdm(splitc['validation'], desc=f"Split {key} Validation Loop {val_index+1}/{len(splitc['validation'])}", leave=True)
            
            for val_scan_index, scan in enumerate(validation_loop): 
                # if val_scan_index ==0:
                save_array_val = []
                combined_grid_val = []      

                validation_DS = test_dataset(scan, HP['training_type'])
                
                validation_loader = DataLoader(validation_DS, batch_size=HP['batch_size'], shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True,
                             prefetch_factor=2)

                scan_name_val = scan.split('/')[-1]

                #Per-patient accumulation
                patient_psnr_total = 0.0
                patient_ssim_total = 0.0
                patient_mse_total = 0.0
                patient_mae_total = 0.0
                patient_slice_count = 0                
                
                # Example: Fetch one batch
                # for idx, (ct, mr, mask) in enumerate(validation_loader):
                for idx, (ct, mr) in enumerate(validation_loader):
                    #CT2MR
                    source_real = ct.to(DEVICE).unsqueeze(1)
                    target_real = mr.to(DEVICE).unsqueeze(1)
                    # mask = mask.to(DEVICE)
                    
                    # target_fake = model(source_real)
                    # With:
                    logits = model(source_real)  # [B, 1000, H, W]
                    class_predictions = torch.softmax(logits, dim=1)  # Convert to probabilities
                    # Weighted average to get continuous intensity
                    target_fake = torch.sum(class_predictions * torch.arange(1000, device=DEVICE).view(1, -1, 1, 1) / 999.0, dim=1, keepdim=True)
                                        
                    for fn in post_processing_order:
                        target_fake = fn(target_fake)
                        
                    target_size = (target_real.shape[2],target_real.shape[3])
                    target_fake = F.interpolate(target_fake, size=target_size, mode=HP['inference_interpolation_mode'], align_corners=HP['inference_interpolation_allign_cornors']).to(DEVICE)
                    
                    
                    
                    # target_fake = target_fake*mask.unsqueeze(1)
                    # target_real = target_real*mask.unsqueeze(1)
                    
                    target_fake = torch.clamp(target_fake, 0, 1)

                    # CHANGE: Accumulate for this patient only
                    batch_size = source_real.shape[0]
                    patient_psnr_total += psnr(target_real, target_fake).mean().item() * batch_size
                    patient_ssim_total += ssim(target_real, target_fake).mean().item() * batch_size
                    patient_mse_total += F.mse_loss(target_real, target_fake).item() * batch_size
                    patient_mae_total += F.l1_loss(target_real, target_fake).item() * batch_size
                    patient_slice_count += batch_size
                    
                    
                    grid_fake = make_grid(target_fake, nrow=target_fake.shape[0], padding=2)
                    grid_real = make_grid(target_real, nrow=target_real.shape[0], padding=2)
                
                
                
                    # if val_scan_index == 0:
                    combined = torch.cat([target_real.squeeze(1), target_fake.squeeze(1)], dim=-1)
                    save_array_val.append(combined)
                    combined_grid_val.append(torch.cat((grid_real, grid_fake), dim=1))
                    val_index+=1  



                # ADD: Calculate per-patient averages and store
                patient_metrics['PSNR'].append(patient_psnr_total / patient_slice_count)
                patient_metrics['SSIM'].append(patient_ssim_total / patient_slice_count)
                patient_metrics['MSE'].append(patient_mse_total / patient_slice_count)
                patient_metrics['MAE'].append(patient_mae_total / patient_slice_count)
                    
  
                # if val_scan_index == 0:
                eval_scanvv = torch.cat(save_array_val, dim=0) 
                eval_scanvv = eval_scanvv.cpu().numpy()
                os.makedirs(f"{split_base_path}/validation", exist_ok=True)
                # np.save(f"{split_base_path}/validation/{scan_name_val}_{epoch}.npy", eval_scanvv)
                combined_grid_val = torch.cat(combined_grid_val, dim=2)
                save_image(combined_grid_val, f"{split_base_path}/validation/{scan_name_val}_{epoch}.png")
                if wand_db_boolean:
                    wandb.log({f"Validation Generation Split_{key}": wandb.Image(f"{split_base_path}/validation/{scan_name_val}_{epoch}.png"),
                                   "step" : epoch,
                                   'tag' : "Validation Image",
                                   'caption' : f"{scan_name_val}_{epoch}_{key} Validation",
                                    "epoch" : epoch,
                                   
                                })
                        # val_img_artifact.add_file(f"{split_base_path}/validation/{scan_name_val}_{epoch}.png", name=f"Epoch {epoch}.png")
                    
                        
        # CHANGE: Calculate final averages across patients (not slices)
        validation_metrics_epoch['ValidationPSNR'] = np.mean(patient_metrics['PSNR'])
        validation_metrics_epoch['ValidationSSIM'] = np.mean(patient_metrics['SSIM'])
        validation_metrics_epoch['ValidationMSE'] = np.mean(patient_metrics['MSE'])
        validation_metrics_epoch['ValidationMAE'] = np.mean(patient_metrics['MAE'])
        
        fold_validation_results = {
        'fold': key,
        'epoch': epoch,
        'avg_psnr': validation_metrics_epoch['ValidationPSNR'],
        'avg_ssim': validation_metrics_epoch['ValidationSSIM'], 
        'avg_mse': validation_metrics_epoch['ValidationMSE'],
        'avg_mae': validation_metrics_epoch['ValidationMAE'],
        'num_val_patients': len(splitc['validation'])}
        os.makedirs(f"{split_base_path}/validation_averages", exist_ok=True)
        with open(f"{split_base_path}/validation_averages/fold_{key}_epoch_{epoch}.json", "w") as f:
            json.dump(fold_validation_results, f, indent=2)


        samples = 0 # Creating div by 0 in case of an error
        
        
        
        best_models_psnr.append((validation_metrics_epoch['ValidationPSNR'], f"{model_path_val}/psnr/epoch_{epoch}__split_{key}__.pth"))
        best_models_ssim.append((validation_metrics_epoch['ValidationSSIM'], f"{model_path_val}/ssim/epoch_{epoch}__split_{key}__.pth"))
        
        
        if wand_db_boolean:
            wandb.log(validation_metrics_epoch)
        
        # Resume logic []
        best_models_psnr = deque(sorted(best_models_psnr, reverse=True)[:3], maxlen=3)
        best_models_ssim = deque(sorted(best_models_ssim, reverse=True)[:3], maxlen=3)
        
        top_models_pnsr = set(m[1] for m in best_models_psnr)
        top_models_ssim = set(m[1] for m in best_models_ssim)
        
        
        if f"{model_path_val}/psnr/epoch_{epoch}__split_{key}__.pth" in top_models_pnsr:
            save_checkpoint(model, optimizer, scheduler,epoch, f"{model_path_val}/psnr/epoch_{epoch}__split_{key}__.pth")
            with open(f"{model_path_val}/Top3PSNR.pkl","wb") as f:
                pickle.dump(best_models_psnr, f)
        
        if f"{model_path_val}/ssim/epoch_{epoch}__split_{key}__.pth" in top_models_ssim:
            save_checkpoint(model, optimizer, scheduler,epoch, f"{model_path_val}/ssim/epoch_{epoch}__split_{key}__.pth")
            with open(f"{model_path_val}/Top3SSIM.pkl", "wb") as f:
                pickle.dump(best_models_ssim, f)
            
        # Delete if bad performer.
        for file in os.listdir(f"{model_path_val}/psnr"):
            file_path = os.path.join(f"{model_path_val}/psnr", file)
            if file_path not in top_models_pnsr:
                os.remove(file_path)
                
        for file in os.listdir(f"{model_path_val}/ssim"):
            file_path = os.path.join(f"{model_path_val}/ssim", file)
            if file_path not in top_models_ssim:
                os.remove(file_path)
            
        df = pd.DataFrame.from_dict(dict(best_models_psnr), orient='index')

        df.to_csv(f"{model_path_val}/Top3PSNR.csv", index_label="Epoch")
        
        df = pd.DataFrame.from_dict(dict(best_models_ssim), orient='index')
        df.to_csv(f"{model_path_val}/Top3SSIM.csv", index_label="Epoch")
        
        # Split completion
        print(f'Split {key} training completed. Best PSNR: {best_models_psnr[0][0]:.4f}, Best SSIM: {best_models_ssim[0][0]:.4f}')

    if wand_db_boolean:
        wandb.log({
                "Final_Best_PSNR": best_models_psnr[0][0],
                "Final_Best_SSIM": best_models_ssim[0][0], 
                "Split": key,
                "tag": "Final Metrics"
            })
        run.finish()   

    
    
######################################################################################################
# ######################################################################################################
# ##########                                                                   #########################
# ##########                    TESTING LOGIC                                  #########################
# ##########                                                                   #########################
# ######################################################################################################
# ######################################################################################################

#     # PSNR LEAD VALIDATION STRATERGY 
    
#     print('Evaluating for top PSNR model')
#     model = load_checkpoint_eval(model, best_models_psnr[0][1], DEVICE)
    
#     testing_metrics = {
#         'TestingPSNR': 0,
#         'TestingSSIM' : 0,
#         'TestingMAE' : 0,
#         'TestingMSE' : 0,
#         'samples': 0,
#         'tag' : "Testing",
#         'epoch' : key,
#         'SPLIT' : key,
#         'step' : key,
        
#     }
#     model.eval()
#     with torch.no_grad():
#         model_path_testing = f"{split_base_path}/testing/psnr"
#         os.makedirs(f"{model_path_testing}/samples", exist_ok=True)
#         test_index = 0
#         splitc['test'] = [ele for ele in splitc['test']]
#         testing_loop = tqdm(splitc['test'], desc=f"Split {key} Testing Loop {test_index+1}/{len(splitc['test'])}", leave=True)
        
#         for ggdz, scan in enumerate(testing_loop):
#             # b, h, w
#             testing_DS = test_dataset(scan, HP['training_type'])
#             train_dataloader = DataLoader(testing_DS, batch_size=HP['batch_size'], shuffle=False)

#             scan_name = scan.split('/')[-1]
#             # print(scan_name)
#             scans_produced = []
            
#             combined_grid = []
            
#             # Example: Fetch one batch
#             for idx, (ct, mr, mask) in enumerate(train_dataloader):

#                 # b,h,w
#                 #CT2MR
#                 source_real = ct.to(DEVICE).unsqueeze(1)
#                 target_real = mr.to(DEVICE).unsqueeze(1)
#                 mask = mask.to(DEVICE).unsqueeze(1)
                
#                 testing_metrics['samples'] +=  source_real.shape[0] # Since SSIM/PSNR will be already mean when saved.
                
#                 target_fake = model(source_real)
                
#                 for fn in post_processing_order:
#                     target_fake = fn(target_fake)
                    
#                 # this case the target for normalization in only going to be the processed modality.
#                 target_size = (target_real.shape[2],target_real.shape[3])
#                 target_fake = F.interpolate(target_fake, size=target_size, mode=HP['inference_interpolation_mode'], align_corners=HP['inference_interpolation_allign_cornors'])
                
#                 target_fake = target_fake
#                 target_real = target_real
                
#                 target_fake = torch.clamp(target_fake, 0, 1)
#                 grid_fake = make_grid(target_fake, nrow=target_fake.shape[0])
#                 grid_real = make_grid(target_real, nrow=target_real.shape[0])
                
                
                
#                 combined_grid.append(torch.cat((grid_real, grid_fake), dim=1))
                
                
#                 testing_metrics['TestingSSIM'] += ssim(target_real, target_fake).mean().item() * source_real.shape[0]
#                 testing_metrics['TestingPSNR'] += psnr(target_real, target_fake).mean().item() * source_real.shape[0]
#                 testing_metrics['TestingMSE'] +=  F.mse_loss(target_real, target_fake).item()
#                 testing_metrics['TestingMAE'] +=  F.l1_loss(target_real, target_fake).item()
                
#                 target_fake = target_fake.squeeze(1)
#                 target_real = target_real.squeeze(1)
                 
#                 combined = torch.cat([target_real, target_fake], dim=-1)
#                 scans_produced.append(combined)
            

#             try:
#                 combined_grid = torch.cat(combined_grid, dim=2)
#                 eval_scan = torch.cat(scans_produced, dim=0)  # Ensure dim=0 for batch concatenation
#                 np.save(f"{model_path_testing}/samples/{scan_name}.npy", eval_scan.cpu())
#                 ASD = [ele.shape for ele in combined_grid]
#                 print("combined shape", combined_grid.shape)
#                 save_image(combined_grid, f"{model_path_testing}/samples/{scan_name}.png")
#                 if wand_db_boolean:
#                     wandb.log({f"Testing Generation Split_{key}": wandb.Image(f"{model_path_testing}/samples/{scan_name}.png"),
#                         "step" : ggdz,
#                         "epoch" : ggdz,
                        
#                         'tag' : "Testing Image",
#                         'caption' : f"{scan_name}_{epoch}_{key} Testing"
                        
#                     })
#             except RuntimeError as e:
#                 print(f"Skipped visualization for {scan_name}")

            
#             test_index+=1
                
                
#     testing_metrics['TestingPSNR'] = testing_metrics['TestingPSNR']/testing_metrics['samples']
#     testing_metrics['TestingSSIM'] = testing_metrics['TestingSSIM']/testing_metrics['samples']
#     testing_metrics['TestingMAE'] = testing_metrics['TestingMAE']/testing_metrics['samples']
#     testing_metrics['TestingMSE'] = testing_metrics['TestingMSE']/testing_metrics['samples']
#     if wand_db_boolean:
#         wandb.log(testing_metrics)
    
#     df_test = pd.DataFrame.from_dict(testing_metrics, orient='index')
#     df_test.to_csv(f"{model_path_testing}/SplitMetrics.csv", index_label="Epoch")
    
#     if wand_db_boolean:
#         run.finish()
            
