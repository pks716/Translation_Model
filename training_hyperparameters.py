import cv2  

import subprocess

session_name = subprocess.check_output(["tmux", "display-message", "-p", "#S"], text=True).strip()


wand_db_boolean = False

# PROJECT = "CT2_MR_Pelvis_only_T2W"
PROJECT = "CT2_MR_Pelvis_only_T2W_rectal"
EXPERIMENT_NAME = 'esau-rectal'
continue_path = ""


data_directory = "/home/pks/Desktop/Peeyush/Project/pelvis/MRI_CT_models/MRI_CT_Models-main/DataCreation/rectal/slices"

# rm_DSStore(data_directory)
 # either 'MR2CT' or 'CT2MR'
# This is only for dataloaders, changes in training_procedures need to be manual.
# CT will always be the first argument returnned by the dataloader

evaluating_run_boolean = True # Keep False while training


HP = {
    'DEVICE' : 'cuda:0',

    'model_params':{
        'type': 'esau-atlas',
        'num_channels': 32
    },
    'System': "Ruby",
    'TMUX' : session_name,
    "batch_size": 1,
    "learning_rate": 1e-3,
    "epochs": 50,
    'loss_weights' : {

    },
    'inference_interpolation_mode' : 'bilinear',
    'inference_interpolation_allign_cornors' : False,
    'training_type' : 'CT2MR' 
}

helper_parameters= {
    'align_corners' : True
}

