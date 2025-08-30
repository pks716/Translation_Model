import os
from training_hyperparameters import data_directory

all_patients = os.listdir(data_directory)
all_patients = sorted([p for p in all_patients if '.DS_Store' not in p])

# 90% train, 10% validation split
split_idx = int(len(all_patients)*0.90)

SPLITS = {1:{
    'train': [os.path.join(data_directory, all_patients[i]) for i in range(split_idx)],
    'validation': [os.path.join(data_directory, all_patients[i]) for i in range(split_idx, len(all_patients))],
}}

if __name__ == '__main__':
    print(f"Total patients: {len(all_patients)}")
    print(f"Train: {len(SPLITS['train'])} patients ({len(SPLITS['train'])/len(all_patients)*100:.1f}%)")
    print(f"Validation: {len(SPLITS['validation'])} patients ({len(SPLITS['validation'])/len(all_patients)*100:.1f}%)")