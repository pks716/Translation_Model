import cv2
import numpy as np


full_dataset = True # False Limits to 5 totals scans
all_slices = True # False Limits to 10 slides per patient

data_directory = '/storage/ss_peeyush/3D-mADUNet-master/data/synthrad23/train/Task1/pelvis'
destination_directory = '/storage/ss_peeyush/MRI_CT_Models-main/DataCreation/Data'


per_sample_normalize = True

view = {
    'axial' : True,
    'coronal' : False,
    'saggital' : False
}
slice_size = (256,256)
interpolation_method = cv2.INTER_LINEAR
# padding_size = 608

# Do not edir
view_index_hh = {
    'axial' : 2,
    'coronal' : 1,
    'saggital' : 0
}

print([n for m, n in view.items()])