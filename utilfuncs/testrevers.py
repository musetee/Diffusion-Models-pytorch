from mydataloader.basics import get_transforms, get_file_list, load_volumes, crop_volumes	
from torch.utils.data import DataLoader
import torch
dataset_path="E:\Projects\yang_proj\Task1\pelvis"
normalize='minmax'
pad='minimum'
train_number=1
val_number=1
train_batch_size=8
val_batch_size=1
saved_name_train='./train_ds_2d.csv'
saved_name_val='./val_ds_2d.csv'
resized_size=(512,512,None)
div_size=(16,16,None)
center_crop=20
ifcheck_volume=False
ifcheck_sclices=False

# volume-level transforms for both image and label
train_transforms = get_transforms(normalize,pad,resized_size,div_size,center_crop=center_crop)
train_ds, val_ds = get_file_list(dataset_path, 
                                    train_number, 
                                    val_number)
train_ds, val_ds = crop_volumes(train_ds, val_ds,center_crop)
"""
train_ds, val_ds = load_volumes(train_transforms, 
                                train_ds, 
                                val_ds, 
                                saved_name_train, 
                                saved_name_val,
                                ifsave=False,
                                ifcheck=ifcheck_volume)
"""
loader = DataLoader(train_ds, batch_size=1)
for idx, checkdata in enumerate(loader):
    print(checkdata['label'].shape)
    break

from monai.transforms.utils import allow_missing_keys_mode
labeldata=checkdata['label']
#print(labeldata.applied_operations)
dict = {"image": labeldata[0,:,:,:,:]} # always set val_batch_size=1
with allow_missing_keys_mode(train_transforms):
    reversed_data=train_transforms.inverse(dict)
print(reversed_data["image"].shape)