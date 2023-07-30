import monai
from monai.transforms import (
    Compose,
    LoadImaged,
    Rotate90d,
    ScaleIntensityd,
    EnsureChannelFirstd,
    ResizeWithPadOrCropd,
    DivisiblePadd,
    ThresholdIntensityd,
    NormalizeIntensityd,
    SqueezeDimd,
    Identityd,
    CenterSpatialCropd,
)
from monai.data import Dataset
from torch.utils.data import DataLoader
import torch

from .basics import get_file_list, check_batch_data, get_transforms, load_volumes

def get_file_list(data_pelvis_path, train_number, val_number):
    #list all files in the folder
    file_list=[i for i in os.listdir(data_pelvis_path) if 'overview' not in i]
    file_list_path=[os.path.join(data_pelvis_path,i) for i in file_list]
    #list all ct and mr files in folder
    ct_file_list=[os.path.join(j,'ct.nii.gz') for j in file_list_path]
    cond_file_list=[os.path.join(j,'ct_slice_cond.csv') for j in file_list_path]
    # Dict Version
    train_ds = [{'image': i, 'label': j, 'A_paths': i, 'B_paths': j} for i, j in zip(ct_file_list[0:train_number], cond_file_list[0:train_number])]
    val_ds = [{'image': i, 'label': j, 'A_paths': i, 'B_paths': j} for i, j in zip(ct_file_list[-val_number:], cond_file_list[-val_number:])]
    print('all files in dataset:',len(file_list))
    return train_ds, val_ds

def load_volumes(train_transforms, train_ds, val_ds, saved_name_train=None, saved_name_val=None,ifsave=False,ifcheck=False):
    train_volume_ds = monai.data.Dataset(data=train_ds, transform=train_transforms) 
    val_volume_ds = monai.data.Dataset(data=val_ds, transform=train_transforms)
    if ifsave:
        save_volumes(train_ds, val_ds, saved_name_train, saved_name_val)
    if ifcheck:
        check_volumes(train_ds, train_volume_ds, val_volume_ds, val_ds)
    return train_volume_ds,val_volume_ds

def get_cond(data_pelvis_path, train_number, val_number):
    #list all files in the folder
    file_list=[i for i in os.listdir(data_pelvis_path) if 'overview' not in i]
    file_list_path=[os.path.join(data_pelvis_path,i) for i in file_list]
    #condition files in folder
    cond_file_list=[os.path.join(j,'ct_slice_cond.csv') for j in file_list_path]

    train_cond_src = cond_file_list[0:train_number] #{'label': j, 'cond_paths': j}
    val_cond_src = cond_file_list[-val_number:]

    train_cond = []
    val_cond = []
    for cond_file in train_cond_src:
        with open(cond_file, 'r', newline='') as csvfile:
            import csv
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                # Append the value to the list
                train_cond.append(int(row['slice']))  # Assuming each row contains only one value

    for cond_file in val_cond_src:
        with open(cond_file, 'r', newline='') as csvfile:
            import csv
            csv_reader = csv.DictReader(csvfile)
            for row in csv_reader:
                # Append the value to the list
                val_cond.append(int(row['slice']))
    #print(len(train_cond))
    #print(len(val_cond))
    num_classes = max(max(train_cond), max(val_cond))
    print(num_classes)

    class_names = [i+1 for i in range(num_classes)]
    print(class_names)
    return train_cond, val_cond, class_names

def get_transforms(normalize,resized_size,div_size,center_crop=0):
    transform_list=[]
    transform_list.append(LoadImaged(keys=["image"]))
    transform_list.append(EnsureChannelFirstd(keys=["image"]))

    if normalize=='zscore':
        transform_list.append(NormalizeIntensityd(keys=["image"], nonzero=False, channel_wise=True))
        print('zscore normalization')

    elif normalize=='minmax':
        transform_list.append(ScaleIntensityd(keys=["image"], minv=-1.0, maxv=1.0))
        print('minmax normalization')
    elif normalize=='none':
        print('no normalization')

    transform_list.append(ResizeWithPadOrCropd(keys=["image"], spatial_size=resized_size,mode="minimum"))
    transform_list.append(Rotate90d(keys=["image"], k=3))
    transform_list.append(DivisiblePadd(["image"], k=div_size, mode="minimum"))
    if center_crop>0:
        transform_list.append(CenterSpatialCropd(keys=["image"], roi_size=(-1,-1,center_crop)))
    train_transforms = Compose(transform_list)
    # volume-level transforms for both image and label
    return train_transforms

##### slices #####
def load_batch_slices(train_volume_ds,val_volume_ds, train_batch_size=8,val_batch_size=1,window_width=1,ifcheck=True):
    patch_func = monai.data.PatchIterd(
        keys=["image"],
        patch_size=(None, None, window_width),  # dynamic first two dimensions
        start_pos=(0, 0, 0)
    )
    if window_width==1:
        patch_transform = Compose(
            [
                SqueezeDimd(keys=["image"], dim=-1),  # squeeze the last dim
            ]
        )
    else:
        patch_transform = None
    # for training
    train_patch_ds = monai.data.GridPatchDataset(
        data=train_volume_ds, patch_iter=patch_func, transform=patch_transform, with_coordinates=False)
    train_loader = DataLoader(
        train_patch_ds,
        batch_size=train_batch_size,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # for validation
    val_patch_ds = monai.data.GridPatchDataset(
        data=val_volume_ds, patch_iter=patch_func, transform=patch_transform, with_coordinates=False)
    val_loader = DataLoader(
        val_patch_ds, #val_volume_ds, 
        num_workers=0, 
        batch_size=val_batch_size,
        pin_memory=torch.cuda.is_available())
    
    if ifcheck:
        check_batch_data(train_loader,val_loader,train_patch_ds,val_volume_ds,train_batch_size,val_batch_size)
    return train_loader,val_loader

def slicer():
    train_volume_ds = monai.data.Dataset(data=train_data, transform=train_transforms)
    val_volume_ds = monai.data.Dataset(data=val_data, transform=train_transforms)
    
    window_width=1
    patch_func = monai.data.PatchIter(
        patch_size=(None, None, window_width),  # dynamic first two dimensions
        start_pos=(0, 0, 0)
    )
    
    if window_width==1:
        patch_transform = Compose(
            [
                SqueezeDim(dim=-1),  # squeeze the last dim
            ]
        ) 
    else:
        patch_transform = None
    
    train_batch_size=8
    val_batch_size=1

    train_dataset = monai.data.GridPatchDataset(data=train_volume_ds, patch_iter=patch_func, transform=patch_transform, with_coordinates=False)
    val_dataset = monai.data.GridPatchDataset(data=val_volume_ds, patch_iter=patch_func, transform=patch_transform, with_coordinates=False)

def myslicesloader(data_pelvis_path,
                   normalize='zscore',
                   train_number=1,
                   val_number=1,
                   train_batch_size=8,
                   val_batch_size=1,
                   saved_name_train='./train_ds_2d.csv',
                   saved_name_val='./val_ds_2d.csv',
                   resized_size=(512,512,None),
                   div_size=(16,16,None),
                   center_crop=20,
                   ifcheck_volume=True,
                   ifcheck_sclices=False,):
    
    # volume-level transforms for both image and label
    train_transforms = get_transforms(normalize,resized_size,div_size,center_crop=center_crop)
    train_ds, val_ds = get_file_list(data_pelvis_path, 
                                     train_number, 
                                     val_number)
    train_volume_ds, val_volume_ds = load_volumes(train_transforms, 
                                                train_ds, 
                                                val_ds, 
                                                saved_name_train, 
                                                saved_name_val,
                                                ifsave=False,
                                                ifcheck=ifcheck_volume)
    train_loader,val_loader = load_batch_slices(train_volume_ds, 
                                                val_volume_ds, 
                                                train_batch_size,
                                                val_batch_size=val_batch_size,
                                                window_width=1,
                                                ifcheck=ifcheck_sclices)
    return train_volume_ds,val_volume_ds,train_loader,val_loader,train_transforms

def len_patchloader(train_volume_ds,train_batch_size):
    slice_number=sum(train_volume_ds[i]['image'].shape[-1] for i in range(len(train_volume_ds)))
    print('total slices in training set:',slice_number)

    import math
    batch_number=sum(math.ceil(train_volume_ds[i]['image'].shape[-1]/train_batch_size) for i in range(len(train_volume_ds)))
    print('total batches in training set:',batch_number)
    return slice_number,batch_number

if __name__ == '__main__':
    dataset_path=r"F:\yang_Projects\Datasets\Task1\pelvis"
    train_volume_ds,_,train_loader,_,_ = myslicesloader(dataset_path,
                    normalize='none',
                    train_number=2,
                    val_number=1,
                    train_batch_size=4,
                    val_batch_size=1,
                    saved_name_train='./train_ds_2d.csv',
                    saved_name_val='./val_ds_2d.csv',
                    resized_size=(512, 512, None),
                    div_size=(16,16,None),
                    ifcheck_volume=False,
                    ifcheck_sclices=False,)
    # test data
    #iter_train = iter(train_loader)
    #batch = next(iter_train)
    #print(batch['image'].shape)

    # slice data
    slice_number=sum(train_volume_ds[i]['image'].shape[-1] for i in range(len(train_volume_ds)))
    print(slice_number)

    # test data loop
    
    #for i, batch in enumerate(train_loader):
    #    print(i, batch['image'].shape)