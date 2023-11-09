from mydataloader.basics import get_transforms, get_file_list, load_volumes, crop_volumes
from torch.utils.data import DataLoader
import torch
import os
from monai.transforms.utils import allow_missing_keys_mode
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib

# Define function to save images
def save_image(image, filename, idx, title, dpi=300):
    plt.figure()
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title(title)
    plt.savefig(f"{filename}_{idx}.png", format='png', bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close()

# Define function to plot histograms
def plot_histogram(data, title, ax, color='blue', alpha=0.7):
    bins = 256
    ax.hist(data.flatten(), bins=bins, color=color, alpha=alpha)
    ax.set_title(title)
    ax.set_xlabel('Pixel intensity')
    ax.set_ylabel('Frequency')

# Arrange three histograms
def arrange_histograms(original, transformed, reversed, mode='ct'):
    # Plot histograms
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    plot_histogram(original, f'Histogram for original {mode}', axs[0],color='red')
    plot_histogram(transformed, f'Histogram for transformed {mode}', axs[1],color='green')
    plot_histogram(reversed, f'Histogram for reversed {mode}', axs[2],color='blue')
    # Show and save the histogram figure
    plt.tight_layout()
    plt.savefig(os.path.join(save_folder, f"{idx}_histograms_{mode}.png"), dpi=300)
    plt.close(fig)


# Define function to normalize and reverse normalize
def normalize_data(tensor, mean=None, std=None, min_val=None, max_val=None, mode='zscore'):
    if mode == 'zscore':
        return (tensor - mean) / std if mean is not None and std is not None else tensor
    elif mode == 'minmax': # for minmax to -1 and 1
        return (tensor - min_val) / (max_val - min_val) if min_val is not None and max_val is not None else tensor
    return tensor

# Define function to reverse normalization
def reverse_normalize_data(tensor, mean=None, std=None, min_val=None, max_val=None, mode='zscore'):
    if mode == 'zscore':
        return tensor * std + mean if mean is not None and std is not None else tensor
    elif mode == 'minmax':
        return (tensor+1) /2 * (max_val - min_val) + min_val if min_val is not None and max_val is not None else tensor
    return tensor

# Normalization settings
normalization_methods = {
    'zscore': {'apply': normalize_data, 'reverse': reverse_normalize_data},
    'minmax': {'apply': normalize_data, 'reverse': reverse_normalize_data},
    'inputonly': {'apply': lambda x: x, 'reverse': reverse_normalize_data}
}


# Other settings
dataset_path = r"E:\Projects\yang_proj\Task1\pelvis"
normalize = 'minmax'
pad = 'minimum'
train_number = 1
val_number = 1
train_batch_size = 8
val_batch_size = 1
saved_name_train = './train_ds_2d.csv'
saved_name_val = './val_ds_2d.csv'
resized_size = (512, 512, None)
div_size = (16, 16, None)
center_crop = 0
ifcheck_volume = False
ifcheck_sclices = False
save_folder = f'./logs/test_{normalize}'
os.makedirs(save_folder, exist_ok=True)

# Define your transforms, file lists, dataloaders, etc...
# volume-level transforms for both image and label
train_transforms = get_transforms(normalize,pad,resized_size,div_size)
train_ds, val_ds = get_file_list(dataset_path, 
                                    train_number, 
                                    val_number,
                                    source='ct',
                                    target='mr',)
train_crop_ds, val_crop_ds = crop_volumes(train_ds, val_ds,center_crop)
untransformed_loader=DataLoader(train_crop_ds, batch_size=1)
train_ds, val_ds = load_volumes(train_transforms, 
                                train_crop_ds, val_crop_ds,
                                train_ds, 
                                val_ds, 
                                saved_name_train, 
                                saved_name_val,
                                ifsave=False,
                                ifcheck=ifcheck_volume)
transformed_loader = DataLoader(train_ds, batch_size=1)

ct_data_list=[]
mri_data_list=[]
mean_list_ct=[]
std_list_ct=[]
mean_list_mri=[]
std_list_mri=[]
ct_shape_list=[]
mri_shape_list=[]
untransformed_CT_min_list=[]
untransformed_CT_max_list=[]
untransformed_MRI_min_list=[]
untransformed_MRI_max_list=[]
# calculate the mean and std of the original data
for idx, checkdata in enumerate(untransformed_loader):
    untransformed_CT=checkdata['image']
    untransformed_MRI=checkdata['label']

    mean_ct=torch.mean(untransformed_CT.float())
    std_ct=torch.std(untransformed_CT.float())
    mean_list_ct.append(mean_ct)
    std_list_ct.append(std_ct)

    mean_mri=torch.mean(untransformed_MRI.float())
    std_mri=torch.std(untransformed_MRI.float())
    mean_list_mri.append(mean_mri)
    std_list_mri.append(std_mri)

    ct_shape_list.append(untransformed_CT.shape)
    mri_shape_list.append(untransformed_MRI.shape)
    untransformed_CT_min_list.append(torch.min(untransformed_CT))
    untransformed_CT_max_list.append(torch.max(untransformed_CT))
    untransformed_MRI_min_list.append(torch.min(untransformed_MRI))
    untransformed_MRI_max_list.append(torch.max(untransformed_MRI))
    ct_data_list.append(untransformed_CT)
    mri_data_list.append(untransformed_MRI)

# Process datasets
for idx, checkdata in enumerate(transformed_loader):
    # Get your data (untransformed and transformed)
    transformed_CT=checkdata['image'] 
    transformed_MRI=checkdata['label']

    dict = {"image": transformed_CT[0,:,:,:,:], "label": transformed_MRI[0,:,:,:,:]} 
    with allow_missing_keys_mode(train_transforms):
        reversed_dict=train_transforms.inverse(dict)
    reversed_ct=reversed_dict["image"]
    reversed_mri=reversed_dict["label"]

    print(f"{idx} original CT data shape:",ct_shape_list[idx])
    print(f"{idx} transformed CT data shape:", transformed_CT.shape) 
    print (f"{idx} reversed CT shape:",reversed_ct.shape)
    print(f"{idx} original MRI data shape:",mri_shape_list[idx])
    print(f"{idx} transformed MRI data shape:", transformed_MRI.shape)
    print(f"{idx} transformed MRI data shape:", transformed_MRI.shape)

    reversed_ct = reversed_ct.squeeze().permute(1,0,2) #[452, 315, 104] -> [315, 452, 104]
    transformed_CT = transformed_CT.squeeze().squeeze().permute(1,0,2) #[452, 315, 104] -> [315, 452, 104]
    reversed_mri = reversed_mri.squeeze().permute(1,0,2) #[452, 315, 104] -> [315, 452, 104]
    transformed_MRI = transformed_MRI.squeeze().squeeze().permute(1,0,2) #[452, 315, 104] -> [315, 452, 104]

    # Normalize and reverse normalization
    norm_method = normalization_methods[normalize]
    reversed_ct = norm_method['reverse'](transformed_CT, mean=mean_list_ct[idx], std=std_list_ct[idx], 
                                         min_val=untransformed_CT_min_list[idx], max_val=untransformed_CT_max_list[idx], mode=normalize)
    reversed_mri = norm_method['reverse'](transformed_MRI, mean=mean_list_mri[idx], std=std_list_mri[idx],
                                          min_val=untransformed_MRI_min_list[idx], max_val=untransformed_MRI_max_list[idx], mode=normalize)

    # Save images
    for i in range(reversed_ct.shape[-1]):
        if 50 <= i <= 51:
            save_image(ct_data_list[idx][0, 0, :, :, i], os.path.join(save_folder, f"{i}_original_ct"), idx, "Original CT")
            save_image(transformed_CT[:, :, i], os.path.join(save_folder, f"{i}_transformed_ct"), idx, "Transformed CT")
            save_image(reversed_ct[:, :, i], os.path.join(save_folder, f"{i}_reversed_ct"), idx, "Reversed CT")
            # ... Repeat for MRI images
            save_image(mri_data_list[idx][0, 0, :, :, i], os.path.join(save_folder, f"{i}_original_mri"), idx, "Original MRI")
            save_image(transformed_MRI[:, :, i], os.path.join(save_folder, f"{i}_transformed_mri"), idx, "Transformed MRI")
            save_image(reversed_mri[:, :, i], os.path.join(save_folder, f"{i}_reversed_mri"), idx, "Reversed MRI")

    ## compare the min and max value of the original and reversed data
    print(f"{idx} untransformed ct data min and max: {untransformed_CT_min_list[idx]}, {untransformed_CT_max_list[idx]}")
    print(f"{idx} transformed ct data min and max: {torch.min(transformed_CT)}, {torch.max(transformed_CT)}")
    print(f"{idx} reversed ct min and max: {torch.min(reversed_ct)}, {torch.max(reversed_ct)}")

    print(f"{idx} untransformed mri data min and max: {untransformed_MRI_min_list[idx]}, {untransformed_MRI_max_list[idx]}")
    print(f"{idx} transformed mri data min and max: {torch.min(transformed_MRI)}, {torch.max(transformed_MRI)}")  
    print(f"{idx} reversed mri min and max: {torch.min(reversed_mri)}, {torch.max(reversed_mri)}")
    arrange_histograms(ct_data_list[idx].numpy(), transformed_CT.numpy(), reversed_ct.numpy(), mode='ct')
    arrange_histograms(mri_data_list[idx].numpy(), transformed_MRI.numpy(), reversed_mri.numpy(), mode='mri')

