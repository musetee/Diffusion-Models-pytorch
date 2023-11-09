from mydataloader.basics import get_transforms, get_file_list, load_volumes, crop_volumes	
from torch.utils.data import DataLoader
import torch
import os
from monai.transforms.utils import allow_missing_keys_mode
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
if __name__ == "__main__":
    dataset_path=r"E:\Projects\yang_proj\Task1\pelvis" #"E:\Projects\yang_proj\Task1\pelvis"  "D:\Projects\data\Task1\pelvis"
    '''
    file_path = os.path.join(dataset_path, "1PC098", "ct.nii.gz")
    ctimg = nib.load(file_path)
    ctimg = ctimg.get_fdata()
    mr_file_path = os.path.join(dataset_path, "1PC098", "mr.nii.gz")
    mrimg = nib.load(mr_file_path)
    mrimg = mrimg.get_fdata()
    print("orig data shape:",ctimg.shape) 
    '''
    #print the min and max value of the original data
    
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
    center_crop=0
    ifcheck_volume=False
    ifcheck_sclices=False

    save_folder = f'./logs/test_{normalize}'
    os.makedirs(save_folder,exist_ok=True)
    # volume-level transforms for both image and label
    train_transforms = get_transforms(normalize,pad,resized_size,div_size)
    train_ds, val_ds = get_file_list(dataset_path, 
                                        train_number, 
                                        val_number,
                                        source='ct',
                                        target='mr',)
    train_crop_ds, val_crop_ds = crop_volumes(train_ds, val_ds,center_crop)

    without_transforms_dataloader=DataLoader(train_crop_ds, batch_size=1)

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
    for idx, checkdata in enumerate(without_transforms_dataloader):
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

    train_ds, val_ds = load_volumes(train_transforms, 
                                    train_crop_ds, val_crop_ds,
                                    train_ds, 
                                    val_ds, 
                                    saved_name_train, 
                                    saved_name_val,
                                    ifsave=False,
                                    ifcheck=ifcheck_volume)
    loader = DataLoader(train_ds, batch_size=1)

    for idx, checkdata in enumerate(loader):
        #transformed_CT=checkdata['image']   
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

        if normalize == 'zscore':
            # reverse the normalization using std and mean
            reversed_ct = reversed_ct*std_list_ct[idx]+mean_list_ct[idx]
            reversed_mri = reversed_mri*std_list_mri[idx]+mean_list_mri[idx]
        elif normalize == 'minmax':
            # reverse the normalization using min and max
            reversed_ct = reversed_ct*(untransformed_CT_max_list[idx]-untransformed_CT_min_list[idx])+untransformed_CT_min_list[idx]
            reversed_mri = reversed_mri*(untransformed_MRI_max_list[idx]-untransformed_MRI_min_list[idx])+untransformed_MRI_min_list[idx]
        elif normalize == 'inputonly':
            reversed_mri = reversed_mri*(untransformed_MRI_max_list[idx]-untransformed_MRI_min_list[idx])+untransformed_MRI_min_list[idx]
        
        ## compare the min and max value of the original and reversed data
        print(f"{idx} untransformed ct data min and max: {untransformed_CT_min_list[idx]}, {untransformed_CT_max_list[idx]}")
        print(f"{idx} transformed ct data min and max: {torch.min(transformed_CT)}, {torch.max(transformed_CT)}")
        print(f"{idx} reversed ct min and max: {torch.min(reversed_ct)}, {torch.max(reversed_ct)}")

        print(f"{idx} untransformed mri data min and max: {untransformed_MRI_min_list[idx]}, {untransformed_MRI_max_list[idx]}")
        print(f"{idx} transformed mri data min and max: {torch.min(transformed_MRI)}, {torch.max(transformed_MRI)}")  
        print(f"{idx} reversed mri min and max: {torch.min(reversed_mri)}, {torch.max(reversed_mri)}")
        # Save as png
        for i in range(reversed_ct.shape[-1]):
            if i>=50 and i<=51:
                imgformat='png'
                dpi=300
                saved_name=os.path.join(save_folder,f"{i}.{imgformat}")
                
                ## save original image
                img_ct = ct_data_list[idx][0,0,:,:,i]
                fig_ct = plt.figure()
                plt.gca().set_axis_off()
                plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                            hspace = 0, wspace = 0)
                plt.margins(0,0)
                plt.imshow(img_ct, cmap='gray')
                plt.savefig(saved_name.replace(f'.{imgformat}',f'_{idx}_original_ct.{imgformat}'), format=f'{imgformat}'
                            , bbox_inches='tight', pad_inches=0, dpi=dpi)
                plt.close(fig_ct)

                ## save transformed image
                img_ct_transformed = transformed_CT[:,:,i]
                fig_ct_transformed = plt.figure()
                plt.gca().set_axis_off()
                plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                            hspace = 0, wspace = 0)
                plt.margins(0,0)
                plt.imshow(img_ct_transformed, cmap='gray')
                plt.savefig(saved_name.replace(f'.{imgformat}',f'_{idx}_transformed_ct.{imgformat}'), format=f'{imgformat}'
                            , bbox_inches='tight', pad_inches=0, dpi=dpi)
                plt.close(fig_ct_transformed)

                ## save reversed image
                img_ct_reversed = reversed_ct[:,:,i]
                fig_ct_reversed = plt.figure()
                plt.gca().set_axis_off()
                plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                            hspace = 0, wspace = 0)
                plt.margins(0,0)
                plt.imshow(img_ct_reversed, cmap='gray') #.squeeze()
                plt.savefig(saved_name.replace(f'.{imgformat}',f'_{idx}_reversed_ct.{imgformat}'), format=f'{imgformat}'
                            , bbox_inches='tight', pad_inches=0, dpi=dpi)
                plt.close(fig_ct_reversed)   

                ## save original image for MRI
                img_mri = mri_data_list[idx][0,0,:,:,i]
                fig_mri = plt.figure()
                plt.gca().set_axis_off()
                plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                            hspace = 0, wspace = 0)
                plt.margins(0,0)
                plt.imshow(img_mri, cmap='gray')
                plt.savefig(saved_name.replace(f'.{imgformat}',f'_{idx}_original_mri.{imgformat}'), format=f'{imgformat}'
                            , bbox_inches='tight', pad_inches=0, dpi=dpi)
                plt.close(fig_mri)

                ## save transformed image
                img_mri_transformed = transformed_MRI[:,:,i]
                fig_mri_transformed = plt.figure()
                plt.gca().set_axis_off()
                plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                            hspace = 0, wspace = 0)
                plt.margins(0,0)
                plt.imshow(img_mri_transformed, cmap='gray')
                plt.savefig(saved_name.replace(f'.{imgformat}',f'_{idx}_transformed_mri.{imgformat}'), format=f'{imgformat}'
                            , bbox_inches='tight', pad_inches=0, dpi=dpi)
                plt.close(fig_mri_transformed)

                ## save reversed image
                img_mri_reversed = reversed_mri[:,:,i]
                fig_mri_reversed = plt.figure()
                plt.gca().set_axis_off()
                plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                            hspace = 0, wspace = 0)
                plt.margins(0,0)
                plt.imshow(img_mri_reversed, cmap='gray') #.squeeze()
                plt.savefig(saved_name.replace(f'.{imgformat}',f'_{idx}_reversed_mri.{imgformat}'), format=f'{imgformat}'
                            , bbox_inches='tight', pad_inches=0, dpi=dpi)
                plt.close(fig_mri_reversed)  

        # pixels' intensity histogram of ct images
        # Flatten the 3D arrays to 1D arrays to calculate the histogram
        flattened_volume1 = ct_data_list[idx].numpy().flatten()
        flattened_volume2 = reversed_ct.numpy().flatten()
        flattened_volume3 = transformed_CT.numpy().flatten()
        print(flattened_volume1.shape)
        print(flattened_volume2.shape)
        # Set up the matplotlib figure and axes
        fig, axs = plt.subplots(3, 1, figsize=(10, 8))
        # Histogram settings
        bins = 256  # Adjust the number of bins for the histogram as needed
        hist_range1 = (np.min(flattened_volume1), 3000)  # Range based on both volumes np.max(flattened_volume1)
        hist_range2 = (np.min(flattened_volume2), 3000)  # Range based on both volumes np.max(flattened_volume2)
        hist_range3 = (np.min(flattened_volume3), np.max(flattened_volume3))  # Range based on both volumes np.max(flattened_volume2)
        # Plot histogram for the first volume
        axs[0].hist(flattened_volume1, bins=bins, range=hist_range1, color='blue', alpha=0.7)
        axs[0].set_title('Histogram of Pixel Intensities for original ct image')
        axs[0].set_xlabel('Pixel intensity')
        axs[0].set_ylabel('Frequency')

        # Plot histogram for the second volume
        axs[1].hist(flattened_volume2, bins=bins, range=hist_range2, color='green', alpha=0.7)
        axs[1].set_title('Histogram of Pixel Intensities for reversed ct image')
        axs[1].set_xlabel('Pixel intensity')
        axs[1].set_ylabel('Frequency')

        # Plot histogram for the third volume
        axs[2].hist(flattened_volume3, bins=bins, range=hist_range3, color='red', alpha=0.7)
        axs[2].set_title('Histogram of Pixel Intensities for transformed ct image')
        axs[2].set_xlabel('Pixel intensity')
        axs[2].set_ylabel('Frequency')

        # Adjust layout for better spacing
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder,f'histograms_{idx}_ct.png'))
        plt.close(fig)


        # pixels' intensity histogram of mri images
        # Flatten the 3D arrays to 1D arrays to calculate the histogram
        flattened_volume1 = mri_data_list[idx].numpy().flatten()
        flattened_volume2 = reversed_mri.numpy().flatten()
        flattened_volume3 = transformed_MRI.numpy().flatten()
        print(flattened_volume1.shape)
        print(flattened_volume2.shape)
        # Set up the matplotlib figure and axes
        fig, axs = plt.subplots(3, 1, figsize=(10, 8))
        # Histogram settings
        bins = 256  # Adjust the number of bins for the histogram as needed
        hist_range1 = (np.min(flattened_volume1), 3000)  # Range based on both volumes np.max(flattened_volume1)
        hist_range2 = (np.min(flattened_volume2), 3000)  # Range based on both volumes np.max(flattened_volume2)
        hist_range3 = (np.min(flattened_volume3), np.max(flattened_volume3))  # Range based on both volumes np.max(flattened_volume2)
        # Plot histogram for the first volume
        axs[0].hist(flattened_volume1, bins=bins, range=hist_range1, color='blue', alpha=0.7)
        axs[0].set_title('Histogram of Pixel Intensities for original mri image')
        axs[0].set_xlabel('Pixel intensity')
        axs[0].set_ylabel('Frequency')

        # Plot histogram for the second volume
        axs[1].hist(flattened_volume2, bins=bins, range=hist_range2, color='green', alpha=0.7)
        axs[1].set_title('Histogram of Pixel Intensities for reversed mri image')
        axs[1].set_xlabel('Pixel intensity')
        axs[1].set_ylabel('Frequency')

        # Plot histogram for the third volume
        axs[2].hist(flattened_volume3, bins=bins, range=hist_range3, color='red', alpha=0.7)
        axs[2].set_title('Histogram of Pixel Intensities for transformed mri image')
        axs[2].set_xlabel('Pixel intensity')
        axs[2].set_ylabel('Frequency')

        # Adjust layout for better spacing
        plt.tight_layout()
        plt.savefig(os.path.join(save_folder,f'histograms_{idx}_mri.png'))
        plt.close(fig)