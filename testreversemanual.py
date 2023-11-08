from mydataloader.manual_slice_loader import mydataloader
from PIL import Image
import matplotlib
import torch
from monai.transforms.utils import allow_missing_keys_mode
matplotlib.use('Qt5Agg')
import os
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
if __name__ == "__main__":
    dataset_path=r"D:\Projects\data\Task1\pelvis"
    file_path = os.path.join(dataset_path, "1PC098", "ct.nii.gz")
    origimg = nib.load(file_path)
    origimg = origimg.get_fdata()
    print("orig data shape:",origimg.shape) 
    print(f"CT data min and max: {np.min(origimg)}, {np.max(origimg)}")

    save_folder = './logs/test_images'
    os.makedirs(save_folder,exist_ok=True)
    train_loader,val_loader,\
    train_transforms_list,train_transforms,\
    all_slices_train,all_slices_val,\
    shape_list_train,shape_list_val = mydataloader(dataset_path)

    for i, batch in enumerate(val_loader):
            images = batch["image"]
            labels = batch["label"]
            CTimages=images[0,0,:,:,None]
            images=images[0,:,:,:] # 1,1,452,315 -> 1,452,315
            #print("image slice shape",images.shape)
            val_output_dict = {"image": images}
            with allow_missing_keys_mode(train_transforms):
                reversed_images_dict=train_transforms.inverse(val_output_dict)
            images=reversed_images_dict["image"]

            images=images[0,:,:,None]
            
            try:
                volume=torch.cat((volume,images),-1)
                CTdata=torch.cat((CTdata,CTimages),-1)
            except:
                volume=images
                CTdata=CTimages
    print ("original data shape:",CTdata.shape) # [1, 565, 338, 20]
    print ("reversed data shape:",volume.shape) # [452, 315, 104]
    volume = volume.permute(1,0,2)
    ## compare the min and max value of the original and reversed data
    print(f"original data min and max: {torch.min(CTdata)}, {torch.max(CTdata)}")
    print(f"reversed data min and max: {torch.min(volume)}, {torch.max(volume)}")

     # Save as png
    for i in range(volume.shape[-1]):
        if i>=50 and i<=60:
            imgformat='png'
            dpi=300
            saved_name=os.path.join(save_folder,f"{i}.{imgformat}")
            img = volume[:,:,i]

            ## save original image
            img = CTdata[:,:,i]
            fig_ct = plt.figure()
            plt.gca().set_axis_off()
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                        hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.imshow(img, cmap='gray')
            plt.savefig(saved_name, format=f'{imgformat}'
                        , bbox_inches='tight', pad_inches=0, dpi=dpi)
            plt.close(fig_ct)
            
            ## save reversed image
            img = volume[:,:,i]
            fig_ct = plt.figure()
            plt.gca().set_axis_off()
            plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                        hspace = 0, wspace = 0)
            plt.margins(0,0)
            plt.imshow(img, cmap='gray') #.squeeze()
            plt.savefig(saved_name.replace(f'.{imgformat}',f'_reversed.{imgformat}'), format=f'{imgformat}'
                        , bbox_inches='tight', pad_inches=0, dpi=dpi)
            plt.close(fig_ct)   

    volume = volume.cpu().numpy() #
    # Save as nifti
    output_file = os.path.join(save_folder, f"output_for_check")
    nifti_img = nib.Nifti1Image(volume, affine=np.eye(4))  # You can customize the affine transformation matrix if needed
    output_file_name = output_file + '.nii.gz'
    nib.save(nifti_img, output_file_name)

