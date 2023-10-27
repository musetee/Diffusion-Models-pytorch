import torch
import os
from mydataloader.slice_loader import myslicesloader,len_patchloader
def setupdata(args):
    dataset_path=args.dataset_path
    saved_logs_name=f'./logs/{args.run_name}/datalogs'
    os.makedirs(saved_logs_name,exist_ok=True)
    saved_name_train=os.path.join(saved_logs_name, 'train_ds_2d.csv')
    saved_name_val=os.path.join(saved_logs_name, 'val_ds_2d.csv')
    train_volume_ds,val_volume_ds,train_loader,val_loader,train_transforms = myslicesloader(dataset_path,
                    normalize=args.normalize,
                    pad=args.pad,
                    train_number=args.train_number,
                    val_number=args.val_number,
                    train_batch_size=args.batch_size,
                    val_batch_size=1,
                    saved_name_train=saved_name_train,
                    saved_name_val=saved_name_val,
                    resized_size=(args.image_size, args.image_size, None),
                    div_size=(16,16,None),
                    center_crop=args.center_crop,
                    ifcheck_volume=False,
                    ifcheck_sclices=False,)
    slice_number,batch_number =len_patchloader(train_volume_ds,args.batch_size)
    return train_loader,batch_number,val_loader,train_transforms

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="DDPM")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--n_epochs", type=int, default=50) 
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--train_number", type=int, default=10)
    parser.add_argument("--normalize", type=str, default="minmax")
    parser.add_argument("--pad", type=str, default="minimum")
    parser.add_argument("--val_number", type=int, default=1)
    parser.add_argument("--center_crop", type=int, default=0) # set to 0 or -1 means no cropping
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--num_inference_steps", type=int, default=1000)
    parser.add_argument("--dataset_path", type=str, default=r"E:\Projects\yang_proj\Task1\pelvis")
    args = parser.parse_args()
    train_loader,batch_number,val_loader,train_transforms=setupdata(args)
    imgformat='jpg'
    dpi=500
    testfolder='./logs/testimages'
    os.makedirs(testfolder,exist_ok=True)
    for idx, val_check_data in enumerate(val_loader):
        label_imgs, input_imgs = val_check_data["image"], val_check_data["label"]	

        saved_name=os.path.join(testfolder, f'val_{idx}.{imgformat}')
        # save individual images
        import matplotlib.pyplot as plt
        # save output image individually
        title1 = 'MRI'
        fig_mri = plt.figure() #, figsize=(5, 4))
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
                    hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.imshow(input_imgs.squeeze(), cmap='gray')
        plt.savefig(saved_name.replace(f'.{imgformat}',f'_mri.{imgformat}'), format=f'{imgformat}'
                    , bbox_inches='tight', pad_inches=0, dpi=dpi)
        plt.close(fig_mri)
        
        title2 = 'CT'
        fig_ct = plt.figure()
        plt.gca().set_axis_off()
        plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                    hspace = 0, wspace = 0)
        plt.margins(0,0)
        plt.imshow(label_imgs.squeeze(), cmap='gray')
        plt.savefig(saved_name.replace(f'.{imgformat}',f'_ct.{imgformat}'), format=f'{imgformat}'
                    , bbox_inches='tight', pad_inches=0, dpi=dpi)
        plt.close(fig_ct)