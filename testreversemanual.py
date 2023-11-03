from mydataloader.manual_slice_loader import mydataloader
from PIL import Image
import matplotlib
import torch
from monai.transforms.utils import allow_missing_keys_mode
matplotlib.use('Qt5Agg')
if __name__ == "__main__":
    dataset_path="D:\Projects\data\Task1\pelvis"

    train_loader,val_loader,\
    train_transforms_list,train_transforms,\
    all_slices_train,all_slices_val,\
    shape_list_train,shape_list_val = mydataloader(dataset_path)

    for i, batch in enumerate(val_loader):
            images = batch["image"]
            labels = batch["label"]
            images=images[0,:,:,:]
            #print(images.shape)
            val_output_dict = {"image": images}
            with allow_missing_keys_mode(train_transforms):
                reversed_images_dict=train_transforms.inverse(val_output_dict)
            images=reversed_images_dict["image"]
            #print(images.shape)
            #images=images[:,:,:,None]
            try:
                volume=torch.cat((volume,images),0)
            except:
                volume=images
    print (volume.shape)