import os
import shutil
import tempfile
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from monai import transforms
from monai.apps import MedNISTDataset
from monai.config import print_config
from monai.data import CacheDataset, DataLoader
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from monai.transforms.utils import allow_missing_keys_mode
from monai.transforms import SaveImage

from generative.inferers import transformDiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
from utils import save_images
from mydataloader.slice_loader import myslicesloader,len_patchloader
from mydataloader.evaluate import calculate_ssim,calculate_mae,calculate_psnr,output_val_log, val_log,compare_imgs, reverse_transforms

def setupdata(args):
    #setup_logging(args.run_name)
    device = args.device
    #dataloader = get_data(args)
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
    #slice_number,batch_number =len_patchloader(train_volume_ds,args.batch_size)
    batch_number = 10000
    return train_loader,batch_number,val_loader,train_transforms
def checkdata(loader,inputtransforms,output_for_check=1,save_folder='./logs/test_images'):
    from PIL import Image
    import matplotlib
    matplotlib.use('Qt5Agg')
    for i, batch in enumerate(loader):
        #images = batch["image"]
        images = batch["label"]
        labels = batch["label"]
        
        images=images[:,:,:,:,None]
        try:
            volume=torch.cat((volume,images),-1)
        except:
            volume=images

    volume = volume[0,:,:,:,:] #(B,C,H,W,D)    
    # the input into reverse transform should be in form: 20 is the cropped depth
    # (1, 512, 512, 20) -> (1, 452, 315, 5) C,H,W,D
    print (volume.shape)
    val_output_dict = {"image": volume}
    with allow_missing_keys_mode(inputtransforms):
        reversed_images_dict=inputtransforms.inverse(val_output_dict)
    #images=reversed_images_dict["image"]

    for i in range(images.shape[0]):
        print(images.shape)
        imgformat='png'
        dpi=300
        os.makedirs(save_folder,exist_ok=True)
        if output_for_check == 1:
            # save images to file
            for j in range(images.shape[-1]):
                saved_name=os.path.join(save_folder,f"{i}_{j}.{imgformat}")
                img = images[:,:,:,j]
                #img =img.squeeze().cpu().numpy()
                img = img.permute(1,2,0).squeeze().cpu().numpy()
                img = (img * 255).astype(np.uint8)
                img = Image.fromarray(img)
                #img.save(saved_name)

                fig_ct = plt.figure()
                plt.gca().set_axis_off()
                plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0,
                            hspace = 0, wspace = 0)
                plt.margins(0,0)
                plt.imshow(img, cmap='gray') #.squeeze()
                plt.savefig(saved_name.replace(f'.{imgformat}',f'_reversed.{imgformat}'), format=f'{imgformat}'
                            , bbox_inches='tight', pad_inches=0, dpi=dpi)
                plt.close(fig_ct)
        '''
        fig = plt.figure(figsize=(8, 8))
        for j in range(images.shape[0]):
            img = images[j,:,:,:]
            img = img.permute(1,2,0).squeeze().cpu().numpy()
            img = (img * 255).astype(np.uint8)
            ax = fig.add_subplot(4, 4, j + 1)
            ax.imshow(img, cmap='gray')
            ax.axis('off')
            plt.show()
        with open(f'{save_folder}/parameter.txt', 'a') as f:
            f.write('image batch:' + str(images.shape)+'\n')
            f.write('label batch:' + str(labels)+'\n')
            f.write('\n')
        '''

import torch.nn as nn
def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
def load_pretrained_model(model, opt, pretrained_path=None):
    if pretrained_path is not None:
        latest_ckpt=pretrained_path
        loaded_state = torch.load(latest_ckpt)
        print(f'use pretrained model: {latest_ckpt}') 
        if 'epoch' in loaded_state:
            init_epoch=loaded_state["epoch"] # load or manually set
            print(f'continue from epoch {init_epoch}') 
            #init_epoch = int(input('Enter epoch number: '))
        else:
            print('no epoch information in the checkpoint file')
            init_epoch = int(input('Enter epoch number: '))
        model.load_state_dict(loaded_state["model"]) #
        opt.load_state_dict(loaded_state["opt"])
    else:
        init_epoch=0
        #model = model.apply(weights_init)
        print(f'start new training') 
    return model, opt, init_epoch

class DiffusionModel:
    def __init__(self,args):
        self.device = torch.device(args.device)
        self.model = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=2,
            out_channels=1,
            num_channels=(64, 128, 256, 256), # (128, 256, 256), (32, 64, 64, 64)
            attention_levels=(False, False, False, True),
            num_res_blocks=2,
            num_head_channels=32, # 256
        )
        self.model.to(self.device)
        self.scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=args.lr) # 2.5e-5
        self.inferer = transformDiffusionInferer(self.scheduler)
        self.saved_results_name=f'./logs/{args.run_name}/results'
        self.saved_models_name=f'./logs/{args.run_name}/models'
        self.saved_runs_name=f'./logs/{args.run_name}/runs'
        os.makedirs(self.saved_results_name, exist_ok=True)
        os.makedirs(self.saved_models_name, exist_ok=True)
    def _train(images, labels, model, inferer, optimizer,scaler,logger,epoch,step,device):
        # the process inside one epoch loop
        with autocast(enabled=True):
            # Generate random noise
            noise = torch.randn_like(images).to(device)
            # noise = torch.cat((noise,labels),1)
            # print(noise.shape)
            # Create timesteps
            timesteps = torch.randint(
                0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
            ).long()

            # Get model prediction
            noise_pred = inferer(inputs=images, orig_image=labels, diffusion_model=model, noise=noise, timesteps=timesteps)

            loss = F.mse_loss(noise_pred.float(), noise.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        epoch_loss += loss.item()
        logger.add_scalar("train_loss_MSE", loss.item(), global_step=epoch * batch_number + step)
    def train(self,args,train_loader,batch_number ,val_loader):
        use_pretrained = False
        device = self.device
        model = self.model
        scheduler = self.scheduler
        optimizer = self.optimizer
        inferer = self.inferer
        logger = SummaryWriter(self.saved_runs_name)

        if use_pretrained:
            model = torch.hub.load("marksgraham/pretrained_generative_models:v0.2", model="ddpm_2d", verbose=True).to(device)
        else:
            n_epochs = args.n_epochs
            val_interval = args.val_interval
            epoch_loss_list = []
            model, optimizer, init_epoch = load_pretrained_model(model, optimizer, args.pretrained_path)

            scaler = GradScaler()
            total_start = time.time()
            for continue_epoch in range(n_epochs):
                epoch = continue_epoch + init_epoch + 1
                epoch_num_total = n_epochs + init_epoch
                print("-" * 10)
                print(f"epoch {epoch}/{epoch_num_total}")
                model.train()
                epoch_loss = 0
                progress_bar = tqdm(enumerate(train_loader), total=batch_number, ncols=70)
                progress_bar.set_description(f"Epoch {epoch}")
                for step, batch in enumerate(train_loader): #progress_bar
                    images = batch["image"].to(device) # CT image
                    labels = batch["label"].to(device) # MRI image
                    optimizer.zero_grad(set_to_none=True)

                    with autocast(enabled=True):
                        # Generate random noise
                        noise = torch.randn_like(images).to(device)
                        # noise = torch.cat((noise,labels),1)
                        # print(noise.shape)
                        # Create timesteps
                        timesteps = torch.randint(
                            0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                        ).long()

                        # Get model prediction
                        noise_pred = inferer(inputs=images, orig_image=labels, diffusion_model=model, noise=noise, timesteps=timesteps)
                        loss = F.mse_loss(noise_pred.float(), noise.float())
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    epoch_loss += loss.item()
                    logger.add_scalar("train_loss_MSE", loss.item(), global_step=epoch * batch_number + step)

                    progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
                epoch_loss_list.append(epoch_loss / (step + 1))
                logger.add_scalar("train_epoch_loss", epoch_loss / (step + 1), global_step=epoch)
                

                model_save_path = os.path.join(self.saved_models_name, f"model_{epoch}.pt")
                torch.save({'epoch': epoch,
                    'model': model.state_dict(),
                    'opt': optimizer.state_dict()}, model_save_path
                    )

                val_epoch_loss_list = []
                if (epoch) % val_interval == 0:
                    model.eval()
                    val_epoch_loss = 0
                    for step, batch in enumerate(val_loader):
                        if step > 50 and step <53:
                            targets = batch["image"].to(device) # CT image
                            labels = batch["label"].to(device) # MRI image
                            with torch.no_grad():
                                with autocast(enabled=True):
                                    noise = torch.randn_like(targets).to(device)
                                    #noise_extra_channel = torch.cat((noise,labels),1)
                                    timesteps = torch.randint(
                                        0, inferer.scheduler.num_train_timesteps, (targets.shape[0],), device=targets.device
                                    ).long()
                                    noise_pred = inferer(inputs=targets, orig_image=labels, diffusion_model=model, noise=noise, timesteps=timesteps)
                                    val_loss = F.mse_loss(noise_pred.float(), noise.float())
                                    image_loss,_,_ = self._sample(model,
                                                                  targets=targets,
                                                                  inputs=labels, 
                                                                  inferer=inferer,
                                                                  scheduler=scheduler,
                                                                  epoch=epoch,step=step, 
                                                                  device=device)
                            val_epoch_loss += val_loss.item()
                            progress_bar.set_postfix({"val_loss": val_epoch_loss / (step + 1)})
                    val_epoch_loss_list.append(val_epoch_loss / (step + 1))
                    logger.add_scalar("val_epoch_loss", val_epoch_loss / (step + 1), global_step=epoch)
                    logger.add_scalar("img_epoch_loss", image_loss, global_step=epoch)
            total_time = time.time() - total_start
            print(f"train completed, total time: {total_time}.")    
    def _sample(self, model,
                targets, inputs, 
                inferer, scheduler,
                epoch, step=0, 
                device="cuda", i=0, 
                save_imgs=True):
        # Sampling
        # targets: CT image, which is  the target
        # inputs: MRI image, which is to be converted to CT image
        noise = torch.randn((1, targets.shape[1], targets.shape[2], targets.shape[3])) # B,C,H,W
        noise = noise.to(device)
        scheduler.set_timesteps(num_inference_steps=args.num_inference_steps)

        targets_single = targets[i,:,:,:]
        targets_single = targets_single[None,:,:,:]                
        inputs_single = inputs[i,:,:,:]
        inputs_single = inputs_single[None,:,:,:]
       
        with autocast(enabled=True):
            image = inferer.sample(input_noise=noise, input_image=inputs_single, diffusion_model=model, scheduler=scheduler)
        image_loss = F.mse_loss(image,targets_single)
        saved_name=os.path.join(self.saved_results_name,f"{epoch}_{step}.jpg")
        '''
        #image = (image.clamp(-1, 1) + 1) / 2 # [-1, 1] -> [0, 1]
        #image = image * 255
        #image = (image * 255).type(torch.uint8)
        
        # if z-score normalization is used, then the image should be converted back to the original range
        from monai.transforms import (
        Compose, NormalizeIntensity)
        normtransform = Compose(
            [NormalizeIntensity(nonzero=False, channel_wise=True)]
            )
        image.applied_operations = targets_single.applied_operations
        with allow_missing_keys_mode(normtransform):
            image=normtransform.inverse(image)
        '''
        targets_single = targets_single.detach().cpu()
        inputs_single = inputs_single.detach().cpu()
        image = image.detach().cpu()

        if save_imgs:                         
            compare_imgs(input_imgs=inputs_single, 
                         target_imgs=targets_single,
                         fake_imgs=image,
                         saved_name=saved_name)
            
        return image_loss,image,targets_single  
    def _test_nifti(self, args, val_loader, val_transforms):
        import nibabel as nib
        device = self.device
        model = self.model
        scheduler = self.scheduler
        optimizer = self.optimizer
        inferer = self.inferer
        model, optimizer, init_epoch = load_pretrained_model(model, optimizer, args.pretrained_path)
        self.saved_logs_name=self.saved_results_name.replace('results','datalogs')
        output_file=os.path.join(self.saved_results_name, f"test")

        total_start = time.time()
        epoch = init_epoch
        model.eval()
        image_losses=[]
        #batch_example=next(iter(val_loader))
        #image_example = batch_example["image"]
        #image_volume=torch.zeros((1,image_example.shape[1],image_example.shape[2],image_example.shape[3]))
        ssim_sum = 0
        psnr_sum = 0
        mae_sum = 0
        metric_sum=0
        for step, batch in enumerate(val_loader):
            print(step)
            targets = batch["image"].to(device) # CT image
            inputs = batch["label"].to(device) # MRI image
            print("original image shape:", targets.shape)
            image_loss, generated_image,orig_image = self._sample(model=model, 
                                                                targets=targets, 
                                                                inputs=inputs, 
                                                                inferer=inferer,
                                                                scheduler=scheduler,
                                                                epoch=epoch, step=step, 
                                                                device=device, i=0,
                                                                save_imgs=True) # 0 because val_loader batch_size=1
            val_metrices, infer_log_file = val_log(epoch, step, generated_image, orig_image, self.saved_logs_name)
            #if val_metrices['ssim']>0.3:
            ssim_sum = ssim_sum + val_metrices['ssim']
            psnr_sum = psnr_sum + val_metrices['psnr']
            mae_sum = mae_sum + val_metrices['mae'] 
            ssim_overall = ssim_sum / (step + 1)
            psnr_overall = psnr_sum / (step + 1)
            mae_overall = mae_sum / (step + 1)
            print("ssim_overall:", ssim_overall)
            print("psnr_overall:", psnr_overall)
            print("mae_overall:", mae_overall)
            with open(infer_log_file, 'a') as f: # append mode
                f.write(f'over all metrices, SSIM: {ssim_overall}, MAE: {mae_overall}, PSNR: {psnr_overall}\n')
           
            print("img_epoch_loss", image_loss)

            print("generated image shape:", generated_image.shape)
           
            generated_image=generated_image.unsqueeze(-1)
            print("unsqueezed image shape:", generated_image.shape)
            image_losses.append(image_loss)
            try:
                image_volume=torch.cat((image_volume,generated_image),-1)#pay attention to the order!
            except:
                image_volume=generated_image
        
        reversed_image = reverse_transforms(output_images=image_volume, 
                                            orig_images=inputs, # output reverse should according to the original inputs MRI
                                            transforms=val_transforms) 
        
        #reversed_image = reversed_image.unsqueeze(0)
        #print("reversed image shape:", reversed_image.shape)
        #SaveImage(output_dir=output_file, resample=True)(reversed_image.detach().cpu())
        
        # Create a NIfTI image object
        reversed_image = torch.squeeze(reversed_image.detach().cpu()).numpy()
        reversed_image = np.rot90(reversed_image, axes=(0, 1), k=2)
        nifti_img = nib.Nifti1Image(reversed_image, affine=np.eye(4))  # You can customize the affine transformation matrix if needed
        # Save the NIfTI image to a file
        output_file_name = output_file + '.nii.gz'
        nib.save(nifti_img, output_file_name)

        '''
        image_volume = torch.squeeze(image_volume).numpy()
        image_volume= np.transpose(image_volume, (1, 2, 0))
        image_volume = np.rot90(image_volume, axes=(0, 1), k=3)  # k=1 means rotate counterclockwise 90 degrees
        #print(image_volume.shape)

        # Create a NIfTI image object
        nifti_img = nib.Nifti1Image(image_volume, affine=np.eye(4))  # You can customize the affine transformation matrix if needed
        # Save the NIfTI image to a file
        nib.save(nifti_img, output_file)
        total_time = time.time() - total_start
        print(f"train completed, total time: {total_time}.")
        '''
    def test(self, args, val_loader):
        use_pretrained = False
        device = self.device
        model = self.model
        scheduler = self.scheduler
        optimizer = self.optimizer
        inferer = self.inferer
        if use_pretrained:
            model = torch.hub.load("marksgraham/pretrained_generative_models:v0.2", model="ddpm_2d", verbose=True).to(device)
        else:
            model, optimizer, init_epoch = load_pretrained_model(model, optimizer, args.pretrained_path)

            total_start = time.time()
            epoch = init_epoch + 1
            model.eval()
            batch = next(iter(val_loader))
            images = batch["image"].to(device)
            inputs = batch["label"].to(device)
            image_loss,image,_ = self._sample(model,images,inputs, inferer,scheduler,epoch, device)

            print("img_epoch_loss", image_loss, "global_step:", epoch)
            total_time = time.time() - total_start
            print(f"train completed, total time: {total_time}.")
    def testdata(self, train_loader, output_for_check=0,save_folder='test_images'):
        from PIL import Image
        for i, data in enumerate(train_loader):
            targets=data['image']
            inputs=data['label']
            print(i, ' image: ',targets.shape)
            print(i, ' label: ',inputs.shape)
            os.makedirs(save_folder,exist_ok=True)
            if output_for_check == 1:
                # save images to file
                for j in range(targets.shape[0]):
                    img = targets[j,:,:,:]
                    img = img.permute(1,2,0).squeeze().cpu().numpy()
                    img = (img * 255).astype(np.uint8)
                    img = Image.fromarray(img)
                    img.save(f'{save_folder}/'+str(i)+'_'+str(j)+'.png')
            with open(f'{save_folder}/parameter.txt', 'a') as f:
                f.write('targets batch:' + str(targets.shape)+'\n')
                f.write('inputs batch:' + str(inputs.shape)+'\n')
                f.write('\n')
    def plotchain(self):
        device = self.device
        model = self.model
        scheduler = self.scheduler
        inferer = self.inferer
        model.eval()
        noise = torch.randn((1, 1, 64, 64))
        noise = noise.to(device)
        scheduler.set_timesteps(num_inference_steps=1000)
        with autocast(enabled=True):
            image, intermediates = inferer.sample(
                input_noise=noise,   input_image=noise,  diffusion_model=model, scheduler=scheduler, save_intermediates=True, intermediate_steps=100
            )

        chain = torch.cat(intermediates, dim=-1)
        num_images = chain.shape[-1]

        plt.style.use("default")
        plt.imshow(chain[0, 0].cpu(), vmin=0, vmax=1, cmap="gray")
        plt.tight_layout()
        plt.axis("off")
        plt.show()

if __name__ == "__main__":
    set_determinism(42)
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", type=str, default="DDPM")
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--n_epochs", type=int, default=50) 
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--train_number", type=int, default=170)
    parser.add_argument("--normalize", type=str, default="minmax")
    parser.add_argument("--pad", type=str, default="minimum")
    parser.add_argument("--val_number", type=int, default=1)
    parser.add_argument("--center_crop", type=int, default=20) # set to 0 or -1 means no cropping
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--pretrained_path", type=str, default=None)
    parser.add_argument("--lr", type=float, default=2.5e-5)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--num_inference_steps", type=int, default=1000)
    parser.add_argument("--dataset_path", type=str, default=r"F:\yang_Projects\Datasets\Task1\pelvis")
    # r"F:\yang_Projects\Datasets\Task1\pelvis" 
    # r"C:\Users\56991\Projects\Datasets\Task1\pelvis" 
    # r"D:\Projects\data\Task1\pelvis" 
    parser.add_argument("--GPU_ID", type=int, default=0)

    args = parser.parse_args()
    args.device = f'cuda:{args.GPU_ID}' if torch.cuda.is_available() else 'cpu'
    print(torch.cuda.get_device_name(args.GPU_ID))
    train_loader,batch_number,val_loader,train_transforms=setupdata(args)
    logs_name=f'./logs/{args.run_name}'	
    os.makedirs(logs_name,exist_ok=True)
    if args.mode == "train":
        Diffuser=DiffusionModel(args)
        #Diffuser.testdata(train_loader=train_loader,output_for_check=1,save_folder='results/test_images')
        Diffuser.train(args,train_loader,batch_number,val_loader)
    elif args.mode == "checkdata":
        #checkdata(train_loader)
        checkdata(val_loader,train_transforms)
    elif args.mode == "test":
        Diffuser=DiffusionModel(args)
        #Diffuser.testdata(train_loader=train_loader,output_for_check=1,save_folder='results/test_images')
        Diffuser.test(args,val_loader)
    elif args.mode == "testnifti":
        current_time=time.time()
        Diffuser=DiffusionModel(args)
        Diffuser._test_nifti(args,val_loader,train_transforms)