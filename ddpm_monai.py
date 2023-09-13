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

from generative.inferers import DiffusionInferer
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
from utils import save_images
from mydataloader.slice_loader import myslicesloader,len_patchloader

def setupdata(args):
    #setup_logging(args.run_name)
    device = args.device
    #dataloader = get_data(args)
    dataset_path=args.dataset_path
    train_volume_ds,_,train_loader,val_loader,_ = myslicesloader(dataset_path,
                    normalize=args.normalize,
                    pad=args.pad,
                    train_number=args.train_number,
                    val_number=args.val_number,
                    train_batch_size=args.batch_size,
                    val_batch_size=1,
                    saved_name_train='./train_ds_2d.csv',
                    saved_name_val='./val_ds_2d.csv',
                    resized_size=(args.image_size, args.image_size, None),
                    div_size=(16,16,None),
                    center_crop=args.center_crop,
                    ifcheck_volume=False,
                    ifcheck_sclices=False,)
    slice_number,batch_number =len_patchloader(train_volume_ds,args.batch_size)
    return train_loader,batch_number,val_loader

def checkdata(train_loader):
    check_data = first(train_loader)
    check_image=check_data['label']
    print(f"batch shape: {check_image.shape}")
    #batch_size=check_image.shape[0]
    i=0
    plt.figure(f"image {i}", (6, 6))
    plt.imshow(check_image[i,0], vmin=0, vmax=1, cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

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
            in_channels=1,
            out_channels=1,
            num_channels=(32, 64, 64, 64), # (128, 256, 256),
            attention_levels=(False, False, False, True),
            num_res_blocks=2,
            num_head_channels=32, # 256
        )
        self.model.to(self.device)
        self.scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=args.lr) # 2.5e-5
        self.inferer = DiffusionInferer(self.scheduler)

    def train(self,args,train_loader,batch_number ,val_loader):
        use_pretrained = False
        device = self.device
        model = self.model
        scheduler = self.scheduler
        optimizer = self.optimizer
        inferer = self.inferer
        logger = SummaryWriter(os.path.join("runs", args.run_name))

        if use_pretrained:
            model = torch.hub.load("marksgraham/pretrained_generative_models:v0.2", model="ddpm_2d", verbose=True).to(device)
        else:
            n_epochs = args.n_epochs
            val_interval = args.val_interval
            epoch_loss_list = []
            val_epoch_loss_list = []
            model, optimizer, init_epoch = load_pretrained_model(model, optimizer, args.pretrained_path)

            scaler = GradScaler()
            total_start = time.time()
            for continue_epoch in range(n_epochs):
                epoch = continue_epoch + init_epoch + 1
                model.train()
                epoch_loss = 0
                progress_bar = tqdm(enumerate(train_loader), total=batch_number, ncols=70)
                progress_bar.set_description(f"Epoch {epoch}")
                for step, batch in progress_bar:
                    images = batch["image"].to(device)
                    optimizer.zero_grad(set_to_none=True)

                    with autocast(enabled=True):
                        # Generate random noise
                        noise = torch.randn_like(images).to(device)

                        # Create timesteps
                        timesteps = torch.randint(
                            0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                        ).long()

                        # Get model prediction
                        noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)

                        loss = F.mse_loss(noise_pred.float(), noise.float())

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    epoch_loss += loss.item()
                    logger.add_scalar("train_loss_MSE", loss.item(), global_step=epoch * batch_number + step)

                    progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
                epoch_loss_list.append(epoch_loss / (step + 1))
                logger.add_scalar("train_epoch_loss", epoch_loss / (step + 1), global_step=epoch)
                
                torch.save({'epoch': epoch,
                    'model': model.state_dict(),
                    'opt': optimizer.state_dict()}, 
                    os.path.join("models", args.run_name, f"ckpt{epoch}.pt"))


                if (epoch) % val_interval == 0:
                    model.eval()
                    val_epoch_loss = 0
                    for step, batch in enumerate(val_loader):
                        images = batch["image"].to(device)
                        with torch.no_grad():
                            with autocast(enabled=True):
                                noise = torch.randn_like(images).to(device)
                                timesteps = torch.randint(
                                    0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                                ).long()
                                noise_pred = inferer(inputs=images, diffusion_model=model, noise=noise, timesteps=timesteps)
                                val_loss = F.mse_loss(noise_pred.float(), noise.float())

                        val_epoch_loss += val_loss.item()
                        progress_bar.set_postfix({"val_loss": val_epoch_loss / (step + 1)})
                    val_epoch_loss_list.append(val_epoch_loss / (step + 1))
                    logger.add_scalar("val_epoch_loss", val_epoch_loss / (step + 1), global_step=epoch)

                    # Sampling image during training
                    noise = torch.randn((1, images.shape[1], images.shape[2], images.shape[3]))
                    noise = noise.to(device)
                    scheduler.set_timesteps(num_inference_steps=args.num_inference_steps)
                    with autocast(enabled=True):
                        image = inferer.sample(input_noise=noise, diffusion_model=model, scheduler=scheduler)
                    save_img_folder = os.path.join("results", args.run_name)
                    saved_name=os.path.join(save_img_folder,f"{epoch}.jpg")
                    '''                    
                    fig=plt.figure(figsize=(2, 2))
                    plt.imshow(image[0, 0].cpu(), vmin=0, vmax=1, cmap="gray")
                    plt.tight_layout()
                    plt.axis("off")
                    #plt.show()

                    fig.savefig(os.path.join(save_img_folder,saved_name))
                    plt.close(fig) 
                    '''
                    image = (image.clamp(-1, 1) + 1) / 2 # [-1, 1] -> [0, 1]
                    image = (image * 255).type(torch.uint8)
                    save_images(image,saved_name)
                    
            total_time = time.time() - total_start
            print(f"train completed, total time: {total_time}.")
    
    def testdata(self, train_loader, output_for_check=0,save_folder='test_images'):
        from PIL import Image
        for i, data in enumerate(train_loader):
            images=data['image']
            labels=data['label']
            print(i, ' image: ',images.shape)
            print(i, ' label: ',labels.shape)
            os.makedirs(save_folder,exist_ok=True)
            if output_for_check == 1:
                # save images to file
                for j in range(images.shape[0]):
                    img = images[j,:,:,:]
                    img = img.permute(1,2,0).squeeze().cpu().numpy()
                    img = (img * 255).astype(np.uint8)
                    img = Image.fromarray(img)
                    img.save(f'{save_folder}/'+str(i)+'_'+str(j)+'.png')
            with open(f'{save_folder}/parameter.txt', 'a') as f:
                f.write('image batch:' + str(images.shape)+'\n')
                f.write('label batch:' + str(labels.shape)+'\n')
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
                input_noise=noise, diffusion_model=model, scheduler=scheduler, save_intermediates=True, intermediate_steps=100
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
    parser.add_argument("--run_name", type=str, default="DDPM_monai0912")
    parser.add_argument("--n_epochs", type=int, default=50) 
    parser.add_argument("--val_interval", type=int, default=5)
    parser.add_argument("--train_number", type=int, default=170)
    parser.add_argument("--normalize", type=str, default="minmax")
    parser.add_argument("--pad", type=str, default="minimum")
    parser.add_argument("--val_number", type=int, default=1)
    parser.add_argument("--center_crop", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
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
    args.device = f'cuda:{args.GPU_ID}' if torch.cuda.is_available() else 'cpu' # 0=TitanXP, 1=P5000
    print(torch.cuda.get_device_name(args.GPU_ID))
    os.makedirs(f'./results/{args.run_name}',exist_ok=True)
    os.makedirs(f'./models/{args.run_name}',exist_ok=True)
    train_loader,batch_number,val_loader=setupdata(args)
    #checkdata(train_loader)

    Diffuser=DiffusionModel(args)
    #Diffuser.testdata(train_loader=train_loader,output_for_check=1,save_folder='results/test_images')
    Diffuser.train(args,train_loader,batch_number,val_loader)