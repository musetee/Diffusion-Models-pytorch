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
                    normalize='zscore',
                    train_number=args.train_number,
                    val_number=args.val_number,
                    train_batch_size=args.batch_size,
                    val_batch_size=1,
                    saved_name_train='./train_ds_2d.csv',
                    saved_name_val='./val_ds_2d.csv',
                    resized_size=(args.image_size, args.image_size, None),
                    div_size=(16,16,None),
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

class DiffusionModel:
    def __init__(self,args):
        self.device = torch.device(args.device)
        self.model = DiffusionModelUNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=1,
            num_channels=(128, 256, 256),
            attention_levels=(False, False, True),
            num_res_blocks=1,
            num_head_channels=256,
        )
        self.model.to(self.device)
        self.scheduler = DDPMScheduler(num_train_timesteps=1000)
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=2.5e-5)
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

            scaler = GradScaler()
            total_start = time.time()
            for epoch in range(n_epochs):
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

                if (epoch + 1) % val_interval == 0:
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
                    scheduler.set_timesteps(num_inference_steps=1000)
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
    args = parser.parse_args()
    args.run_name = "DDPM_monai_1"
    args.n_epochs=5
    args.val_interval=1
    args.train_number = 1
    args.val_number = 1
    args.batch_size = 1

    args.image_size = 512

    args.dataset_path = r"D:\Projects\data\Task1\pelvis"
    # r"F:\yang_Projects\Datasets\Task1\pelvis" 
    # r"C:\Users\56991\Projects\Datasets\Task1\pelvis" 
    # r"D:\Projects\data\Task1\pelvis" 
    GPU_ID = 0
    args.device = f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu' # 0=TitanXP, 1=P5000
    print(torch.cuda.get_device_name(GPU_ID))
    os.makedirs(f'./results/{args.run_name}',exist_ok=True)
    os.makedirs(f'./models/{args.run_name}',exist_ok=True)
    train_loader,batch_number,val_loader=setupdata(args)
    #checkdata(train_loader)
    Diffuser=DiffusionModel(args)
    Diffuser.train(args,train_loader,batch_number,val_loader)