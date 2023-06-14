import os
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import optim
from utils import *
from modules import UNet
import logging
from torch.utils.tensorboard import SummaryWriter

logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")

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
            print(f'continue training from epoch {init_epoch}') 
            #init_epoch = int(input('Enter epoch number: '))
        else:
            print('no epoch information in the checkpoint file')
            init_epoch = int(input('Enter epoch number: '))
        model.load_state_dict(loaded_state["model"]) #
        opt.load_state_dict(loaded_state["opt"])
    else:
        init_epoch=0
        model = model.apply(weights_init)
        print(f'start new training') 
    return model, opt, init_epoch

class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, img_channel=1, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        self.img_channel = img_channel

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    # return beta
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    # forward diffusion
    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None] # [:, None, None, None] add 4 dims
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n, epoch, save_img_folder):
        logging.info(f"Sampling {n} new images....")
        model.eval()
        img_samples = []
        with torch.no_grad():
            x = torch.randn((n, self.img_channel, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps+1)), position=0):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                
                if i % 100 == 0:
                    #img = (x.clamp(-1, 1) + 1) / 2 # [-1, 1] -> [0, 1]
                    #img = (img * 255).type(torch.uint8)
                    img=x
                    img_samples.append(img.detach().cpu().numpy())

        #model.train() 
        x = (x.clamp(-1, 1) + 1) / 2 # [-1, 1] -> [0, 1]
        x = (x * 255).type(torch.uint8)

        num_images = len(img_samples)
        # after all the sampling steps (10 steps), save the images
        titles = [i for i in range(0,num_images)]
        fig,axs=plt.subplots(1, int(num_images), figsize=(20,2))
        cnt = 0
        for j in range(num_images):
            #print(gen_imgs[cnt].shape)
            axs[j].imshow(img_samples[cnt].squeeze(), cmap='gray') # show the process of generating images from left to right
            axs[j].set_title(titles[j])
            axs[j].axis('off')
            cnt += 1
        os.makedirs(save_img_folder,exist_ok=True)
        saved_name=f"{epoch}.jpg"
        fig.savefig(os.path.join(save_img_folder,saved_name))
        #plt.show()   
        plt.close(fig) 

        saved_name_final=f"{epoch}_final.jpg"  
        save_images(x, os.path.join(save_img_folder,saved_name_final))
        return x


from my_dataset import myslicesloader
def train(args):
    #setup_logging(args.run_name)
    device = args.device
    #dataloader = get_data(args)
    dataset_path=args.dataset_path
    train_volume_ds,_,train_loader,_,_ = myslicesloader(dataset_path,
                    normalize='none',
                    train_number=args.train_number,
                    val_number=1,
                    train_batch_size=args.batch_size,
                    val_batch_size=1,
                    saved_name_train='./train_ds_2d.csv',
                    saved_name_val='./val_ds_2d.csv',
                    resized_size=(args.image_size, args.image_size, None),
                    div_size=(16,16,None),
                    ifcheck_volume=False,
                    ifcheck_sclices=False,)
    dataloader=train_loader
    #l = len(dataloader)
    l=1000 # only first test

    model = UNet(c_in=1, c_out=1,time_dim=args.time_dim, depth=args.UNet_depth).to(device)
    # print parameter number 
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    diffusion = Diffusion(noise_steps=args.noise_steps, img_size=args.image_size, device=device)
    logger = SummaryWriter(os.path.join("runs", args.run_name))
    model, optimizer, init_epoch = load_pretrained_model(model, optimizer, args.pretrained_path)

    for continue_epoch in range(args.epochs):
        epoch = continue_epoch + init_epoch + 1
        logging.info(f"Starting epoch {epoch}:")
        pbar = tqdm(dataloader)
        
        for i, images in enumerate(pbar): #(images, _)
            images = images["image"].to(device)
            t = diffusion.sample_timesteps(images.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(images, t)
            model.train() 
            predicted_noise = model(x_t, t)
            loss = mse(noise, predicted_noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_postfix(MSE=loss.item())
            logger.add_scalar("MSE", loss.item(), global_step=epoch * l + i)

        if epoch % args.sample_interval == 0:
            save_img_folder = os.path.join("results", args.run_name)
            sampled_images = diffusion.sample(model, n=images.shape[0], epoch=epoch, save_img_folder=save_img_folder)
            
            
        torch.save({'epoch': epoch,
            'model': model.state_dict(),
            'opt': optimizer.state_dict()}, 
            os.path.join("models", args.run_name, f"ckpt{epoch}.pt"))

def inference(args):
    #setup_logging(args.run_name)
    device = args.device
    #dataloader = get_data(args)
    dataset_path=args.dataset_path
    train_volume_ds,_,train_loader,_,_ = myslicesloader(dataset_path,
                    normalize='none',
                    train_number=args.train_number,
                    val_number=1,
                    train_batch_size=args.batch_size,
                    val_batch_size=1,
                    saved_name_train='./train_ds_2d.csv',
                    saved_name_val='./val_ds_2d.csv',
                    resized_size=(args.image_size, args.image_size, None),
                    div_size=(16,16,None),
                    ifcheck_volume=False,
                    ifcheck_sclices=False,)
    dataloader=train_loader
    #l = len(dataloader)
    l=1000 # only first test

    model = UNet(c_in=1, c_out=1,time_dim=args.time_dim, depth=args.UNet_depth).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    diffusion = Diffusion(noise_steps=args.noise_steps, img_size=args.image_size, device=device)
    model, optimizer, init_epoch = load_pretrained_model(model, optimizer, args.pretrained_path)

    epoch = 'test'
    pbar = tqdm(dataloader)  
    for i, images in enumerate(pbar): #(images, _)
        images = images["image"].to(device)
        save_img_folder = os.path.join("results", args.run_name)
        sampled_images = diffusion.sample(model, n=images.shape[0], epoch=epoch, save_img_folder=save_img_folder)


def launch():
    import argparse
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.run_name = "DDPM_Uncondtional"
    args.epochs = 1000
    args.train_number = 20
    args.batch_size = 1
    args.sample_interval = 10
    args.image_size = 512
    args.time_dim = 32
    args.UNet_depth = 128
    args.dataset_path = r"F:\yang_Projects\Datasets\Task1\pelvis" # C:\Users\56991\Projects\Datasets\Task1\pelvis D:\Projects\data\Task1\pelvis
    args.device = "cuda:1"
    args.lr = 5e-3
    args.noise_steps = 1000
    args.pretrained_path = 'F:\yang_Projects\Diffusion-Models-pytorch/models/DDPM_Uncondtional\ckpt191.pt'
    os.makedirs('./results/DDPM_Uncondtional',exist_ok=True)
    os.makedirs('./models/DDPM_Uncondtional',exist_ok=True)
    GPU_ID = 1
    device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu') # 0=TitanXP, 1=P5000
    print(torch.cuda.get_device_name(GPU_ID))
    #train(args)
    inference(args)

if __name__ == '__main__':
    launch()
