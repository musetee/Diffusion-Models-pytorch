"""
This code train a conditional diffusion model on CIFAR.
It is based on @dome272.

@wandbcode{condition_diffusion}
"""

import argparse, logging, copy
from types import SimpleNamespace
from contextlib import nullcontext

import torch
from torch import optim
import torch.nn as nn
import numpy as np
from fastprogress import progress_bar

import wandb
from utils import *
from modules import UNet_conditional, EMA
from mydataloader.conditional_loader import get_dataset
logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, num_classes=10, c_in=1, c_out=1, device="cuda", **kwargs):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.img_size = img_size
        self.model = UNet_conditional(c_in=1, c_out=1, num_classes=num_classes, **kwargs).to(device)
        self.ema_model = copy.deepcopy(self.model).eval().requires_grad_(False)
        self.device = device
        self.c_in = c_in
        self.num_classes = num_classes
    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
    
    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def noise_images(self, x, t):
        "Add noise to images at instant t"
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ
    
    @torch.inference_mode()
    def sample(self, use_ema, labels, cfg_scale=3):
        model = self.ema_model if use_ema else self.model
        n =len(labels)
        logging.info(f"Sampling {n} new images....")
        model.eval()
        with torch.inference_mode():
            x = torch.randn((n, self.c_in, self.img_size, self.img_size)).to(self.device)
            for i in progress_bar(reversed(range(0, self.noise_steps)), total=self.noise_steps-1, leave=False):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                if cfg_scale > 0:
                    uncond_predicted_noise = model(x, t, None)
                    predicted_noise = torch.lerp(uncond_predicted_noise, predicted_noise, cfg_scale)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

    def train_step(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.ema.step_ema(self.ema_model, self.model)
        self.scheduler.step()

    def one_epoch(self, train=True):
        avg_loss = 0.
        if train: self.model.train()
        else: self.model.eval()
        pbar = progress_bar(self.train_dataloader, leave=False)
        for i, (images, labels) in enumerate(pbar):
            with torch.autocast("cuda") and (torch.inference_mode() if not train else torch.enable_grad()):
                images = images.to(self.device)
                labels = labels.to(self.device)
                #print('labels: ',labels)
                
                t = self.sample_timesteps(images.shape[0]).to(self.device)
                x_t, noise = self.noise_images(images, t)
                if np.random.random() < 0.1:
                    labels = None
                predicted_noise = self.model(x_t, t, labels)
                loss = self.mse(noise, predicted_noise)
                avg_loss += loss
            if train:
                self.train_step(loss)
                wandb.log({"train_mse": loss.item(), "learning_rate": self.scheduler.get_last_lr()[0]})
            pbar.comment = f"MSE={loss.item():2.3f}"        
        return avg_loss.mean().item()

    def log_images(self):
        "Log images to wandb and save them to disk"
        labels = torch.tensor([100,101,102,103,104]).long().to(self.device) #torch.arange(self.num_classes).long().to(self.device) # 0,1,2,3......149 
        sampled_images = self.sample(use_ema=False, labels=labels)
        wandb.log({"sampled_images":     [wandb.Image(img.permute(1,2,0).squeeze().cpu().numpy()) for img in sampled_images]})

        # EMA model sampling
        ema_sampled_images = self.sample(use_ema=True, labels=labels)
        #plot_images(sampled_images)  #to display on jupyter if available
        wandb.log({"ema_sampled_images": [wandb.Image(img.permute(1,2,0).squeeze().cpu().numpy()) for img in ema_sampled_images]})

    def load(self, model_cpkt_path, model_ckpt="ckpt.pt", ema_model_ckpt="ema_ckpt.pt"):
        self.model.load_state_dict(torch.load(os.path.join(model_cpkt_path, model_ckpt)))
        self.ema_model.load_state_dict(torch.load(os.path.join(model_cpkt_path, ema_model_ckpt)))

    def save_model(self, run_name, epoch=-1):
        logging.info(f"Save model locally and on wandb")
        torch.save(self.model.state_dict(), os.path.join("./models", run_name, f"ckpt{epoch}.pt"))
        torch.save(self.ema_model.state_dict(), os.path.join("./models", run_name, f"ema_ckpt{epoch}.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join("./models", run_name, f"optim{epoch}.pt"))
        at = wandb.Artifact("model", type="model", description="Model weights for DDPM conditional", metadata={"epoch": epoch})
        at.add_dir(os.path.join("models", run_name))
        wandb.log_artifact(at)

    

    def prepare(self, args):
        mk_folders(args.run_name)
        #self.train_dataloader, self.val_dataloader = get_data(args)
        data_pelvis_path=args.dataset_path
        train_number=args.train_number
        val_number=args.val_number
        normalize=args.normalize
        resized_size=(args.img_size, args.img_size)
        div_size=(args.div_size,args.div_size)
        center_crop=args.center_crop
        manual_crop=args.manual_crop
        train_batch_size=args.train_batch_size
        val_batch_size=args.val_batch_size
        self.train_dataloader, self.val_dataloader = get_dataset(data_pelvis_path, 
                                                                train_number, 
                                                                val_number, 
                                                                normalize, 
                                                                resized_size, 
                                                                div_size, 
                                                                center_crop,
                                                                manual_crop,
                                                                train_batch_size,
                                                                val_batch_size)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=args.lr, eps=1e-5)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=args.lr, 
                                                 steps_per_epoch=len(self.train_dataloader), epochs=args.epochs)
        self.mse = nn.MSELoss()
        self.ema = EMA(0.995)
        self.scaler = torch.cuda.amp.GradScaler()

    def testdata(self):
        for i, (images, labels) in enumerate(self.train_dataloader):
            print(i, ' image: ',images.shape)
            print(i, ' label: ',labels.shape)
            break

    def fit(self, args):
        for epoch in progress_bar(range(args.epochs), total=args.epochs, leave=False):
            current_epoch = epoch + 1
            logging.info(f"Starting epoch {current_epoch}:")
            _  = self.one_epoch(train=True)
            
            ## validation
            if args.do_validation:
                logging.info(f"Starting validation...")
                avg_loss = self.one_epoch(train=False)
                wandb.log({"val_mse": avg_loss})
            
            # log predicitons
            if current_epoch % args.log_every_epoch == 0:
                logging.info(f"Logging images...")
                self.log_images()

            # save model
            self.save_model(run_name=args.run_name, epoch=epoch)

    def test(self, model_cpkt_path, epoch_i):
        self.load(model_cpkt_path, model_ckpt=f"ckpt{epoch_i}.pt", ema_model_ckpt=f"ema_ckpt{epoch_i}.pt")
        self.log_images()

def parse_args(config):
    parser = argparse.ArgumentParser(description='Process hyper-parameters')
    parser.add_argument('--run_name', type=str, default=config.run_name, help='name of the run')
    parser.add_argument('--epochs', type=int, default=config.epochs, help='number of epochs')
    parser.add_argument('--seed', type=int, default=config.seed, help='random seed')
    #parser.add_argument('--batch_size', type=int, default=config.batch_size, help='batch size')
    parser.add_argument('--img_size', type=int, default=512, help='image size')
    parser.add_argument('--num_classes', type=int, default=config.num_classes, help='number of classes')
    parser.add_argument('--dataset_path', type=str, default=config.dataset_path, help='path to dataset')
    parser.add_argument('--device', type=str, default=config.device, help='device')
    parser.add_argument('--lr', type=float, default=config.lr, help='learning rate')
    parser.add_argument('--slice_size', type=int, default=config.slice_size, help='slice size')
    parser.add_argument('--noise_steps', type=int, default=config.noise_steps, help='noise steps')
    parser.add_argument('--train_number', type=int, default=config.train_number , help='number of train data')
    parser.add_argument('--val_number', type=int, default=config.val_number, help='number of val data')
    parser.add_argument('--normalize', type=str, default=config.normalize, help='normalize')
    parser.add_argument('--div_size', type=int, default=config.div_size, help='div size')
    parser.add_argument('--center_crop', type=int, default=config.center_crop,help='center crop')
    parser.add_argument('--train_batch_size', type=int, default=config.train_batch_size, help='train batch size')
    parser.add_argument('--val_batch_size', type=int, default=config.val_batch_size, help='val batch size')
    args = vars(parser.parse_args())
    
    # update config with parsed args
    for k, v in args.items():
        setattr(config, k, v)


if __name__ == '__main__':
    dataset_path_razer = r'C:\Users\56991\Projects\Datasets\Task1\pelvis' # get_cifar(img_size=64),
    dataset_path_server = r"F:\yang_Projects\Datasets\Task1\pelvis"

    config = SimpleNamespace(    
        run_name = "DDPM_conditional",
        epochs = 100,
        noise_steps=1000,
        seed = 10,
        #batch_size = 10,
        img_size = 512,
        num_classes = 10, #152, # the maximum slices number of all patient data
        dataset_path = dataset_path_server,
        #train_folder = "train",
        #val_folder = "test",
        device = "cuda:0",
        slice_size = 1,
        do_validation = True,
        #fp16 = True,
        log_every_epoch = 5,
        num_workers=0,
        lr = 5e-3,
        train_number=2,
        val_number=1,
        normalize='zscore',
        div_size=16,
        center_crop=0,
        manual_crop=[41,50],
        train_batch_size=1,
        val_batch_size=1)
    parse_args(config)

    ## seed everything
    set_seed(config.seed)

    diffuser = Diffusion(config.noise_steps, img_size=config.img_size, num_classes=config.num_classes, device=config.device)

    with wandb.init(project="train_sd", group="train_crop", config=config):
        diffuser.prepare(config)
        diffuser.fit(config)
        model_cpkt_path = r'F:\yang_Projects\Diffusion-Models-pytorch\models\DDPM_conditional_1'
        epoch_i =99
        #diffuser.test(model_cpkt_path, epoch_i)
