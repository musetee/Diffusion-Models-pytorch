import torch.nn.functional as F
import torch
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchvision import transforms 
from torch.utils.data import DataLoader
import numpy as np
from torch import nn
import math
import datetime

def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    """ 
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device="cpu"):
    """ 
    Takes an image and a timestep as input and 
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x_0.shape
    )
    # mean + variance
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

device="cuda" if torch.cuda.is_available() else "cpu"
# Define beta schedule
T = 1000
betas = linear_beta_schedule(timesteps=T)

# Pre-calculate different terms for closed form
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
# cumprod retursn the cumulative product in a dimension,
# For example, if input is a vector of size N, 
# the result will also be a vector of size N,
# with element in place i: yi=x1*x2*x3...*xi

alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
# namely delete the last column and insert a column of 1 before the start column

sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        
    def forward(self, x, t, ):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))
        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        # Extend last 2 dimensions
        time_emb = time_emb[(..., ) + (None, ) * 2]
        # Add time channel
        h = h + time_emb
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        # Down or Upsample
        return self.transform(h)

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings

class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self):
        super().__init__()
        image_channels = 1
        down_channels = (4,8,16,32,64)
        up_channels = (64, 32, 16, 8, 4)
        out_dim = 1 
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )
        
        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], \
                                    time_emb_dim) \
                    for i in range(len(down_channels)-1)])
        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], \
                                        time_emb_dim, up=True) \
                    for i in range(len(up_channels)-1)])

        # make the output and input dimensions the same
        self.output = nn.Conv2d(up_channels[-1], out_dim, 3, padding=1)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)
        # Initial conv
        x = self.conv0(x)
        # Unet
        residual_inputs = []
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)           
            x = up(x, t)
        return self.output(x)

def get_loss(model, x_0, t): # x_0 is the initial image, t is the time step
    x_noisy, noise = forward_diffusion_sample(x_0, t, device=device) # the last batch is not full, only 4 images
    noise_pred = model(x_noisy, t)
    return F.l1_loss(x_0, noise_pred)

@torch.no_grad()
def sample_timestep(model, x, t):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    Applies noise to this image, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    
    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
    
    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

def show_tensor_image(image):
    reverse_transforms = transforms.Compose([
        #transforms.Lambda(lambda t: (t + 1) / 2),
        #transforms.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        #transforms.Lambda(lambda t: t * 255.),
        #transforms.Lambda(lambda t: t.numpy().astype(np.uint8)),
        transforms.ToPILImage(),
    ])
    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 
    plt.imshow(reverse_transforms(image))

import os
@torch.no_grad()
def sample_plot_image(model,IMG_SIZE,saved_img_folder,epoch,batch_i):
    img_size = IMG_SIZE # ([1, 400, 608])
    # generate a noise image to test the model
    img = torch.randn((1, img_size[0], img_size[1], img_size[2]), device=device)
    num_images = 10
    stepsize = int(T/num_images) 
    img_samples = []
    sample_step=0
    for i in range(0,T)[::-1]: # means from T to 0, not including T  
        # torch.full is to create a tensor of size(1,) with the same value i,  torch.full(size, fill_value, ....
        t = torch.full((1,), i, device=device, dtype=torch.long) 
        #img = sample_timestep(model, img, t)
        img = model(img, t)
        if i % stepsize == 0: # plot images every stepsize steps
            img_samples.append(img.cpu().detach().numpy())
            sample_step+=1

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

    if not os.path.exists(saved_img_folder):
            os.makedirs(saved_img_folder)
    saved_name=f"{epoch}_{batch_i}.jpg"
    fig.savefig(os.path.join(saved_img_folder,saved_name))
    #plt.show()   
    plt.close(fig)          

def config_path(model_name):
    # pathes for logs
    now = datetime.datetime.now()
    date_time = now.strftime('%Y%m%d_%H%M')
    file_prefix = f'{date_time}_{model_name}'
    root=f'./logs/{date_time}_{model_name}'
    saved_logs_folder=os.path.join(root, 'saved_logs')
    os.makedirs(saved_logs_folder, exist_ok=True) # exist_ok=True: if the folder exists, do nothing
    parameter_file=os.path.join(saved_logs_folder, f'{file_prefix}_parameters.txt')
    train_loss_file = os.path.join(saved_logs_folder, f'{file_prefix}_train_loss.txt')
    saved_img_folder = os.path.join(root, 'saved_outputs')
    saved_model_folder = os.path.join(root, 'saved_models')
    os.makedirs(saved_model_folder, exist_ok=True)
    log_dir=os.path.join(root, 'saved_tensorboard')
    os.makedirs(log_dir, exist_ok=True)
    saved_name_train = os.path.join(saved_logs_folder, f'{file_prefix}_train_loader.csv')
    saved_name_val= os.path.join(saved_logs_folder, f'{file_prefix}_val_loader.csv')
    val_log_file = os.path.join(saved_logs_folder, f'{file_prefix}_val_log.txt')
    saved_inference_folder = os.path.join(root, 'saved_inference')
    return (saved_logs_folder, 
            parameter_file, 
            train_loss_file, 
            saved_img_folder, 
            saved_model_folder, 
            log_dir, 
            saved_name_train, 
            saved_name_val,
            val_log_file,
            saved_inference_folder)

from torch.optim import Adam
class simpleDifftrainer():
    def __init__(self, 
                 #num_channels, strides, 
                 device, 
                 model_name='diffusion'):
        
        #self.num_channels = num_channels
        #self.strides = strides
        self.device=device
        #self.val_batch_size = val_batch_size
        #self.writer = writer
        self.model = SimpleUnet()
        
        my_paths=config_path(model_name)
        self.saved_logs_folder=my_paths[0]
        self.parameter_file=my_paths[1]
        self.train_loss_file = my_paths[2]
        self.saved_img_folder = my_paths[3]
        self.saved_model_folder = my_paths[4]
        self.log_dir=my_paths[5]
        self.saved_name_train = my_paths[6]
        self.saved_name_val= my_paths[7]
        self.device=device
    
    def plotCorrupt(self, train_loader):
        batch_test = next(iter(train_loader))
        x=batch_test["image"]
        print('Input shape:', x.shape)
        plt.imshow(torchvision.utils.make_grid(x)[0], cmap='Greys')
        
        # Plotting the input data
        fig, axs = plt.subplots(2, 1, figsize=(12, 5))
        axs[0].set_title('Input data')
        axs[0].imshow(torchvision.utils.make_grid(x)[0], cmap='Greys')

        # Adding noise
        #amount = torch.linspace(0, 1, x.shape[0]) # Left to right -> more corruption
        t=torch.linspace(0, 199, x.shape[0]).long()
        noised_x = forward_diffusion_sample(x, t, self.device)
        
        # Plotting the noised version
        axs[1].set_title('Corrupted data (-- amount increases -->)')
        axs[1].imshow(torchvision.utils.make_grid(noised_x)[0], cmap='Greys')

    def train(self, train_loader, learning_rate, epoch_num=4,val_interval=1, batch_interval=10):
        device = self.device
        model_Unet=self.model
        model_Unet.to(device)
        self.learning_rate = learning_rate
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
        loss_fn=nn.MSELoss()
        for epoch in range(epoch_num):
            print("-" * 10)
            print(f"epoch {epoch + 1}/{epoch_num}")
            model_Unet.train()
            step=0
            #loss=0
            losses = []
            for batch in tqdm(train_loader):
                len_batch=len(batch["image"])
                inputs, labels = batch["image"].to(device), batch["label"].to(device)
                IMG_SIZE=batch["image"][0].shape
                step+=1
                self.optimizer.zero_grad()
                t = torch.randint(0, T, (len_batch,), device=device).long() # the last batch is not full, only 4 images, so here should be matched
                # forward diffusion
                x_noisy, noise = forward_diffusion_sample(inputs, t, device=device) # the last batch is not full, only 4 images
                
                # backward diffusion ## problem!!!
                noise_pred = model_Unet(x_noisy, t)
                #loss=F.l1_loss(inputs, noise_pred)
                #loss=loss_fn(noise, noise_pred)
                loss = loss_fn(noise_pred, inputs)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                ### sample images
                if step % batch_interval == 0:
                    #print(f"Epoch {epoch+1} | step {step:03d} Loss: {loss.item()} ")
                    sample_plot_image(model_Unet,IMG_SIZE,self.saved_img_folder,epoch,step)

from mydataloader import myslicesloader
device='cuda' if torch.cuda.is_available() else 'cpu'
# path to dataset
dataset_path='./datasets/pelvis'
if __name__ == '__main__':
    model=simpleDifftrainer(device)
    train_volume_ds,_,train_loader,_,_ = myslicesloader(dataset_path,
                    normalize='zscore',
                    train_number=1,
                    val_number=1,
                    train_batch_size=8,
                    val_batch_size=1,
                    saved_name_train='./train_ds_2d.csv',
                    saved_name_val='./val_ds_2d.csv',
                    resized_size=(512,512,None),
                    div_size=(16,16,None),
                    ifcheck_volume=False,
                    ifcheck_sclices=False,)
    model.train(train_loader, 
                learning_rate=3e-3, 
                epoch_num=50,
                val_interval=1,
                batch_interval=10)
   