import torch
from torchvision.utils import save_image
from ddpm import Diffusion
from utils import get_data

from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torchvision.utils import save_image
import torch
from torchvision.datasets import ImageFolder
image_folder_path = "./logs"

def noise_images(x, t, device="cpu", img_channel=3):
    beta_start = 1e-4
    beta_end = 0.02
    noise_steps = 1000
    img_channel = img_channel

    prepare_noise_schedule = torch.linspace(beta_start, beta_end, noise_steps)
    beta = prepare_noise_schedule.to(device)
    alpha = 1. - beta
    alpha_hat = torch.cumprod(alpha, dim=0)

    
    sqrt_alpha_hat = torch.sqrt(alpha_hat[t])[:, None, None, None] # [:, None, None, None] add 4 dims
    sqrt_one_minus_alpha_hat = torch.sqrt(1 - alpha_hat[t])[:, None, None, None]
    Ɛ = torch.randn_like(x)
    return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--noiselevel", type=int, default=50)
    args = parser.parse_args()
    # Define the data transforms (modify as needed)
    data_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
    ])

    # Create a custom dataset
    custom_dataset = ImageFolder(root=image_folder_path, transform=data_transforms)

    # Create a DataLoader for your dataset
    batch_size = 1  # Adjust this according to your needs
    dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

    image = next(iter(dataloader))[0]
    print(image.shape)

    noise_level = args.noiselevel
    t = torch.Tensor([noise_level]).long() #, 100, 150, 200, 300, 600, 700, 999

    noised_image, _ = noise_images(image, t)
    save_image(noised_image.add(1).mul(0.5), f"noise_{noise_level}.jpg")
