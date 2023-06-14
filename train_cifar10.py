from types import SimpleNamespace

import wandb
from ddpm_conditional import Diffusion
from utils import get_cifar

if __name__ == '__main__':
    # Trains a conditional diffusion model on CIFAR10
    # This is a very simple example, for more advanced training, see `ddpm_conditional.py`
    img_size=32
    config = SimpleNamespace(    
        run_name = "cifar10_ddpm_conditional",
        epochs = 25,
        noise_steps=1000,
        seed = 42,
        batch_size = 128,
        img_size = img_size,
        num_classes = 10,
        dataset_path = get_cifar(img_size=img_size),
        train_folder = "train",
        val_folder = "test",
        device = "cuda:1",
        slice_size = 1,
        do_validation = True,
        fp16 = True,
        log_every_epoch = 5,
        num_workers=2,
        lr = 5e-3)

    diff = Diffusion(noise_steps=config.noise_steps , img_size=config.img_size)
    import torch
    GPU_ID = 1
    device = torch.device(f'cuda:{GPU_ID}' if torch.cuda.is_available() else 'cpu') # 0=TitanXP, 1=P5000
    print(torch.cuda.get_device_name(GPU_ID))

    with wandb.init(project="train_sd", group="train", config=config):
        diff.prepare(config)
        diff.fit(config)