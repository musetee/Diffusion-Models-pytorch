import matplotlib.pyplot as plt
import torch
import os
from utils import save_images
image= torch.randn((1, 1, 512, 512))
saved_name="testplot.jpg"
saved_name_torch="testorchplot.jpg"
fig=plt.figure(figsize=(2, 2))
plt.imshow(image[0, 0].cpu(), vmin=0, vmax=1, cmap="gray")
plt.tight_layout()
plt.axis("off")
plt.show()
fig.savefig(saved_name)
plt.close(fig) 
image = (image.clamp(-1, 1) + 1) / 2 # [-1, 1] -> [0, 1]
image = (image * 255).type(torch.uint8)
save_images(image,saved_name_torch)
