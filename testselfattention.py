import torch
import torch.nn as nn
class SelfAttention(nn.Module):
    def __init__(self, channels, heads):
        super(SelfAttention, self).__init__()
        self.channels = channels        
        self.mha = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-1]
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2) # [B, H, W, C] -> [B, H*W, C]
        x_ln = self.ln(x) # [B, H*W, C]
        attention_value, _ = self.mha(x_ln, x_ln, x_ln) # [B, H*W, C]
        attention_value = attention_value + x # [B, H*W, C]
        attention_value = self.ff_self(attention_value) + attention_value # [B, H*W, C]
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size) # [B, C, H, W]
    
import subprocess as sp
import os
def get_gpu_memory():
    command = "nvidia-smi --query-gpu=memory.free --format=csv"
    memory_free_info = sp.check_output(command.split()).decode('ascii').split('\n')[:-1][1:]
    memory_free_values = [int(x.split()[0]) for i, x in enumerate(memory_free_info)]
    return memory_free_values

if __name__ == "__main__":
    channels = 8
    img_size = 256
    heads = 1
    device = 'cuda:1'
    x = torch.randn((1, channels, img_size, img_size)).to(device)
    model_sa = SelfAttention(channels, heads).to(device)
    x_sa = model_sa(x)
    print(f"attention heads: {heads}")
    print(f"using gpu memory: {get_gpu_memory()}")
    print(x.shape)
    print(x_sa.shape)
    #print(model(x))