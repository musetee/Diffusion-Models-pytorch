import torch
import torch.nn as nn
class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels        
        self.mha = nn.MultiheadAttention(channels, 2, batch_first=True)
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

if __name__ == "__main__":
    channels = 8
    img_size = 128
    device = 'cuda:0'
    x = torch.randn((1, channels, img_size, img_size)).to(device)
    model_sa = SelfAttention(channels).to(device)
    x_sa = model_sa(x)
    print(x.shape)
    print(x_sa.shape)
    #print(model(x))