
import torch
import torch.nn as nn
import torch.nn.functional as F

def one_param(m):
    "get model first parameter"
    return next(iter(m.parameters()))

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    def __init__(self, channels,heads=2):
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
 

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim, # in features
                out_channels # out features
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, remove_deep_conv=False):
        super().__init__()
        self.time_dim = time_dim
        #ensrue time_dim is divisible by 8
        assert time_dim % 8 == 0 
        depth=64
        depth_2 = int(depth/2) #256
        depth_4 = int(depth/4) #128
        depth_8 = int(depth/8) #64

        self.remove_deep_conv = remove_deep_conv
        self.inc = DoubleConv(c_in, depth_8) # [B, depth_8, img_size, img_size]
        self.down1 = Down(depth_8, depth_4, emb_dim=time_dim) # [B, depth_4, img_size/2, img_size/2]
        self.sa1 = SelfAttention(depth_4)
        self.down2 = Down(depth_4, depth_2, emb_dim=time_dim) # [B, depth_2, img_size/4, img_size/4]
        self.sa2 = SelfAttention(depth_2)
        self.down3 = Down(depth_2, depth_2, emb_dim=time_dim) # [B, depth_2, img_size/8, img_size/8]
        self.sa3 = SelfAttention(depth_2)


        if remove_deep_conv:
            self.bot1 = DoubleConv(depth_2, depth_2) # [B, depth_2, img_size/8, img_size/8]
            self.bot3 = DoubleConv(depth_2, depth_2) # [B, depth_2, img_size/8, img_size/8]
        else:
            self.bot1 = DoubleConv(depth_2, depth) # [B, depth, img_size/8, img_size/8]
            self.bot2 = DoubleConv(depth, depth) # [B, depth, img_size/8, img_size/8]
            self.bot3 = DoubleConv(depth, depth_2) # [B, depth_2, img_size/8, img_size/8]

        self.up1 = Up(depth, depth_4, emb_dim=time_dim) # [B, depth_4, img_size/4, img_size/4]
        self.sa4 = SelfAttention(depth_4)
        self.up2 = Up(depth_2, depth_8, emb_dim=time_dim) # [B, depth_8, img_size/2, img_size/2]
        self.sa5 = SelfAttention(depth_8)
        self.up3 = Up(depth_4, depth_8, emb_dim=time_dim) # [B, depth_8, img_size, img_size]
        self.sa6 = SelfAttention(depth_8)
        self.outc = nn.Conv2d(depth_8, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=one_param(self).device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def unet_forwad(self, x, t): # example based on depth=512, img_size=64
        x1 = self.inc(x) # [B, depth_8, img_size, img_size] = [B, 64, 64, 64] 
        #print(f'shape of x1 {x1.shape}') 
        #print(f'shape of t {t.shape}') 
        x2 = self.down1(x1, t) # [B, depth_4, img_size/2, img_size/2] = [B, 128, 32, 32]
        x2 = self.sa1(x2) # [B, 128, 32, 32]
        x3 = self.down2(x2, t) # [B, depth_2, img_size/4, img_size/4]
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t) # [B, depth_2, img_size/8, img_size/8]
        x4 = self.sa3(x4)

        x4 = self.bot1(x4) # [B, depth, img_size/8, img_size/8]
        if not self.remove_deep_conv:
            x4 = self.bot2(x4)
        x4 = self.bot3(x4) # [B, depth_2, img_size/8, img_size/8]

        x = self.up1(x4, x3, t) # concat(up(x4), x3) -> [B, depth_4, img_size/4, img_size/4]
        x = self.sa4(x)
        x = self.up2(x, x2, t) # [B, depth_8, img_size/2, img_size/2]
        x = self.sa5(x) 
        x = self.up3(x, x1, t) # [B, depth_8, img_size, img_size]
        x = self.sa6(x)
        output = self.outc(x) # [B, c_out, img_size, img_size] 
        return output
    
    def forward(self, x, t):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim) # [B, time_dim]
        return self.unet_forwad(x, t)


class UNet_conditional(UNet):
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, **kwargs):
        super().__init__(c_in, c_out, time_dim, **kwargs)
        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def forward(self, x, t, y=None):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            t += self.label_emb(y)

        return self.unet_forwad(x, t)