
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, channels, size):
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)


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
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, device="mps"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


class UNet_conditional(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, device="mps"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

        if num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t, y):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        if y is not None:
            t += self.label_emb(y)

        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


"""
====================================================================================================================================================================================================================================================================================================================
"""
class ConvBlock(nn.Module):
    def __init__(self, start, middle=None, end = None, activation=nn.ReLU, end_activation=nn.ReLU, emb_channels=16, cond_embed_channels=16):
        super().__init__()
        if end == None:
            end = start
        if middle == None:
            middle = end


        if end_activation == None:
            end_activation = activation
        
        self.block = nn.Sequential(
            nn.Conv2d(start, middle, kernel_size=3, stride=1,padding=1),
            activation(),
            nn.Conv2d(middle, end, kernel_size=3, stride=1, padding=1),
            end_activation()
        )

        self.emb = nn.Linear(emb_channels, emb_channels)

        self.emb_cond = nn.Linear(cond_embed_channels, emb_channels)

    def forward(self, x, t=None,c=None): 
        #T and C being embedding labels for timestep + conditioning before embedding
        # Skip connection concatenated beforehand


        if t != None:
            t = self.emb(t)

            t = t.unsqueeze(-1).unsqueeze(-1)
            t = t.expand(-1, -1, x.size(2), x.size(3))
            x = torch.cat([x, t], dim=1)
        
        if c != None:

            c = self.emb_cond(c)


            c = c.unsqueeze(-1).unsqueeze(-1)
            c = c.expand(-1, -1, x.size(2), x.size(3))
            x = torch.cat([x, c], dim=1)

        x = self.block(x)

        return x


class ConvDownBlock(ConvBlock):
    def __init__(self, start, end=None, middle=None,  activation=nn.ReLU, end_activation=nn.ReLU, emb_channels=16, cond_embed_channels=16):
        if end == None:
            end = start * 2
        if middle == None:
            middle = end
        super().__init__(start, middle, end, activation, end_activation, emb_channels, cond_embed_channels)

        

        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(start, middle, kernel_size=3, stride=1,padding=1),
            activation(),
            nn.Conv2d(middle, end, kernel_size=3, stride=1, padding=1),
            end_activation()
        )

class ConvUpBlock(ConvBlock):
    def __init__(self, start,end=None, middle=None,  activation=nn.ReLU, end_activation=nn.ReLU,  emb_channels=16, cond_embed_channels=16):
        super().__init__(start, middle, end, activation, end_activation, emb_channels, cond_embed_channels)

        if end == None:
            end = start // 2
        if middle == None:
            middle = end


        self.block = nn.Sequential(
            nn.ConvTranspose2d(start, middle, kernel_size=4, stride=2,padding=1),
            activation(),
            nn.ConvTranspose2d(middle, end, kernel_size=3, stride=1, padding=1),
            end_activation()
        )




class SimpleUNet(nn.Module):
    def __init__(
            self, 
            pos_channels: int = 16,
            prefix: nn.Sequential = nn.Sequential(), 
            down: nn.ModuleList = nn.ModuleList(), 
            bottleneck: nn.Sequential = nn.Sequential(), 
            up: nn.ModuleList = nn.ModuleList(), 
            postfix: nn.Sequential = nn.Sequential()
        ):
        super().__init__()

        self.prefix = prefix
        start_depth = 8
        self.down = nn.ModuleList([
            ConvDownBlock(3+pos_channels, start_depth), # ends 8*32*32
            ConvDownBlock(start_depth+pos_channels, start_depth*2), # ends 16*16*16
            ConvDownBlock(start_depth*2+pos_channels, start_depth*4), # ends 32*8*8
        ])
        self.bottleneck = ConvBlock(start_depth*4, start_depth*8, start_depth*4)
        self.up = nn.ModuleList([
            ConvUpBlock(2*start_depth*4+pos_channels, start_depth*2),
            ConvUpBlock(2*start_depth*2+pos_channels, start_depth),
            ConvUpBlock(2*start_depth+pos_channels, 3, end_activation=nn.Sigmoid)

        ])
        self.postix = postfix
        self.channels = 16
    @staticmethod
    def positional_encode(t, channels):

        t = t.unsqueeze(-1).squeeze(0)

        inv_freq = 1.0 / (
        10000 ** (torch.arange(0, channels, 2).float() / channels)
        )

        inv_freq = inv_freq.to(t.device)
        # print(t.size())
        # print(inv_freq.size())

        # print(t.repeat(1, channels // 2).size())
        
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc




    def forward(self, x, t):

        t = self.positional_encode(t, self.channels)



        x = self.prefix(x)

        skips = []
        for layer in self.down:
            x = layer(x, t)
            skips.append(torch.clone(x))
        
        x = self.bottleneck(x)

        for i, layer in enumerate(self.up):

            x = torch.cat((x, skips[-(i+1)]), dim=1)

            x = layer(x, t)

        x = self.postix(x)

        return x  
if __name__ == '__main__':
    net = UNet(device="cpu")
    #net = UNet_conditional(num_classes=10, device="cpu")
    print(sum([p.numel() for p in net.parameters()]))
    x = torch.randn(3, 3, 64, 64)
    t = x.new_tensor([256] * x.shape[0]).long()
    y = x.new_tensor([1] * x.shape[0]).long()
    print(net(x, t).shape)
