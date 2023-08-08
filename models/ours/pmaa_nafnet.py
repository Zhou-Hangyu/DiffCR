import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
from abc import abstractmethod

class EmbedBlock(nn.Module):
    """
    Any module where forward() takes embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` embeddings.
        """


class EmbedSequential(nn.Sequential, EmbedBlock):
    """
    A sequential module that passes embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, EmbedBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

def gamma_embedding(gammas, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.
    :param gammas: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0,
                                             end=half, dtype=torch.float32) / half
    ).to(device=gammas.device)
    args = gammas[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat(
            [embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class LayerNormFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):

    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(EmbedBlock):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel,
                               kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c,
                               kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel,
                               kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c,
                               kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(
            drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(
            drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros(
            (1, c, 1, 1)), requires_grad=True)
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(256, c),
        )

    def forward(self, inp, t):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        y = y+self.time_emb(t)[..., None, None]

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class UNet(nn.Module):

    def __init__(
        self,
        img_channel=12,
        width=64,
        middle_blk_num=1,
        enc_blk_nums=[1, 1, 1, 1],
        dec_blk_nums=[1, 1, 1, 1],
    ):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                               bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=3, kernel_size=3, padding=1, stride=1, groups=1,
                                bias=True)
        self.inp_ending = nn.Conv2d(in_channels=img_channel, out_channels=3, kernel_size=3, padding=1, stride=1, groups=1,
                                    bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                EmbedSequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, chan, 2, 2, groups=chan)
            )

        self.middle_blks = \
            EmbedSequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan*4, 1, groups=chan, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            self.decoders.append(
                EmbedSequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)
        self.emb = partial(gamma_embedding, dim=64)
        self.map = nn.Sequential(
            nn.Linear(64, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
        )

    def forward(self, inp, gammas):
        t = self.map(self.emb(gammas.view(-1, )))
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []
        multi_fea = []
        # import pdb; pdb.set_trace()
        out_shape = torch.div(torch.tensor([x.shape[-2], x.shape[-1]]), 2**(len(self.encoders)), rounding_mode='trunc')
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x, t)
            encs.append(x)
            multi_fea.append(F.interpolate(x, [out_shape[0], out_shape[1]]))
            x = down(x)
            
        multi_fea = torch.sum(torch.stack(multi_fea), dim=0)
        global_f = self.middle_blks(multi_fea, t)
        
        for idx in range(len(encs)):
            encs[idx] = encs[idx] * torch.sigmoid(F.interpolate(global_f, encs[idx].shape[-2:]))

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x, t)

        x = self.ending(x)
        x = x + self.inp_ending(inp)

        return x[:, :, :H, :W]

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h %
                     self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w %
                     self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x


if __name__ == '__main__':
    # img_channel = 5
    # width = 32

    # # enc_blks = [2, 2, 4, 8]
    # # middle_blk_num = 12
    # # dec_blks = [2, 2, 2, 2]

    # enc_blks = [1, 1, 1, 28]
    # middle_blk_num = 1
    # dec_blks = [1, 1, 1, 1]

    # net = ours(
    #     img_channel=img_channel,
    #     width=width,
    #     middle_blk_num=middle_blk_num,
    #     enc_blk_nums=enc_blks,
    #     dec_blk_nums=dec_blks,
    # )

    # inp_shape = (2, 5, 64, 64)

    # out_shape = net(torch.Tensor(*inp_shape)).shape

    # print(out_shape)
    # print(UNet())
    # print(UNet()(*(torch.Tensor(2, 12, 256, 256), torch.ones(2,))).shape)
    net = UNet()
    # 测试模型参数量和计算量
    from thop import profile
    from thop import clever_format
    flops, params = profile(net, inputs=(torch.Tensor(1, 12, 256, 256), torch.ones(1,)), verbose=False)
    flops, params = clever_format([flops, params], "%.3f")
    print(flops, params) # 5.095G 515.402K
    # 使用ptflops计算模型参数量和计算量
    from ptflops import get_model_complexity_info
    macs, params = get_model_complexity_info(net, input_constructor=(torch.Tensor(1, 12, 256, 256), torch.ones(1,)), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print(macs, params) # 5.095G 515.402K