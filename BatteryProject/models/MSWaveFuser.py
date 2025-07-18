import torch.nn as nn
import torch
from torch.nn import functional as F
from layers.DWT.DWT_Decomposition import Decomposition


class Conv1D(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv1d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm1d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class MSCBlockLayer(nn.Module):
    def __init__(self, inc, ouc, k):
        super().__init__()
        self.in_conv = Conv1D(inc, ouc, 1)
        self.mid_conv = Conv1D(ouc, ouc, k, g=ouc)
        self.out_conv = Conv1D(ouc, inc, 1)

    def forward(self, x):
        return self.out_conv(self.mid_conv(self.in_conv(x)))


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv1d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm1d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class Upsample(nn.Module):
    def __init__(self, c1, c2, scale_factor=2):
        super().__init__()
        if scale_factor == 2:
            self.cv1 = nn.ConvTranspose1d(c1, c2, 2, 2, 0, bias=True)
        elif scale_factor == 4:
            self.cv1 = nn.ConvTranspose1d(c1, c2, 4, 4, 0, bias=True)

    def forward(self, x):
        return self.cv1(x)


class MLAF(nn.Module):
    def __init__(self, c1, c2, level=0):
        super().__init__()
        c1_l, c1_m, c1_h = c1[0], c1[1], c1[2]
        self.level = level
        self.dim = c1_l, c1_m, c1_h
        self.inter_dim = self.dim[self.level]
        compress_c = 8  # 8

        if level == 0:
            self.stride_level_1 = Upsample(c1_m, self.inter_dim)
            self.stride_level_2 = Upsample(c1_h, self.inter_dim, scale_factor=4)
        if level == 1:
            self.stride_level_0 = Conv(c1_l, self.inter_dim, 2, 2, 0)
            self.stride_level_2 = Upsample(c1_h, self.inter_dim)
        if level == 2:
            self.stride_level_0 = Conv(c1_l, self.inter_dim, 4, 4, 0)
            self.stride_level_1 = Conv(c1_m, self.inter_dim, 2, 2, 0)

        self.weight_level_0 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weights_levels = nn.Conv1d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
        self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)

    def forward(self, x):
        x_level_0, x_level_1, x_level_2 = x[0], x[1], x[2]

        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 1:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_resized = self.stride_level_0(x_level_0)
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_resized = x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)

        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        w = self.weights_levels(levels_weight_v)
        w = F.softmax(w, dim=1)

        fused_out_reduced = level_0_resized * w[:, :1] + level_1_resized * w[:, 1:2] + level_2_resized * w[:, 2:]
        return self.conv(fused_out_reduced)


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False):
        super().__init__()
        self.dims, self.contiguous = dims, contiguous

    def forward(self, x):
        if self.contiguous:
            return x.transpose(*self.dims).contiguous()
        else:
            return x.transpose(*self.dims)


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-6, data_format="channels_last"):
        super(LayerNorm, self).__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        # print(x.shape)
        B, M, D, N = x.shape
        # x = x.permute(0, 1, 3, 2)
        x = x.reshape(B * M, D, N)
        x = self.norm(x)
        x = x.reshape(B, M, D, N)
        # x = x.permute(0, 1, 3, 2)
        return x


class TokenMixer(nn.Module):
    def __init__(self, input_seq=[], batch_size=[], channel=[], pred_seq=[], dropout=[], factor=[], d_model=[]):
        super(TokenMixer, self).__init__()
        self.input_seq = input_seq
        self.batch_size = batch_size
        self.channel = channel
        self.pred_seq = pred_seq
        self.dropout = dropout
        self.factor = factor
        self.d_model = d_model

        self.dropoutLayer = nn.Dropout(self.dropout)
        self.layers = nn.Sequential(nn.Linear(self.input_seq, self.pred_seq * self.factor),
                                    nn.GELU(),
                                    nn.Dropout(self.dropout),
                                    nn.Linear(self.pred_seq * self.factor, self.pred_seq)
                                    )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.layers(x)
        x = x.transpose(1, 2)
        return x


# Wavelet-Hierarchical Patch Mixer(WHP-Mixer)
class WHP_Mixer(nn.Module):
    def __init__(self,
                 input_length=[],
                 pred_length=[],
                 wavelet_name=[],
                 level=[],
                 batch_size=[],
                 channel=[],
                 d_model=[],
                 dropout=[],
                 embedding_dropout=[],
                 tfactor=[],
                 dfactor=[],
                 device=[],
                 patch_len=[],
                 patch_stride=[],
                 no_decomposition=[],
                 use_amp=[]):
        super(WHP_Mixer, self).__init__()
        self.input_length = input_length
        self.pred_length = pred_length
        self.wavelet_name = wavelet_name
        self.level = level
        self.batch_size = batch_size
        self.channel = channel
        self.d_model = d_model
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.device = device
        self.no_decomposition = no_decomposition
        self.tfactor = tfactor
        self.dfactor = dfactor
        self.use_amp = use_amp

        self.Decomposition_model = Decomposition(input_length=self.input_length,
                                                 pred_length=self.pred_length,
                                                 wavelet_name=self.wavelet_name,
                                                 level=self.level,
                                                 batch_size=self.batch_size,
                                                 channel=self.channel,
                                                 d_model=self.d_model,
                                                 tfactor=self.tfactor,
                                                 dfactor=self.dfactor,
                                                 device=self.device,
                                                 no_decomposition=self.no_decomposition,
                                                 use_amp=self.use_amp)

        self.input_w_dim = self.Decomposition_model.input_w_dim  # list of the length of the input coefficient series
        self.pred_w_dim = self.Decomposition_model.pred_w_dim  # list of the length of the predicted coefficient series

        self.patch_len = patch_len
        self.patch_stride = patch_stride

        self.mpmblock = nn.ModuleList([MPMBlock(input_seq=self.input_w_dim[i],
                                                pred_seq=self.pred_w_dim[i],
                                                batch_size=self.batch_size,
                                                channel=self.channel,
                                                d_model=self.d_model,
                                                dropout=self.dropout,
                                                embedding_dropout=self.embedding_dropout,
                                                tfactor=self.tfactor,
                                                dfactor=self.dfactor,
                                                patch_len=self.patch_len,
                                                patch_stride=self.patch_stride) for i in
                                       range(len(self.input_w_dim))])

    def forward(self, xL):
        x = xL.transpose(1, 2)
        # xA: approximation coefficient series,
        # xD: detail coefficient series
        # yA: predicted approximation coefficient series
        # yD: predicted detail coefficient series

        xA, xD = self.Decomposition_model.transform(x)

        yA = self.mpmblock[0](xA)
        yD = []
        for i in range(len(xD)):
            yD_i = self.mpmblock[i + 1](xD[i])
            yD.append(yD_i)

        y = self.Decomposition_model.inv_transform(yA, yD)
        y = y.transpose(1, 2)
        xT = y[:, -self.pred_length:, :]  # decomposition output is always even, but pred length can be odd

        return xT


class Multi_PatchMixer(nn.Module):
    def __init__(self, input_seq, pred_seq, patch_len, channel, dropout, embedding_dropout):
        super(Multi_PatchMixer, self).__init__()
        self.input_seq = input_seq
        self.pred_seq = pred_seq
        self.patch_len = patch_len
        self.d_model = patch_len * 2
        self.channel = channel
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.patch_stride = self.patch_len // 2
        self.patch_num = int((self.input_seq - self.patch_len) / self.patch_stride + 2)

        self.patch_norm = nn.BatchNorm2d(self.channel)
        self.patch_embedding_layer = nn.Linear(self.patch_len, self.d_model)
        self.dropoutLayer = nn.Dropout(self.embedding_dropout)

        self.patch_mixer = nn.Sequential(
            LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_model * 2, self.d_model),
            nn.Dropout(dropout),
        )

        self.time_mixer = nn.Sequential(
            Transpose(2, 3), LayerNorm(self.patch_num),
            nn.Linear(self.patch_num, self.patch_num * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.patch_num * 2, self.patch_num),
            nn.Dropout(dropout),
            Transpose(2, 3)
        )

        self.head = nn.Sequential(nn.Flatten(start_dim=-2, end_dim=-1),
                                  nn.Linear(self.patch_num * self.d_model, self.pred_seq))

    def forward(self, x):
        x_patch = self.do_patching(x)
        x_patch = self.patch_norm(x_patch)
        x_emb = self.dropoutLayer(self.patch_embedding_layer(x_patch))

        u = self.patch_mixer(x_emb) + x_emb
        v = self.time_mixer(u) + u  # + x_emb
        out = self.head(v)
        return out

    def do_patching(self, x):
        x_end = x[:, :, -1:]
        x_padding = x_end.repeat(1, 1, self.patch_stride)
        x_new = torch.cat((x, x_padding), dim=-1)
        x_patch = x_new.unfold(dimension=-1, size=self.patch_len, step=self.patch_stride)
        return x_patch


# Multi Patch Mixer Block(MPMBlock)
class MPMBlock(nn.Module):
    def __init__(self,
                 input_seq=[],
                 pred_seq=[],
                 batch_size=[],
                 channel=[],
                 d_model=[],
                 dropout=[],
                 embedding_dropout=[],
                 tfactor=[],
                 dfactor=[],
                 patch_len=[],
                 patch_stride=[]):
        super(MPMBlock, self).__init__()
        self.input_seq = input_seq
        self.pred_seq = pred_seq
        self.batch_size = batch_size
        self.channel = channel
        self.d_model = d_model
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.tfactor = tfactor
        self.dfactor = dfactor
        self.patch_len = patch_len
        self.patch_stride = patch_stride

        self.multi_patch0 = Multi_PatchMixer(input_seq, pred_seq, patch_len[0], channel, dropout, embedding_dropout)
        self.multi_patch1 = Multi_PatchMixer(input_seq, pred_seq, patch_len[1], channel, dropout, embedding_dropout)
        self.multi_patch2 = Multi_PatchMixer(input_seq, pred_seq, patch_len[2], channel, dropout, embedding_dropout)

        self.weights = nn.Parameter(torch.ones(3))

        self.variable_mixer = nn.Sequential(
            Transpose(1, 2),
            nn.LayerNorm(channel),
            nn.Linear(channel, channel * 2),  # 2
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channel * 2, channel),
            nn.Dropout(dropout),
            Transpose(1, 2)
        )

    def forward(self, x):
        x0 = self.multi_patch0(x)
        x1 = self.multi_patch1(x)
        x2 = self.multi_patch2(x)

        norm_weights = torch.softmax(self.weights, dim=0)
        x = x0 * norm_weights[0] + x1 * norm_weights[1] + x2 * norm_weights[2]

        out = self.variable_mixer(x) + x

        return out


class MSCBlock(nn.Module):
    def __init__(self, inc, ouc, kernel_sizes, in_expand_ratio=3., mid_expand_ratio=2., layers_num=3, in_down_ratio=2.):
        super().__init__()

        in_channel = int(inc * in_expand_ratio // in_down_ratio)
        self.mid_channel = in_channel // len(kernel_sizes)
        groups = int(self.mid_channel * mid_expand_ratio)

        self.in_conv = Conv1D(inc, in_channel, 1)

        self.mid_convs = nn.ModuleList()
        for kernel_size in kernel_sizes:
            if kernel_size == 1:
                self.mid_convs.append(nn.Identity())
                continue
            mid_convs = [MSCBlockLayer(self.mid_channel, groups, k=kernel_size) for _ in range(layers_num)]
            self.mid_convs.append(nn.Sequential(*mid_convs))

        self.out_conv = Conv1D(in_channel, ouc, 1)

    def forward(self, x):
        out = self.in_conv(x)
        channels = []
        for i, mid_conv in enumerate(self.mid_convs):
            channel = out[:, i * self.mid_channel:(i + 1) * self.mid_channel, :]
            if i >= 1:
                channel = channel + channels[i - 1]
            channel = mid_conv(channel)
            channels.append(channel)
        out = torch.cat(channels, dim=1)
        return self.out_conv(out)


class Model(nn.Module):
    def __init__(self, args, tfactor=5, dfactor=5, wavelet='db2', level=5, stride=8, no_decomposition=False):
        super(Model, self).__init__()
        self.args = args
        self.down_sampling_method = 'avg'
        self.down_sampling_window = 2
        self.down_sampling_layers = 2

        # Multi-LevelAdaptiveFusion
        self.mlaf = MLAF(c1=(4, 4, 4), c2=4, level=0)

        # self.device = torch.device('cpu')
        self.device = torch.device('cuda:{}'.format(1))

        self.whpmixer = WHP_Mixer(input_length=128,
                                  pred_length=64,
                                  wavelet_name=wavelet,
                                  level=1,
                                  batch_size=32,
                                  channel=4,
                                  d_model=64,
                                  dropout=0.5,
                                  embedding_dropout=0.4,
                                  tfactor=tfactor,
                                  dfactor=dfactor,
                                  device=self.device,
                                  patch_len=[8, 12, 16],
                                  patch_stride=stride,
                                  no_decomposition=no_decomposition,
                                  use_amp=False)

        self.mscblock = MSCBlock(inc=4, ouc=4, kernel_sizes=[1, 3, 5, 7], layers_num=3, in_expand_ratio=3.,
                                 in_down_ratio=1)

        self.predictor = nn.Sequential(
            nn.Linear(64 * 4, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1)
        )

    def forecast(self, x_enc):
        # Normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # Multi-LevelAdaptiveFusion(MLAF)
        for i in range(2):
            x_enc_list = self.__multi_level_process_inputs(x_enc)
            x_enc = self.mlaf(
                [x_enc_list[0].permute(0, 2, 1), x_enc_list[1].permute(0, 2, 1), x_enc_list[2].permute(0, 2, 1)])
            x_enc = x_enc.transpose(1, 2)

        # Wavelet-Hierarchical Patch Mixer(WHP-Mixer)
        pred = self.wpmixerCore(x_enc)
        pred = pred[:, :, -4:]

        # Multi-Scale Convolution Block(MSCBlock)
        pred = self.mscblock(pred.transpose(1, 2))
        pred = pred.transpose(1, 2)

        # De-Normalization
        dec_out = pred * (stdev[:, 0].unsqueeze(1).repeat(1, 64, 1))
        dec_out = dec_out + (means[:, 0].unsqueeze(1).repeat(1, 64, 1))
        return dec_out

    def forward(self, x_enc):
        x_enc = x_enc.permute(0, 2, 1)
        B = x_enc.shape[0]
        dec_out = self.forecast(x_enc)
        outputs = dec_out.reshape(B, -1)
        outputs = self.predictor(outputs)

        return outputs

    def __multi_level_process_inputs(self, x_enc):
        if self.down_sampling_method == 'max':
            down_pool = torch.nn.MaxPool1d(self.down_sampling_window, return_indices=False)
        elif self.down_sampling_method == 'avg':
            down_pool = torch.nn.AvgPool1d(self.down_sampling_window)
        else:
            return x_enc
        x_enc = x_enc.permute(0, 2, 1)
        x_enc_ori = x_enc
        x_enc_sampling_list = []
        x_enc_sampling_list.append(x_enc.permute(0, 2, 1))
        for i in range(self.down_sampling_layers):
            x_enc_sampling = down_pool(x_enc_ori)
            x_enc_sampling_list.append(x_enc_sampling.permute(0, 2, 1))
            x_enc_ori = x_enc_sampling
        x_enc = x_enc_sampling_list
        return x_enc


if __name__ == '__main__':
    x = torch.randn(32, 4, 128)
    model = Model(args=None)
    out = model(x)
    print(out.shape)  # (32, 1)