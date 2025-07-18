import torch
import torch.nn as nn
from models import MSWaveFuser, Attention, CNN, CNN_BiGRU, LSTM, FHG_Model, GRU, LightTS, MedGNN, MFN, MLP, PatchTST, TConv, \
    TConv2, TConv3, Testmodel, Testmodel2, CNN1D
from models.HDMixer import HDMixer
from models.TSLANet import TSLANet

class Exp_basic():
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            'MSWaveFuser': MSWaveFuser,
            'Attention': Attention,
            'CNN': CNN,
            'CNN_BiGRU': CNN_BiGRU,
            'LSTM': LSTM,
            'FHG_Model': FHG_Model,
            'GRU': GRU,
            'LightTS': LightTS,
            'MedGNN': MedGNN,
            'MFN': MFN,
            'MLP': MLP,
            'PatchTST': PatchTST,
            'TConv': TConv,
            'TConv2': TConv2,
            'TConv3': TConv3,
            'Testmodel': Testmodel,
            'Testmodel2': Testmodel2,
            'CNN1D': CNN1D,
            'HDMixer': HDMixer,
            'TSLANet': TSLANet,
        }
        self.model = self.model_dict[self.args.model_name].Model(self.args).float()

