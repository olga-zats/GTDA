from turtle import st
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import math
from einops import rearrange

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    

class BitDiffPredictorTCN(nn.Module):
    def __init__(self, args, causal=False):
        super(BitDiffPredictorTCN, self).__init__()

        self.ms_tcn = DiffMultiStageModel(
            args.layer_type,
            args.kernel_size,
            args.num_stages,
            args.num_layers,
            args.model_dim,
            args.input_dim + 2 * args.num_classes,
            args.num_classes,
            args.channel_dropout_prob,
            args.use_features,
            causal
        )
        
        self.use_inp_ch_dropout = args.use_inp_ch_dropout
        if args.use_inp_ch_dropout:
            self.channel_dropout = torch.nn.Dropout1d(args.channel_dropout_prob)
    

        
    def forward(self, x, t, stage_masks, obs_cond=None, self_cond=None):
        # arange
        x = rearrange(x, 'b t c -> b c t')
        obs_cond = rearrange(obs_cond, 'b t c -> b c t')
        self_cond = rearrange(self_cond, 'b t c -> b c t')
        stage_masks = [rearrange(mask, "b t c -> b c t") for mask in stage_masks]
       
        if self.use_inp_ch_dropout:
            x = self.channel_dropout(x)
        
        # condition on input
        x = torch.cat((x, obs_cond), dim=1)
        x = torch.cat((x, self_cond), dim=1)
        
        frame_wise_pred, _ = self.ms_tcn(x, t, stage_masks)
        frame_wise_pred = rearrange(frame_wise_pred, "s b c t -> s b t c")
        return frame_wise_pred



class DiffMultiStageModel(nn.Module):
    def __init__(self, 
                layer_type,
                kernel_size,
                num_stages,
                num_layers,
                num_f_maps,
                dim, num_classes,
                dropout,
                use_features=False,
                causal=False):
        super(DiffMultiStageModel, self).__init__()

        self.use_features = use_features
        stage_in_dim = num_classes
        if self.use_features:
            stage_in_dim = num_classes + dim

        self.stage1 = DiffSingleStageModel(
            layer_type, 
            kernel_size,
            num_layers,
            num_f_maps,
            dim, 
            num_classes, 
            dropout, 
            causal_conv=causal)
        

        self.stages = nn.ModuleList(
            [
                copy.deepcopy(
                    DiffSingleStageModel(
                        layer_type,
                        kernel_size, 
                        num_layers, 
                        num_f_maps, 
                        stage_in_dim, 
                        num_classes, 
                        dropout, 
                        causal_conv=causal
                        )
                    ) for s in range(num_stages-1)
            ]
        )


    def forward(self, x, t, stage_masks):
        out, out_features = self.stage1(x, t, stage_masks[0])
        outputs = out.unsqueeze(0)
     
        for sn, s in enumerate(self.stages):
            if self.use_features:
                out, out_features = s(torch.cat((F.softmax(out, dim=1) * stage_masks[sn], x), dim=1), t, stage_masks[sn])
            else:
                out, out_features = s(F.softmax(out, dim=1) * stage_masks[sn], t, stage_masks[sn])
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)

        return outputs, out_features



class DiffSingleStageModel(nn.Module):
    def __init__(self, 
                layer_type, 
                kernel_size, 
                num_layers, 
                num_f_maps, 
                dim, 
                num_classes, 
                dropout, 
                causal_conv=False):
        super(DiffSingleStageModel, self).__init__()
        
        self.layer_types = {
            'base_dr': DiffDilatedResidualLayer,
            'gated': DiffDilatedGatedResidualLayer,
        }

        #    
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)

        # time cond
        time_dim = num_f_maps * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(num_f_maps),
            nn.Linear(num_f_maps, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # conv layers
        self.layers = nn.ModuleList(
            [ 
                copy.deepcopy(
                    self.layer_types[layer_type](
                        kernel_size,
                        2 ** i,
                        num_f_maps,
                        num_f_maps,
                        time_dim,
                        dropout,
                        causal_conv)
                    ) for i in range(num_layers)
            ]
        )

        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

        


    def forward(self, x, t, mask):
        # embed 
        out = self.conv_1x1(x) * mask  
        time = self.time_mlp(t)
        
        # pass through layers
        for layer in self.layers:
            out = layer(out, time, mask)

        # output
        out_features = out * mask
        out_logits = self.conv_out(out) * mask
        return out_logits, out_features



# BASE
class DiffDilatedResidualLayer(nn.Module):
    def __init__(self, kernel_size, dilation, in_channels, out_channels, time_channels=-1, dropout=0.2, causal_conv=False):
        super(DiffDilatedResidualLayer, self).__init__()

        # Net
        self.conv_dilated = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            padding=int(kernel_size/2)*dilation, 
            dilation=dilation
        )
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout)
        self.ch_dropout = nn.Dropout1d(dropout)      

        # Time Net
        self.time_channels = time_channels
        if time_channels > 0:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_channels, out_channels * 2)
            )


    def forward(self, x, t, mask):
        # conv net
        out = F.relu(self.conv_dilated(x))
        out = self.ch_dropout(out)
        out = self.conv_1x1(out)
        out = self.dropout(out)

        # time conditioning
        if self.time_channels > 0:
            time_scale, time_shift = self.time_mlp(t).chunk(2, dim=1)
            time_scale = rearrange(time_scale, 'b d -> b d 1')
            time_shift = rearrange(time_shift, 'b d -> b d 1')
            out = out * (time_scale + 1) + time_shift

        return (x + out) * mask


# GATED
class DiffDilatedGatedResidualLayer(nn.Module):
    def __init__(self,
                kernel_size, 
                dilation, 
                in_channels, 
                out_channels, 
                time_channels=-1, 
                dropout=0.2, 
                causal_conv=False
    ):
        super(DiffDilatedGatedResidualLayer, self).__init__()
        
        self.conv_dilated = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            padding=int(kernel_size/2)*dilation, 
            dilation=dilation
        )
        self.gate_conv_dilated = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size, 
            padding=int(kernel_size/2)*dilation, 
            dilation=
            dilation
        )
        self.sigmoid = nn.Sigmoid()
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout)
        self.ch_dropout = nn.Dropout1d(dropout)
    

        # Time Net
        self.time_channels = time_channels
        if time_channels > 0:
            self.time_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(time_channels, out_channels * 2)
        )

        
    def forward(self, x, t, mask):
        conv_out = self.conv_dilated(x)
        gate_out = self.sigmoid(self.gate_conv_dilated(x)) 
        out = torch.mul(conv_out, gate_out)
        out = self.ch_dropout(out)

        out = self.conv_1x1(out)
        out = F.relu(out)
        out = self.dropout(out)

        # time conditioning
        if self.time_channels > 0:
            time_scale, time_shift = self.time_mlp(t).chunk(2, dim=1)
            time_scale = rearrange(time_scale, 'b d -> b d 1')
            time_shift = rearrange(time_shift, 'b d -> b d 1')
            out = out * (time_scale + 1) + time_shift

        return (x + out) * mask
