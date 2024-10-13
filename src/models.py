import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import numpy as np
import math


class PredictorTCN(nn.Module):
    def __init__(self, args, causal=False):
        super(PredictorTCN, self).__init__()

        self.num_stages = args.num_stages
        self.ms_tcn = MultiStageModel(
            args.layer_type,
            args.kernel_size,
            args.num_stages,
            args.num_layers,
            args.model_dim,
            args.input_dim,
            args.num_classes,
            args.channel_dropout_prob,
            args.use_features,
            causal,
        )

        self.use_inp_ch_dropout = args.use_inp_ch_dropout
        if args.use_inp_ch_dropout:
            self.channel_dropout = torch.nn.Dropout1d(args.channel_dropout_prob)

    def forward(self, x_in, masks):
        assert len(masks) == self.num_stages
        if self.use_inp_ch_dropout:
            x_in = self.channel_dropout(x_in)
        frame_wise_pred, out_features = self.ms_tcn(x_in, masks)
        return frame_wise_pred, out_features



class MultiStageModel(nn.Module):
    def __init__(
        self,
        layer_type,
        kernel_size,
        num_stages,
        num_layers,
        num_f_maps,
        dim,
        num_classes,
        dropout,
        use_features=False,
        causal=False,
    ):
        super(MultiStageModel, self).__init__()
        self.use_features = use_features
        self.stage1 = SingleStageModel(
            layer_type,
            kernel_size,
            num_layers,
            num_f_maps,
            dim,
            num_classes,
            dropout,
            causal_conv=causal,
        )

        stage_in_dim = num_classes
        if self.use_features:
            stage_in_dim = num_classes + dim

        self.stages = nn.ModuleList(
            [
                copy.deepcopy(
                    SingleStageModel(
                        layer_type,
                        kernel_size,
                        num_layers,
                        num_f_maps,
                        stage_in_dim,
                        num_classes,
                        dropout,
                        causal_conv=causal,
                    )
                )
                for s in range(num_stages - 1)
            ]
        )


    def forward(self, x, masks):
        out, out_features = self.stage1(x, masks[0])
        outputs = out.unsqueeze(0)
        outputs_features = out_features.unsqueeze(0)

        for sn, s in enumerate(self.stages):
            if self.use_features:
                out, out_features = s(torch.cat((F.softmax(out, dim=1) * masks[sn], x), dim=1), masks[sn])
            else:
                out, out_features = s(F.softmax(out, dim=1) * masks[sn], masks[sn])
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
            outputs_features = torch.cat((outputs_features, out_features.unsqueeze(0)), dim=0)

        return outputs, outputs_features



class SingleStageModel(nn.Module):
    def __init__(
        self,
        layer_type,
        kernel_size,
        num_layers,
        num_f_maps,
        dim,
        num_classes,
        dropout,
        causal_conv=False,
    ):
        super(SingleStageModel, self).__init__()
        self.layer_types = {
            "base_dr": DilatedDrResidualLayer,
            "gated": DilatedGatedResidualLayer,
        }

        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [
                copy.deepcopy(
                    self.layer_types[layer_type](
                        kernel_size,
                        2**i,
                        num_f_maps,
                        num_f_maps,
                        dropout,
                        causal_conv,
                    )
                )
                for i in range(num_layers)
            ]
        )
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)


    def forward(self, x, mask):
        out = self.conv_1x1(x) * mask
        for layer in self.layers:
            out = layer(out, mask)

        out_features = out * mask
        out_logits = self.conv_out(out) * mask
        return out_logits, out_features


# BASE DROPOUT
class DilatedDrResidualLayer(nn.Module):
    def __init__(
        self,
        kernel_size,
        dilation,
        in_channels,
        out_channels,
        dropout,
        causal_conv=False,
    ):
        super(DilatedDrResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=int(kernel_size / 2) * dilation,
            dilation=dilation,
        )

        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout)
        self.ch_dropout = nn.Dropout1d(dropout)

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.ch_dropout(out)

        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask


# GATED
class DilatedGatedResidualLayer(nn.Module):
    def __init__(
        self,
        kernel_size,
        dilation,
        in_channels,
        out_channels,
        dropout,
        causal_conv=False,
    ):
        super(DilatedGatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=int(kernel_size / 2) * dilation,
            dilation=dilation,
        )
        self.gate_conv_dilated = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=int(kernel_size / 2) * dilation,
            dilation=dilation,
        )
        self.sigmoid = nn.Sigmoid()

        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout)
        self.ch_dropout = nn.Dropout1d(dropout)


    def forward(self, x, mask):
        conv_out = self.conv_dilated(x)
        gate_out = self.sigmoid(self.gate_conv_dilated(x))
        out = torch.mul(conv_out, gate_out)
        out = self.ch_dropout(out)

        out = self.conv_1x1(out)
        out = F.relu(out)
        out = self.dropout(out)

        return (x + out) * mask


