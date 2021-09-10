import torch
import numpy
from Model.AttentionModule import VanillaAttention, SelfAttention, AveragePooling


class TCN_Part(torch.nn.Module):
    def __init__(self, feature_number):
        super(TCN_Part, self).__init__()
        self.tcn_layer_1st = torch.nn.Conv1d(
            in_channels=feature_number, out_channels=128, kernel_size=3, padding=1, dilation=1)
        self.batch_normalization_1st = torch.nn.BatchNorm1d(num_features=128)

        self.tcn_layer_2nd = torch.nn.Conv1d(
            in_channels=128, out_channels=128, kernel_size=3, padding=2, dilation=2)
        self.batch_normalization_2nd = torch.nn.BatchNorm1d(num_features=128)

        self.tcn_layer_3rd = torch.nn.Conv1d(
            in_channels=128, out_channels=128, kernel_size=3, padding=4, dilation=4)
        self.batch_normalization_3rd = torch.nn.BatchNorm1d(num_features=128)

        self.tcn_layer_4th = torch.nn.Conv1d(
            in_channels=128, out_channels=128, kernel_size=3, padding=8, dilation=8)
        self.batch_normalization_4th = torch.nn.BatchNorm1d(num_features=128)

    def forward(self, batch_data):
        batch_data = batch_data.permute(0, 2, 1)
        tcn_1st_result = self.tcn_layer_1st(batch_data)
        bn_1st_result = self.batch_normalization_1st(tcn_1st_result).relu()

        tcn_2nd_result = self.tcn_layer_2nd(bn_1st_result)
        bn_2nd_result = self.batch_normalization_2nd(tcn_2nd_result).relu()

        tcn_3rd_result = self.tcn_layer_3rd(bn_2nd_result)
        bn_3rd_result = self.batch_normalization_3rd(tcn_3rd_result).relu()

        tcn_4th_result = self.tcn_layer_4th(bn_3rd_result)
        bn_4th_result = self.batch_normalization_4th(tcn_4th_result).relu()
        return bn_4th_result.permute(0, 2, 1)


class HierarchyTCN(torch.nn.Module):
    def __init__(self, feature_number, first_attention_type, second_attention_type=None):
        super(HierarchyTCN, self).__init__()
        self.first_attention_type = first_attention_type
        if second_attention_type is not None:
            self.second_attention_type = second_attention_type
        else:
            self.second_attention_type = first_attention_type

        self.tcn_first = TCN_Part(feature_number=feature_number)
        self.tcn_second = TCN_Part(feature_number=128)
        self.predict_layer = torch.nn.Linear(in_features=128, out_features=1)

        if self.first_attention_type == 'Vanilla':
            self.attention_first = VanillaAttention(in_features=128)
        elif self.first_attention_type == 'Average':
            self.attention_first = AveragePooling()
        elif self.first_attention_type == 'Self':
            self.attention_first = SelfAttention(in_features=128, head_numbers=8)
        print('First', self.first_attention_type)

        if self.second_attention_type == 'Vanilla':
            self.attention_second = VanillaAttention(in_features=128)
        elif self.second_attention_type == 'Average':
            self.attention_second = AveragePooling()
        elif self.second_attention_type == 'Self':
            self.attention_second = SelfAttention(in_features=128, head_numbers=8)
        print('Second', self.second_attention_type)
        self.loss = torch.nn.SmoothL1Loss()

    def forward(self, treat_data, treat_seq, treat_label=None):
        treat_data, treat_seq = treat_data.squeeze(), treat_seq.squeeze()
        assert treat_data.size()[0] == treat_seq.size()[0]

        tcn_first_result = self.tcn_first(treat_data)
        pooling_first_result = self.attention_first(tcn_first_result, treat_seq).unsqueeze(0)
        tcn_second_result = self.tcn_second(pooling_first_result)
        pooling_second_result = self.attention_second(tcn_second_result, None)
        predict = self.predict_layer(pooling_second_result)

        if treat_label is None: return predict
        return self.loss(input=predict, target=treat_label.squeeze())
