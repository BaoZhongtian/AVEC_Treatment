import torch
import numpy
from Model.AttentionModule import VanillaAttention, AveragePooling, SelfAttention


class HierarchyBLSTM(torch.nn.Module):
    def __init__(self, feature_number, first_attention_type, second_attention_type=None):
        super(HierarchyBLSTM, self).__init__()
        self.first_attention_type = first_attention_type
        if second_attention_type is not None:
            self.second_attention_type = second_attention_type
        else:
            self.second_attention_type = first_attention_type

        self.blstm_first = torch.nn.LSTM(
            input_size=feature_number, hidden_size=128, bidirectional=True, batch_first=True)
        self.blstm_second = torch.nn.LSTM(
            input_size=256, hidden_size=128, bidirectional=True, batch_first=True)

        self.predict_layer = torch.nn.Linear(in_features=256, out_features=1)

        if self.first_attention_type == 'Vanilla':
            self.attention_first = VanillaAttention(in_features=256)
        elif self.first_attention_type == 'Average':
            self.attention_first = AveragePooling()
        elif self.first_attention_type == 'Self':
            self.attention_first = SelfAttention(in_features=256, head_numbers=8)
        print('First', self.first_attention_type)

        if self.second_attention_type == 'Vanilla':
            self.attention_second = VanillaAttention(in_features=256)
        elif self.second_attention_type == 'Average':
            self.attention_second = AveragePooling()
        elif self.second_attention_type == 'Self':
            self.attention_second = SelfAttention(in_features=256, head_numbers=8)
        print('Second', self.second_attention_type)

        self.loss = torch.nn.SmoothL1Loss()

    def forward(self, treat_data, treat_seq, treat_label=None):
        treat_data, treat_seq = treat_data.squeeze(), treat_seq.squeeze()
        assert treat_data.size()[0] == treat_seq.size()[0]

        blstm_first_hidden, _ = self.blstm_first(treat_data)
        pooling_first_result = self.attention_first(blstm_first_hidden, treat_seq).unsqueeze(0)
        blstm_second_hidden, _ = self.blstm_second(pooling_first_result)
        pooling_second_result = self.attention_second(blstm_second_hidden, None)

        predict = self.predict_layer(pooling_second_result)

        if treat_label is None: return predict
        return self.loss(input=predict, target=treat_label.squeeze())
