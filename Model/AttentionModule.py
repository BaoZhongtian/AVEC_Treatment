import math
import torch
import numpy
from transformers import BertModel


class AveragePooling(torch.nn.Module):
    def __init__(self):
        super(AveragePooling, self).__init__()

    def forward(self, blstm_data, blstm_seq):
        if blstm_seq is None: return torch.mean(blstm_data, dim=0).unsqueeze(0)
        pooling_result = []
        for index in range(blstm_data.size()[0]):
            pooling_result.append(torch.mean(blstm_data[index][:blstm_seq[index]], dim=0).unsqueeze(0))
        pooling_result = torch.cat(pooling_result, dim=0)
        return pooling_result


class VanillaAttention(torch.nn.Module):
    def __init__(self, in_features, cuda_flag=True):
        super(VanillaAttention, self).__init__()
        self.cuda_flag = cuda_flag
        self.attention_weight_layer = torch.nn.Linear(in_features=in_features, out_features=1)

    def forward(self, batch_data, batch_seq=None):
        attention_weight = self.attention_weight_layer(batch_data).squeeze(-1)

        if batch_seq is not None:
            attention_mask = []
            for index in range(batch_seq.size()[0]):
                attention_mask.append(torch.cat(
                    [torch.ones(batch_seq[index]), -torch.ones(batch_data.size()[1] - batch_seq[index])]).unsqueeze(0))
            attention_mask = torch.cat(attention_mask, dim=0) * 9999
            if self.cuda_flag: attention_mask = attention_mask.cuda()
            masked_weighted = torch.min(attention_mask, attention_weight)
        else:
            masked_weighted = attention_weight
        softmax_weighted = masked_weighted.softmax(dim=-1).unsqueeze(-1)
        padding_weighted = softmax_weighted.repeat([1, 1, batch_data.size()[-1]])

        weighted_result = torch.multiply(batch_data, padding_weighted).sum(dim=1)
        return weighted_result


class SelfAttention(torch.nn.Module):
    def __init__(self, in_features, head_numbers, cuda_flag=True):
        super(SelfAttention, self).__init__()
        self.cuda_flag = cuda_flag

        self.num_attention_heads = head_numbers
        self.attention_head_size = int(in_features / head_numbers)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = torch.nn.Linear(in_features, self.all_head_size)
        self.key = torch.nn.Linear(in_features, self.all_head_size)
        self.value = torch.nn.Linear(in_features, self.all_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, batch_data, batch_seq=None):
        mixed_query_layer = self.query(batch_data)
        mixed_key_layer = self.key(batch_data)
        mixed_value_layer = self.value(batch_data)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = torch.nn.Softmax(dim=-1)(attention_scores)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        result = torch.mean(context_layer, dim=1)
        return result
