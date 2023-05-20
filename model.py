import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict


class DNN(nn.Module):
    def __init__(self, inputs_dim, hidden_units):
        super(DNN, self).__init__()
        self.inputs_dim = inputs_dim
        self.hidden_units = hidden_units
        self.dropout = nn.Dropout(0.8)

        self.hidden_units = [inputs_dim] + list(self.hidden_units)
        self.linear = nn.ModuleList([
            nn.Linear(self.hidden_units[i], self.hidden_units[i + 1]) for i in range(len(self.hidden_units) - 1)
        ])

        self.activation = nn.ReLU()

    def forward(self, X):
        inputs = X
        for i in range(len(self.linear)):
            fc = self.linear[i](inputs)
            fc = self.activation(fc)
            fc = self.dropout(fc)
            inputs = fc
        return inputs


class InteractingLayer(nn.Module):
    def __init__(self, embedding_size, use_res=True, scaling=False):
        super(InteractingLayer, self).__init__()
        self.att_embedding_size = embedding_size // 2
        self.use_res = use_res
        self.scaling = scaling

        self.W_Query = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.W_Key = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        self.W_Value = nn.Parameter(torch.Tensor(embedding_size, embedding_size))

        if self.use_res:
            self.W_Res = nn.Parameter(torch.Tensor(embedding_size, embedding_size))
        for tensor in self.parameters():
            nn.init.normal_(tensor, mean=0.0, std=0.05)

    def forward(self, inputs):
        querys = torch.tensordot(inputs, self.W_Query, dims=([-1], [0]))
        keys = torch.tensordot(inputs, self.W_Key, dims=([-1], [0]))
        values = torch.tensordot(inputs, self.W_Value, dims=([-1], [0]))

        querys = torch.stack(torch.split(querys, self.att_embedding_size, dim=2))
        keys = torch.stack(torch.split(keys, self.att_embedding_size, dim=2))
        values = torch.stack(torch.split(values, self.att_embedding_size, dim=2))

        inner_product = torch.einsum('bnik,bnjk->bnij', querys, keys)

        if self.scaling:
            inner_product /= self.att_embedding_size ** 0.5
        self.normalized_att_scores = F.softmax(inner_product, dim=-1)

        result = torch.matmul(self.normalized_att_scores, values)
        result = torch.cat(torch.split(result, 1, ), dim=-1)
        result = torch.squeeze(result, dim=0)
        if self.use_res:
            result += torch.tensordot(inputs, self.W_Res, dims=([-1], [0]))
        result = F.relu(result)
        return result


class MHAM(nn.Module):
    def __init__(self, feat_size, embedding_size, dnn_feature_columns,
                 att_layer_num=3, att_res=True, dnn_hidden_units=(256, 128)):
        super(MHAM, self).__init__()
        self.sparse_feature_columns = list(filter(lambda x: x[1] == 'sparse', dnn_feature_columns))
        self.embedding_dic = nn.ModuleDict({
            feat[0]: nn.Embedding(feat_size[feat[0]], embedding_size, sparse=False) for feat in
            self.sparse_feature_columns
        })
        self.dense_feature_columns = list(filter(lambda x: x[1] == 'dense', dnn_feature_columns))

        self.feature_index = defaultdict(int)
        start = 0
        for feat in feat_size:
            self.feature_index[feat] = start
            start += 1

        self.dnn = DNN(len(self.dense_feature_columns) + embedding_size * len(self.embedding_dic), dnn_hidden_units)

        self.dnn_linear = nn.Linear(dnn_hidden_units[-1] + embedding_size * len(self.embedding_dic), 1, bias=False)

        self.int_layers = nn.ModuleList([
            InteractingLayer(embedding_size, att_res) for _ in range(att_layer_num)
        ])

    def forward(self, X):
        sparse_embedding = [
            self.embedding_dic[feat[0]](X[:, self.feature_index[feat[0]]].long()).reshape(X.shape[0], 1, -1)
            for feat in self.sparse_feature_columns]
        sparse_input = torch.cat(sparse_embedding, dim=1)
        sparse_input = torch.flatten(sparse_input, start_dim=1)
        dense_values = [X[:, self.feature_index[feat[0]]].reshape(-1, 1) for feat in self.dense_feature_columns]
        dense_input = torch.cat(dense_values, dim=1)

        att_input = torch.cat(sparse_embedding, dim=1)
        for layer in self.int_layers:
            att_input = layer(att_input)
        att_output = torch.flatten(att_input, start_dim=1)

        dnn_input = torch.cat((dense_input, sparse_input), dim=1)
        deep_out = self.dnn(dnn_input)

        stack_out = torch.cat((att_output, deep_out), dim=-1)
        logit = self.dnn_linear(stack_out)
        y_pred = torch.sigmoid(logit)
        return y_pred