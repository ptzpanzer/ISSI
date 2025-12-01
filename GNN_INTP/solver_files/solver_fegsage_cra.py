# Transformer implementation from https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec
# GELU implementation from https://github.com/karpathy/minGPT

import json
import os

import copy
import math

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.nn.models import GraphSAGE

import support_functions


required_settings = {
    # run_mode
    "debug": None,
    
    # wandb settings
    "agent_id": None,

    # HPC settings
    "job_id": None,
    "coffer_slot": None,

    # Dataset settings
    "dataset": None,
    "fold": None,
    "holdout": None,

    # Model settings
    "model": "FEGSAGE",
    'trans_embedding_dim': 64,
    'num_trans_head': 2,
    'num_trans_layers': 2,
    'trans_dropout': 0.0,
    'k': 15,
    'conv_dim': 256,
    'p': 0.0, 

    # Training settings
    'seed': 1,
    'full_batch': 64,
    'real_batch': 64,
    'epoch': 100,
    'nn_lr': 1e-3,
    'es_mindelta': 0.0,
    'es_endure': 3,
}


def padded_seq_to_vectors(padded_seq, logger):
    # Get the actual lengths of each sequence in the batch
    actual_lengths = logger.int()
    # Step 1: Form the first tensor containing all actual elements from the batch
    mask = torch.arange(padded_seq.size(1), device=padded_seq.device) < actual_lengths.view(-1, 1)
    tensor1 = torch.masked_select(padded_seq, mask.unsqueeze(-1)).view(-1, padded_seq.size(-1))
    # Step 2: Form the second tensor to record which row each element comes from
    tensor2 = torch.repeat_interleave(torch.arange(padded_seq.size(0), device=padded_seq.device), actual_lengths)
    return tensor1, tensor2


def extract_first_element_per_batch(tensor1, tensor2):
    # Get the unique batch indices from tensor2
    unique_batch_indices = torch.unique(tensor2)
    # Initialize a list to store the first elements of each batch item
    first_elements = []
    # Iterate through each unique batch index
    for batch_idx in unique_batch_indices:
        # Find the first occurrence of the batch index in tensor2
        first_occurrence = torch.nonzero(tensor2 == batch_idx, as_tuple=False)[0, 0]
        # Extract the first element from tensor1 and append it to the list
        first_element = tensor1[first_occurrence]
        first_elements.append(first_element)
    # Convert the list to a tensor
    result = torch.stack(first_elements, dim=0)
    return result


class NewGELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

    
class FeedForward(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout):
        super().__init__() 
        self.linear_1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(dim_feedforward, d_model)
    
    def forward(self, x):
        x = self.dropout(NewGELU()(self.linear_1(x)))
        x = self.linear_2(x)
        return x


class Norm(nn.Module):
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()
        self.size = d_model
        self.eps = eps
        
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm


def attention(q, k, v, d_k, mask, dropout):
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)

    # print(scores.size())
    # print(mask.size())
    
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    output = torch.matmul(scores, v)
    
    return output


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, nhead, dropout, k_mode):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model
        self.h = nhead
        self.k_mode = k_mode
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model + k_mode, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(nhead * d_model, d_model)
    
    def forward(self, q, k, v, mask):
        bs = q.size(0)
        # perform linear operation and split into N heads
        k = self.k_linear(k).unsqueeze(-2).repeat(1, 1, self.h, 1)
        q = self.q_linear(q).unsqueeze(-2).repeat(1, 1, self.h, 1)
        v = self.v_linear(v).unsqueeze(-2).repeat(1, 1, self.h, 1)
        # transpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k, mask, self.dropout)
        # concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.h * self.d_model)
        output = self.out(concat)
    
        return output


class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout, k_mode):
        super().__init__()
        self.k_mode = k_mode
        
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

        self.attn = MultiHeadAttention(d_model, nhead, dropout, k_mode)

        self.ff = FeedForward(d_model, dim_feedforward, dropout)
        
    def forward(self, x, env, mask):
        # x2 = self.norm_1(x)
        x2 = x
        
        # if env is not None:
        #     print(f"x2: {x2.size()}")
        #     print(f"env: {env.size()}")
        #     print(f"cb: {torch.concat([x2, env.unsqueeze(1).repeat(1, x2.size(1), 1)], dim=2).size()}")
        
        if self.k_mode == 0:
            x = x + self.dropout_1(self.attn(x2, x2, x2, mask))
        else:
            x = x + self.dropout_1(self.attn(x2, torch.concat([x2, env.unsqueeze(1).repeat(1, x2.size(1), 1)], dim=2), x2, mask))
        
        # x2 = self.norm_2(x)
        x2 = x
        x = x + self.dropout_2(self.ff(x2))
        return x


def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder(nn.Module):
    def __init__(self, d_model, nhead, num_encoder_layers, dim_feedforward, dropout, k_mode):
        super().__init__()
        self.num_encoder_layers = num_encoder_layers
        self.layers = get_clones(EncoderLayer(d_model, nhead, dim_feedforward, dropout, k_mode), num_encoder_layers)
        self.norm = Norm(d_model)
        
    def forward(self, src, env, mask):
        for i in range(self.num_encoder_layers):
            src = self.layers[i](src, env, mask)
        # return self.norm(src)
        return src


def length_to_mask(lengths, total_len, device):
    max_len = total_len
    mask = torch.arange(max_len).expand(lengths.shape[0], max_len).to(device) < lengths.unsqueeze(1)
    return mask.unsqueeze(-2)


# Returns the closest number that is a power of 2 to the given real number x
def closest_power_of_2(x):
    return 2 ** round(math.log2(x))


# Returns a list of n numbers that are evenly spaced between a and b.
def evenly_spaced_numbers(a, b, n):
    if n == 1:
        return [(a+b)/2]
    step = (b-a)/(n-1)
    return [a + i*step for i in range(n)]
    
    
# generate a V-shape MLP as torch.nn.Sequential given input_size, output_xibnsize, and layer_count(only linear layer counted)
def generate_sequential(a, b, n, p):
    layer_sizes = evenly_spaced_numbers(a, b, n)
    layer_sizes = [int(layer_sizes[0])] + [int(closest_power_of_2(x)) for x in layer_sizes[1:-1]] + [int(layer_sizes[-1])]
    
    layers = []
    for i in range(n-1):
        layers.append(torch.nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        if i == 0:
            layers.append(NewGELU())
        elif 0 < i < n-2:
            layers.append(NewGELU())
            # layers.append(torch.nn.Dropout(p))
    
    model = torch.nn.Sequential(*layers)
    return model


def create_mat_masks(tensor, real_sizes, p):
    batch_size, padded_length, _ = tensor.shape
    softmax_mask = torch.ones_like(tensor, dtype=torch.bool)
    mat_mask = torch.ones_like(tensor, dtype=torch.bool)
    rnd_mask = torch.ones_like(tensor, dtype=torch.bool)
    for i in range(batch_size):
        real_size = real_sizes[i].item()
        softmax_mask[i, :, :real_size] = False
        mat_mask[i, :real_size, :real_size] = False
        rnd_mask[i, :real_size, :real_size] = torch.rand(real_size, real_size) < p
    return softmax_mask, mat_mask, rnd_mask


def lr_schedule(epoch):
    warmup_steps = 1000
    if epoch < warmup_steps:
        return float(epoch) / float(warmup_steps)
    else:
        return 1.0


def alarm(t, name):
    if torch.isinf(t).any():
        print(f"INF detected in {name}")
    if torch.isnan(t).any():
        print(f"NAN detected in {name}")
        

class INTP_Model(nn.Module):
    """
        GCN
    """
    def __init__(self, settings, device):
        super(INTP_Model, self).__init__()

        self.settings = settings
        self.device = device

        origin_path = f"./Datasets/{settings['dataset']}/"
        with open(origin_path + f'meta_data.json', 'r') as f:
            self.dataset_info = json.load(f)

        if torch.cuda.is_available():
            self.ngpu = torch.cuda.device_count()
        else:
            self.ngpu = 0

        settings['trans_fe_dim'] = len(self.dataset_info["op_dic"]) + len(list(self.dataset_info["non_eu_col"].keys()))
        settings['trans_feedforward_dim'] = settings['trans_embedding_dim'] * 4
        settings['gnn_fe_dim'] = settings['trans_embedding_dim']

        self.c_emb = torch.nn.Linear(len(self.dataset_info["op_dic"]) + len(list(self.dataset_info["non_eu_col"].keys())) + len(list(self.dataset_info["eu_col"].keys())), settings['trans_embedding_dim']).to(self.device)
        self.cc_emb = torch.nn.Linear(len(self.dataset_info["op_dic"]) + len(list(self.dataset_info["non_eu_col"].keys())) + len(list(self.dataset_info["eu_col"].keys())), settings['trans_embedding_dim']).to(self.device)
        
        self.embedding_layer_a = torch.nn.Linear(settings['trans_embedding_dim'], settings['trans_embedding_dim']).to(self.device)
        self.embedding_layer_b = torch.nn.Linear(settings['trans_embedding_dim'], settings['trans_embedding_dim']).to(self.device)

        self.gsage = GraphSAGE(in_channels=settings['gnn_fe_dim'], hidden_channels=settings['conv_dim'], num_layers=2, out_channels=1).to(self.device)

        # self.c_factor = torch.nn.Parameter(torch.tensor([0.5, ], device=self.device))
        
        # init weights
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.kaiming_normal_(p)

        self.optimizer = torch.optim.Adam(self.parameters(), weight_decay=5e-4, lr=settings['nn_lr'])
        self.scheduler = None

    def forward(self, batch):
        inputs, coords, targets, cc, input_lenths = batch
        head_only = True

        inputs_c = inputs - inputs[:, 0:1, :]

        i_emb = self.c_emb(torch.cat((inputs, coords), dim=-1))
        ic_emb = self.cc_emb(torch.cat((inputs_c, cc), dim=-1))

        tokens_emb_a = self.embedding_layer_a(i_emb)
        tokens_emb_b = self.embedding_layer_b(i_emb)
        tokens_emb_a_c = self.embedding_layer_a(ic_emb)
        tokens_emb_b_c = self.embedding_layer_b(ic_emb)

        attention_mask = length_to_mask(input_lenths, inputs.size(1), self.device)

        a = tokens_emb_a
        b = tokens_emb_b
        a_c = tokens_emb_a_c
        b_c = tokens_emb_b_c

        adj_mat = torch.matmul(a, b.transpose(1, 2)) / math.sqrt(self.settings['trans_embedding_dim'])
        adj_mat_c = torch.matmul(a_c, b_c.transpose(1, 2)) / math.sqrt(self.settings['trans_embedding_dim'])
        
        softmax_mask, mat_mask, rnd_mask = create_mat_masks(adj_mat, input_lenths, self.settings['p'])
        adj_mat[softmax_mask] = float('-inf')
        adj_mat_c[softmax_mask] = float('-inf')

        adj_mat_diff = adj_mat.clone()
        adj_mat_diff[softmax_mask] = 0
        adj_mat_c_diff = adj_mat_c.clone()
        adj_mat_c_diff[softmax_mask] = 0
        diff = torch.nn.MSELoss()(adj_mat_diff, adj_mat_c_diff)
        
        # adj_mat = F.softmax(adj_mat, dim=-1)
        adj_mat_inplace = adj_mat.clone()
        adj_mat_inplace[mat_mask] = float('-inf')
        if self.training:
            adj_mat_inplace[rnd_mask] = float('-inf')
        cor_indices = torch.arange(adj_mat_inplace.size(-1))
        adj_mat_inplace[:, cor_indices, cor_indices] = float('-inf')
        
        k = adj_mat.size(-1)
        top_values, top_indices = torch.topk(adj_mat, k=k, dim=-1)
        row_indices = torch.arange(top_indices.size(1)).unsqueeze(0).unsqueeze(-1).unsqueeze(-1).to(self.device)
        row_indices = row_indices.expand(top_indices.size(0), -1, k, -1)
        batch_indices = torch.arange(top_indices.size(0)).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(self.device)
        batch_indices = batch_indices.expand(-1, top_indices.size(1), k, -1)
        top_indices_with_row = torch.cat((batch_indices, top_indices.unsqueeze(-1), row_indices), dim=3)
        top_values_inplace, top_indices_inplace = torch.topk(adj_mat_inplace, k=k, dim=-1)
            
        valid_mask = top_values_inplace != float('-inf')
        valid_top_values = torch.masked_select(top_values, valid_mask)
        valid_top_indices = torch.masked_select(top_indices_with_row, valid_mask.unsqueeze(-1)).view(-1, 3)

        cum_sum = torch.cumsum(input_lenths, dim=0)
        cum_sum_shifted = torch.cat([torch.zeros_like(cum_sum[:1]), cum_sum[:-1]], dim=0)
        idx_start = torch.index_select(cum_sum_shifted, dim=0, index=valid_top_indices[:, 0])

        edge_index = torch.transpose(valid_top_indices[:, 1:], 0, 1) + idx_start
        edge_weight = valid_top_values

        gnn_fes = i_emb - ic_emb
        x_l, indexer = padded_seq_to_vectors(gnn_fes, input_lenths)
        y_l, _ = padded_seq_to_vectors(targets, input_lenths)

        output = self.gsage(x_l, edge_index, edge_weight)
        
        if not head_only:
            return (output, diff), y_l
        else:
            output_head = extract_first_element_per_batch(output, indexer)
            target_head = extract_first_element_per_batch(y_l, indexer)
            return (output_head, diff), target_head

    def loss_func(self, model_output, target):
        output, diff = model_output

        if self.ngpu <= 1:
            loss = torch.nn.L1Loss()(output, target) + 0.1 * diff
        else:
            loss = torch.nn.L1Loss()(output, target) + 0.1 * sum(diff)

        return loss, output, target
