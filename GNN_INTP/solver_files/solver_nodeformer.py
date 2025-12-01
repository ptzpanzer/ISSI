# NodeFormer implementation from https://github.com/OpenGSL/OpenGSL

import json
import os
import math

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch_geometric.nn import GCNConv, knn_graph
# from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree

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
    "model": "NODEFORMER",
    'num_layers': 2,
    'num_heads': 2,
    'k': 5,
    'gumbel_sample': 15,
    'conv_dim': 256,

    'lambda': 0.1,

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


BIG_CONSTANT = 1e8


def adj_mul(adj_i, adj, N):
    adj_i_sp = torch.sparse_coo_tensor(adj_i, torch.ones(adj_i.shape[1], dtype=torch.float).to(adj.device), (N, N))
    adj_sp = torch.sparse_coo_tensor(adj, torch.ones(adj.shape[1], dtype=torch.float).to(adj.device), (N, N))
    adj_j = torch.sparse.mm(adj_i_sp, adj_sp)
    adj_j = adj_j.coalesce().indices()

    # print(f'adj_j: {adj_j}')
    return adj_j


def create_projection_matrix(m, d, seed=0):
    block_list = []
    current_seed = seed
    torch.manual_seed(current_seed)
    unstructured_block = torch.randn((d, d))
    q, _ = torch.qr(unstructured_block)
    q = torch.t(q)
    block_list.append(q[0:m])
    final_matrix = torch.vstack(block_list)

    current_seed += 1
    torch.manual_seed(current_seed)
    multiplier = torch.norm(torch.randn((m, d)), dim=1)

    return torch.matmul(torch.diag(multiplier), final_matrix)


def softmax_kernel_transformation(data, is_query, projection_matrix=None, numerical_stabilizer=0.000001):
    data_normalizer = 1.0 / torch.sqrt(torch.sqrt(torch.tensor(data.shape[-1], dtype=torch.float32)))   # 不太理解这一行
    data = data_normalizer * data
    ratio = 1.0 / torch.sqrt(torch.tensor(projection_matrix.shape[0], dtype=torch.float32))
    data_dash = torch.einsum("bnhd,md->bnhm", data, projection_matrix)
    diag_data = torch.square(data)
    diag_data = torch.sum(diag_data, dim=len(data.shape)-1)
    diag_data = diag_data / 2.0
    diag_data = torch.unsqueeze(diag_data, dim=len(data.shape)-1)
    last_dims_t = len(data_dash.shape) - 1
    attention_dims_t = len(data_dash.shape) - 3
    if is_query:
        # 下面的几行也不理解
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(data_dash, dim=last_dims_t, keepdim=True)[0]) + numerical_stabilizer
        )
    else:
        data_dash = ratio * (
            torch.exp(data_dash - diag_data - torch.max(torch.max(data_dash, dim=last_dims_t, keepdim=True)[0],
                    dim=attention_dims_t, keepdim=True)[0]) + numerical_stabilizer
        )
    return data_dash


def numerator(qs, ks, vs):
    kvs = torch.einsum("nbhm,nbhd->bhmd", ks, vs) # kvs refers to U_k in the paper
    return torch.einsum("nbhm,bhmd->nbhd", qs, kvs)


def denominator(qs, ks):
    all_ones = torch.ones([ks.shape[0]]).to(qs.device)
    ks_sum = torch.einsum("nbhm,n->bhm", ks, all_ones) # ks_sum refers to O_k in the paper
    return torch.einsum("nbhm,bhm->nbh", qs, ks_sum)


def numerator_gumbel(qs, ks, vs):
    kvs = torch.einsum("nbhkm,nbhd->bhkmd", ks, vs) # kvs refers to U_k in the paper
    rtn = torch.einsum("nbhm,bhkmd->nbhkd", qs, kvs)
    alarm(kvs, "kvs_n")
    alarm(rtn, "rtn_n")
    return rtn


def denominator_gumbel(qs, ks):
    all_ones = torch.ones([ks.shape[0]]).to(qs.device)
    ks_sum = torch.einsum("nbhkm,n->bhkm", ks, all_ones) # ks_sum refers to O_k in the paper
    rtn = torch.einsum("nbhm,bhkm->nbhk", qs, ks_sum)
    alarm(ks_sum, "ks_sum_d")
    alarm(rtn, "rtn_d")
    return rtn


def kernelized_softmax(query, key, value, projection_matrix=None, edge_index=None, tau=0.25):
    '''
    fast computation of all-pair attentive aggregation with linear complexity
    input: query/key/value [B, N, H, D]
    return: updated node emb, attention weight (for computing edge loss)
    B = graph number (always equal to 1 in Node Classification), N = node number, H = head number,
    M = random feature dimension, D = hidden size
    '''
    query = query / math.sqrt(tau)
    key = key / math.sqrt(tau)
    query_prime = softmax_kernel_transformation(query, True, projection_matrix) # [B, N, H, M]， 只有softmax_kernel_transformation一种
    key_prime = softmax_kernel_transformation(key, False, projection_matrix) # [B, N, H, M]
    query_prime = query_prime.permute(1, 0, 2, 3) # [N, B, H, M]
    key_prime = key_prime.permute(1, 0, 2, 3) # [N, B, H, M]
    value = value.permute(1, 0, 2, 3) # [N, B, H, D]

    # compute updated node emb, this step requires O(N)
    z_num = numerator(query_prime, key_prime, value)
    z_den = denominator(query_prime, key_prime)

    z_num = z_num.permute(1, 0, 2, 3)  # [B, N, H, D]
    z_den = z_den.permute(1, 0, 2)
    z_den = torch.unsqueeze(z_den, len(z_den.shape))
    z_output = z_num / z_den # [B, N, H, D]

    # 一定是True
    start, end = edge_index
    query_end, key_start = query_prime[end], key_prime[start] # [E, B, H, M]
    edge_attn_num = torch.einsum("ebhm,ebhm->ebh", query_end, key_start) # [E, B, H]
    edge_attn_num = edge_attn_num.permute(1, 0, 2) # [B, E, H]
    attn_normalizer = denominator(query_prime, key_prime) # [N, B, H]
    edge_attn_dem = attn_normalizer[end]  # [E, B, H]
    edge_attn_dem = edge_attn_dem.permute(1, 0, 2) # [B, E, H]
    A_weight = edge_attn_num / edge_attn_dem # [B, E, H]

    return z_output, A_weight


def kernelized_gumbel_softmax(query, key, value, projection_matrix=None, edge_index=None,
                                K=10, tau=0.25):
    '''
    fast computation of all-pair attentive aggregation with linear complexity
    input: query/key/value [B, N, H, D]
    return: updated node emb, attention weight (for computing edge loss)
    B = graph number (always equal to 1 in Node Classification), N = node number, H = head number,
    M = random feature dimension, D = hidden size, K = number of Gumbel sampling
    '''
    query = query / math.sqrt(tau)
    key = key / math.sqrt(tau)

    alarm(query, "query_g")
    alarm(key, "key_g")
    
    query_prime = softmax_kernel_transformation(query, True, projection_matrix) # [B, N, H, M]
    key_prime = softmax_kernel_transformation(key, False, projection_matrix) # [B, N, H, M]

    alarm(query_prime, "queryp_g")
    alarm(key_prime, "keyp_g")
    
    query_prime = query_prime.permute(1, 0, 2, 3) # [N, B, H, M]
    key_prime = key_prime.permute(1, 0, 2, 3) # [N, B, H, M]
    value = value.permute(1, 0, 2, 3) # [N, B, H, D]

    # compute updated node emb, this step requires O(N)
    gumbels = (
        -torch.empty(key_prime.shape[:-1]+(K, ), memory_format=torch.legacy_contiguous_format).exponential_().log()
    ).to(query.device) / tau # [N, B, H, K]
    gumbels = gumbels.clamp(max=20.0)

    alarm(gumbels, "gumbels_g")
    
    key_t_gumbel = key_prime.unsqueeze(3) * gumbels.exp().unsqueeze(4) # [N, B, H, K, M]

    alarm(key_t_gumbel, "key_t_gumbel_g")
    
    z_num = numerator_gumbel(query_prime, key_t_gumbel, value) # [N, B, H, K, D]
    z_den = denominator_gumbel(query_prime, key_t_gumbel) # [N, B, H, K]

    alarm(z_num, "z_num_g")
    alarm(z_den, "z_den_g")

    z_num = z_num.permute(1, 0, 2, 3, 4) # [B, N, H, K, D]
    z_den = z_den.permute(1, 0, 2, 3) # [B, N, H, K]
    z_den = torch.unsqueeze(z_den, len(z_den.shape))
    z_output = torch.mean(z_num / z_den, dim=3) # [B, N, H, D]

    alarm(z_output, "z_output_g")

    start, end = edge_index
    query_end, key_start = query_prime[end], key_prime[start] # [E, B, H, M]
    edge_attn_num = torch.einsum("ebhm,ebhm->ebh", query_end, key_start) # [E, B, H]
    edge_attn_num = edge_attn_num.permute(1, 0, 2) # [B, E, H]
    attn_normalizer = denominator(query_prime, key_prime) # [N, B, H]
    edge_attn_dem = attn_normalizer[end]  # [E, B, H]
    edge_attn_dem = edge_attn_dem.permute(1, 0, 2) # [B, E, H]
    A_weight = edge_attn_num / edge_attn_dem # [B, E, H]

    return z_output, A_weight


def add_conv_relational_bias(x, edge_index, b, trans='sigmoid', device='cpu'):
    '''
    compute updated result by the relational bias of input adjacency
    the implementation is similar to the Graph Convolution Network with a (shared) scalar weight for each edge
    '''
    row, col = edge_index
    d_in = degree(col, x.shape[1]).float()
    d_norm_in = (1. / d_in[col]).sqrt()
    d_out = degree(row, x.shape[1]).float()
    d_norm_out = (1. / d_out[row]).sqrt()
    conv_output = []
    for i in range(x.shape[2]):
        if trans == 'sigmoid':
            b_i = F.sigmoid(b[i])
        elif trans == 'identity':
            b_i = b[i]
        else:
            raise NotImplementedError
        value = (torch.ones_like(row) * b_i * d_norm_in * d_norm_out).to(device)
        # adj_i = SparseTensor(row=col, col=row, value=value, sparse_sizes=(x.shape[1], x.shape[1]))
        # conv_output.append(matmul(adj_i, x[:, :, i]) )  # [B, N, D]

        adj_i = torch.sparse.FloatTensor(torch.stack([col, row]), value, torch.Size((x.shape[1], x.shape[1]))).to(device)
        # print(f"take: {x[:, :, i].size()}")
        # print(f"adj_i: {adj_i.size()}")
        
        conv_output.append(torch.sparse.mm(adj_i, x[:, :, i].squeeze(0).reshape(-1, x.shape[3])).unsqueeze(0).reshape(1, x.shape[1], x.shape[3]))
        
    conv_output = torch.stack(conv_output, dim=2) # [B, N, H, D]
    return conv_output


class NodeFormerConv(nn.Module):
    '''
    one layer of NodeFormer that attentive aggregates all nodes over a latent graph
    return: node embeddings for next layer, edge loss at this layer
    '''
    def __init__(self, in_channels, out_channels, num_heads, nb_random_features=10, use_gumbel=True,
                 nb_gumbel_sample=10, rb_order=0, rb_trans='sigmoid', device='cpu'):
        super(NodeFormerConv, self).__init__()
        self.device = device
        
        self.Wk = nn.Linear(in_channels, out_channels * num_heads)
        self.Wq = nn.Linear(in_channels, out_channels * num_heads)
        self.Wv = nn.Linear(in_channels, out_channels * num_heads)
        self.Wo = nn.Linear(out_channels * num_heads, out_channels)
        if rb_order >= 1:
            self.b = torch.nn.Parameter(torch.FloatTensor(rb_order, num_heads), requires_grad=True)

        self.out_channels = out_channels
        self.num_heads = num_heads
        self.nb_random_features = nb_random_features
        self.use_gumbel = use_gumbel
        self.nb_gumbel_sample = nb_gumbel_sample
        self.rb_order = rb_order
        self.rb_trans = rb_trans

    def reset_parameters(self):
        self.Wk.reset_parameters()
        self.Wq.reset_parameters()
        self.Wv.reset_parameters()
        self.Wo.reset_parameters()
        if self.rb_order >= 1:
            if self.rb_trans == 'sigmoid':
                torch.nn.init.constant_(self.b, 0.1)
            elif self.rb_trans == 'identity':
                torch.nn.init.constant_(self.b, 1.0)

    def forward(self, z, adjs, tau):
        B, N = z.size(0), z.size(1)
        query = self.Wq(z).reshape(-1, N, self.num_heads, self.out_channels)
        key = self.Wk(z).reshape(-1, N, self.num_heads, self.out_channels)
        value = self.Wv(z).reshape(-1, N, self.num_heads, self.out_channels)
        # print(value.size())

        alarm(query, "query")
        alarm(key, "key")
        alarm(value, "value")

        dim = query.shape[-1]
        seed = torch.ceil(torch.abs(torch.sum(query) * BIG_CONSTANT)).to(torch.int32)
        projection_matrix = create_projection_matrix(
            self.nb_random_features, dim, seed=seed).to(query.device)

        # compute all-pair message passing update and attn weight on input edges, requires O(N) or O(N + E)
        if self.use_gumbel and self.training:  # only using Gumbel noise for training
            z_next, weight = kernelized_gumbel_softmax(query,key,value,projection_matrix,adjs[0], self.nb_gumbel_sample,
                                                       tau)
        else:
            z_next, weight = kernelized_softmax(query, key, value, projection_matrix, adjs[0], tau)

        alarm(z_next, "z_next0")
        
        # compute update by relational bias of input adjacency, requires O(E)
        for i in range(self.rb_order):
            z_next += add_conv_relational_bias(value, adjs[i], self.b[i], self.rb_trans, self.device)

        alarm(z_next, "z_next1")

        # aggregate results of multiple heads
        z_next = self.Wo(z_next.flatten(-2, -1))

        alarm(z_next, "z_next2")

        row, col = adjs[0]
        d_in = degree(col, query.shape[1]).float()
        d_norm = 1. / d_in[col]
        d_norm_ = d_norm.reshape(1, -1, 1).repeat(1, 1, weight.shape[-1])
        link_loss = torch.mean(weight.log() * d_norm_)

        # print(f'mean: {link_loss}')

        return z_next, link_loss


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

        settings['num_features_in'] = len(self.dataset_info["op_dic"]) + len(list(self.dataset_info["non_eu_col"].keys())) + len(list(self.dataset_info["eu_col"].keys()))
        settings['num_features_out'] = 1

        in_channels = settings["num_features_in"]
        out_channels = settings['num_features_out']
        hidden_channels = settings["conv_dim"]
        num_layers = settings["num_layers"]
        num_heads = settings["num_heads"]
        nb_random_features = 30
        use_gumbel = True
        nb_gumbel_sample = settings["gumbel_sample"]
        self.rb_order = 2
        rb_trans = 'sigmoid'

        if torch.cuda.is_available():
            self.ngpu = torch.cuda.device_count()
        else:
            self.ngpu = 0

        self.dropout = 0.5
        self.k = settings["k"]
        self.activation = F.relu
        self.use_bn = True
        self.use_residual = True
        self.use_act = True
        self.use_jk = False
        
        self.convs = nn.ModuleList()
        self.fcs = nn.ModuleList()
        self.fcs.append(nn.Linear(in_channels, hidden_channels).to(self.device))
        self.bns = nn.ModuleList()
        self.bns.append(nn.LayerNorm(hidden_channels).to(self.device))
        for i in range(num_layers):
            self.convs.append(
                NodeFormerConv(hidden_channels, hidden_channels, num_heads=num_heads,
                               nb_random_features=nb_random_features, use_gumbel=use_gumbel,
                               nb_gumbel_sample=nb_gumbel_sample, rb_order=self.rb_order, rb_trans=rb_trans, device=device).to(self.device))
            self.bns.append(nn.LayerNorm(hidden_channels).to(self.device))

        if self.use_jk:
            self.fcs.append(nn.Linear(hidden_channels * num_layers + hidden_channels, out_channels).to(self.device))
        else:
            self.fcs.append(nn.Linear(hidden_channels, out_channels).to(self.device))

        # init weights
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.kaiming_normal_(p)

        self.optimizer = torch.optim.Adam(self.parameters(), weight_decay=5e-4, lr=settings['nn_lr'])
        self.scheduler = None

    def forward(self, batch):
        inputs, coords, targets, input_lenths = batch
        head_only = True

        x_l, indexer = padded_seq_to_vectors(inputs, input_lenths)
        y_l, _ = padded_seq_to_vectors(targets, input_lenths)
        c_l, _ = padded_seq_to_vectors(coords, input_lenths)

        # print(x_l.size())
        
        edge_index = knn_graph(c_l, k=self.k, batch=indexer)

        # print(f'edge_index: {edge_index}')
        # print(f'edge_index_size: {edge_index.size()}')

        all_nodes = torch.unique(edge_index).to(self.device)
        self_loops = torch.cat([all_nodes.unsqueeze(1), all_nodes.unsqueeze(1)], dim=1).transpose(0, 1)
        # print(f'self_loops: {self_loops}')
        # print(f'self_loops size: {self_loops.size()}')
        
        adj = torch.cat([edge_index, self_loops], dim=1)

        # print(f'adj_sl: {adj}')
        # print(f'adj_sl size: {adj.size()}')
        
        adjs_ = []
        adjs_.append(adj)
        for i in range(self.rb_order - 1):  # edge_index of high order adjacency
            adj = adj_mul(adj, adj, x_l.size(0))
            adjs_.append(adj)

        layer_ = []
        link_loss_ = []

        x = torch.cat([x_l, c_l], dim=1)
        # x = x_l

        alarm(x, "x")
        
        z = self.fcs[0](x.unsqueeze(0))

        alarm(z, "z0")
        
        if self.use_bn:
            z = self.bns[0](z)

        alarm(z, "z1")
        
        z = self.activation(z)

        alarm(z, "z2")
        
        z = F.dropout(z, p=self.dropout, training=self.training)

        alarm(z, "z3")
        
        layer_.append(z)

        for i, conv in enumerate(self.convs):
            z, link_loss = conv(z, adjs_, 0.25)

            alarm(z, f"zloop_{i}")
            alarm(link_loss, f"llloop_{i}")
            
            link_loss_.append(link_loss)
            if self.use_residual:
                z += layer_[i]

            alarm(z, f"zresidual_{i}")
            
            if self.use_bn:
                z = self.bns[i+1](z)

            alarm(z, f"zbn_{i}")
            
            if self.use_act:
                z = self.activation(z)

            alarm(z, f"zact_{i}")
            
            z = F.dropout(z, p=self.dropout, training=self.training)

            alarm(z, f"zdp_{i}")
            
            layer_.append(z)

        if self.use_jk: # use jk connection for each layer
            z = torch.cat(layer_, dim=-1)

        output = self.fcs[-1](z).squeeze(0)

        alarm(output, f"output")

        # print(f'output size: {output.size()}')

        if not head_only:
            return (output, link_loss_), y_l
        else:
            output_head = extract_first_element_per_batch(output, indexer)
            target_head = extract_first_element_per_batch(y_l, indexer)
            return (output_head, link_loss_), target_head
        

    def loss_func(self, model_output, target):
        output_head, link_loss = model_output

        loss_train = torch.nn.L1Loss()(output_head, target)

        # print(f"loss_train: {loss_train}")
        # print(f"link_loss: {link_loss}")
        # print(f"sum(link_loss): {sum(link_loss)}")
        # print(f"len(link_loss): {len(link_loss)}")

        if self.ngpu <= 1:
            loss_train -= self.settings['lambda'] * sum(link_loss) / len(link_loss)
        else:
            loss_train -= self.settings['lambda'] * sum(sum(link_loss)) / len(link_loss)

        alarm(loss_train, f"loss_train")

        return loss_train, output_head, target
