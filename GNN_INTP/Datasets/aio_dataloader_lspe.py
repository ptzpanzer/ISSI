import os
import json
import pickle
import concurrent.futures
import torch
import pandas as pd
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from torch_geometric.nn import knn_graph
import dgl
from scipy import sparse as sp

import support_functions


def chunkify(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]


def encode_and_bind(original_dataframe, feature_to_encode, possible_values):
    enc_df = original_dataframe.copy()
    for value in possible_values:
        enc_df.loc[:, feature_to_encode + '_' + str(value)] = (enc_df[feature_to_encode] == value).astype(int)
    res = enc_df.drop([feature_to_encode], axis=1)
    return res


def add_eig_vec(g, pos_enc_dim):
    """
     Graph positional encoding v/ Laplacian eigenvectors
     This func is for eigvec visualization, same code as positional_encoding() func,
     but stores value in a diff key 'eigvec'
    """

    # Laplacian
    A = g.adj_external(scipy_fmt="csr")
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort() # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:,idx])
    g.ndata['eigvec'] = torch.from_numpy(EigVec[:,1:pos_enc_dim+1]).float() 

    # zero padding to the end if n < pos_enc_dim
    n = g.number_of_nodes()
    if n <= pos_enc_dim:
        g.ndata['eigvec'] = F.pad(g.ndata['eigvec'], (0, pos_enc_dim - n + 1), value=float('0'))

    return g


def init_positional_encoding(g, pos_enc_dim, type_init):
    """
        Initializing positional encoding with RWPE
    """
    
    n = g.number_of_nodes()

    if type_init == 'rand_walk':
        # Geometric diffusion features with Random Walk
        A = g.adj_external(scipy_fmt="csr")
        Dinv = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -1.0, dtype=float) # D^-1
        RW = A * Dinv  
        M = RW
        
        # Iterate
        nb_pos_enc = pos_enc_dim
        PE = [torch.from_numpy(M.diagonal()).float()]
        M_power = M
        for _ in range(nb_pos_enc-1):
            M_power = M_power * M
            PE.append(torch.from_numpy(M_power.diagonal()).float())
        PE = torch.stack(PE,dim=-1)
        g.ndata['pos_enc'] = PE  
    
    return g


class IntpDataset(Dataset):
    def __init__(self, settings, mask_distance, call_name):
        # save init parameters
        self.settings = settings
        self.mask_distance = mask_distance
        self.call_name = call_name

        # load dataset info
        self.origin_path = f"./Datasets/{settings['dataset']}/"
        with open(self.origin_path + f'meta_data.json', 'r') as f:
            self.dataset_info = json.load(f)
        with open(self.origin_path + f"Folds_Info/norm_{settings['fold']}_{settings['holdout']}.json", 'r') as f:
            self.dic_op_minmax = json.load(f)
        with open(self.origin_path + f"Folds_Info/divide_set_{settings['fold']}_{settings['holdout']}.info", 'rb') as f:
            divide_set = pickle.load(f)

        self.out_root = f"{settings['tmpdir']}/{settings['dataset']}/pkl/{settings['fold']}_{settings['holdout']}_{call_name}_{mask_distance}/"

        if not os.path.exists(self.out_root + '0/0.pkl'):
            # load file list
            if call_name == 'train':
                call_scene_list = divide_set[0]
            elif call_name == 'test':
                call_scene_list = divide_set[1]
            elif call_name == 'eval':
                call_scene_list = divide_set[2]
            if settings['debug'] and len(call_scene_list) > 100:
                call_scene_list = call_scene_list[:100]
    
            # do op filtering and normalization in the parallel fashion
            self.total_df_dict = {}
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.process_child, file_name) for file_name in call_scene_list]
                results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
            for file_name, file_content in results:
                self.total_df_dict[file_name] = file_content
    
            print(f"Length of df dict: {len(list(self.total_df_dict.keys()))}", flush=True)
    
            self.call_list = []
            target_op_range = self.dataset_info["holdouts"][str(self.settings['holdout'])][self.call_name]
            for scene in call_scene_list:
                df = self.total_df_dict[scene]
                target_row_index_list = list(df[df['op'].isin(target_op_range)].index)
                target_row_index_list.sort()
                for index in target_row_index_list:
                    self.call_list.append((scene, index))
            tail_index = (len(self.call_list) // 256) * 256
            self.call_list = self.call_list[:tail_index]
    
            print(f"Length of call list: {len(self.call_list)}", flush=True)

            for i in range(0, len(self.call_list), 10000):
                support_functions.build_folder_and_clean(self.out_root + f"{i}/")
    
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures_g = [executor.submit(self.process_g, i) for i in range(len(self.call_list))]
                results_g = [futures_g.result() for futures_g in concurrent.futures.as_completed(futures_g)]

            self.dataset_len = len(self.call_list)

        else:
            print(f"Loading from disk...", flush=True)

            chunk_list = []
            for subdir in os.listdir(self.out_root):
                if os.path.isdir(os.path.join(self.out_root, subdir)):
                    chunk_list.append(int(subdir))
            max_chunk = max(chunk_list)

            file_list = []
            for file in os.listdir(self.out_root + f'{max_chunk}/'):
                if '.pkl' in file:
                    file_list.append(int(file.split('.')[0]))
            last_file = max(file_list)

            self.dataset_len = last_file + 1

            if settings['debug'] and self.dataset_len > 6399:
                self.dataset_len = 6399
                
            print(f"Length of call list: {self.dataset_len}", flush=True)

    
    def __len__(self):
        return self.dataset_len


    def norm(self, d):
        d_list = []
        for op in d['op'].unique():
            d_op = d[d['op']==op].copy()
            if op in self.dataset_info["tgt_logical"]:
                op_norm = self.dataset_info["tgt_op"]
            else:
                op_norm = op
            d_op['Result_norm'] = (d_op['Result'] - self.dic_op_minmax[op_norm][0]) / (self.dic_op_minmax[op_norm][1] - self.dic_op_minmax[op_norm][0])
            d_list.append(d_op)
        return pd.concat(d_list, axis=0, ignore_index=False).drop(columns=['Result'])


    def process_child(self, filename):
        df = pd.read_csv(self.origin_path + 'Dataset_Separation/' + filename, sep=';')
        # drop everything in bad quality
        df = df[df['Thing']>=self.dataset_info['lowest_rank']]
        # drop everything with non-whitelisted op
        op_whitelist = list(self.dataset_info["op_dic"].keys())
        for holdout in self.dataset_info["holdouts"].keys():
            op_whitelist = op_whitelist + self.dataset_info["holdouts"][holdout]["train"] + self.dataset_info["holdouts"][holdout]["test"] + self.dataset_info["holdouts"][holdout]["eval"]
        op_whitelist = list(set(op_whitelist))
        df = df[df['op'].isin(op_whitelist)]
        # normalize all values (coordinates will be normalized later)
        df = self.norm(d=df)
        return filename, df

    
    def process_g(self, idx):
        # load data item
        df = self.total_df_dict[self.call_list[idx][0]]
        target_row_index = self.call_list[idx][1]
        
        # keep target row
        target_row = pd.DataFrame(df.loc[target_row_index]).transpose()
        target_row = target_row.apply(pd.to_numeric, errors='ignore')
        df = df.drop(target_row_index)
        # clean unrelated labels
        keep_op_range = list(set(
            list(self.dataset_info["op_dic"].keys()) + 
            self.dataset_info["holdouts"][str(self.settings['holdout'])]["train"]
        ))
        df = df[df['op'].isin(keep_op_range)]
        
        while True:
            if self.call_name == 'eval':
                data_augmentation_option = 'dst'
                this_mask = self.mask_distance
                df_filtered = df.loc[(abs(df['Longitude'] - target_row['Longitude'].values[0]) + abs(df['Latitude'] - target_row['Latitude'].values[0])) >= this_mask, :].copy()
            else:
                df_filtered = df

            df_filtered.loc[df_filtered['op'].isin(self.dataset_info["tgt_logical"]), 'op'] = self.dataset_info["tgt_op"]
            target_row.loc[target_row['op'].isin(self.dataset_info["tgt_logical"]), 'op'] = self.dataset_info["tgt_op"]

            df_known = df_filtered[df_filtered['op']==self.dataset_info["tgt_op"]].copy()
            df_auxil = df_filtered[df_filtered['op']!=self.dataset_info["tgt_op"]].copy()
            # check and quit loop
            if len(df_known) >= 1 and len(df_auxil) >= 1:
                break

        df_known = self.norm_fcol(df_known)
        df_auxil = self.norm_fcol(df_auxil)
        target_row = self.norm_fcol(target_row)

        graph_candidates = pd.concat([target_row, df_known], axis=0, ignore_index=True)
        
        coords = torch.from_numpy(graph_candidates[list(self.dataset_info["eu_col"].keys())].values).float()
        
        answers = torch.from_numpy(graph_candidates[['Result_norm']].values).float()

        full_df = pd.concat([df_known, df_auxil], axis=0, ignore_index=True)
        aggregated_df = full_df.groupby(list(self.dataset_info["eu_col"].keys()) + ['op']).mean().reset_index()
        features = torch.zeros((len(graph_candidates), len(self.dataset_info["op_dic"]) + 1))
        possible_values = list(self.dataset_info["op_dic"].keys())
        possible_values.sort()
        for op in possible_values:
            aggregated_df_op = aggregated_df[aggregated_df['op']==op]
            interpolated_grid = torch.zeros((len(graph_candidates), 1))
            if len(aggregated_df_op) != 0:
                c = aggregated_df_op[list(self.dataset_info["eu_col"].keys())].values
                v = aggregated_df_op['Result_norm'].values
                ci = graph_candidates[list(self.dataset_info["eu_col"].keys())].values
                interpolated_values = self.idw_interpolation(c, v, ci)
                interpolated_grid = torch.from_numpy(interpolated_values).reshape((len(graph_candidates), 1))
            features[:, self.dataset_info["op_dic"][op]:self.dataset_info["op_dic"][op]+1] = interpolated_grid
        
        conditions = graph_candidates['op'] == self.dataset_info["tgt_op"]
        features[:, -1] = torch.from_numpy(np.where(conditions, graph_candidates['Thing'], (self.dataset_info['lowest_rank']-1)/self.dataset_info["non_eu_col"]["Thing"][1]))
        features = features.float()

        features[0, :] = 0

        node_features = torch.cat([features, coords], dim=-1)
        edge_index = knn_graph(coords, k=len(node_features)-1)
        graph = dgl.graph((edge_index[0], edge_index[1]))
        graph.ndata['feat'] = node_features
        graph.edata['feat'] = torch.ones(edge_index.size(1), 1)
        graph = init_positional_encoding(graph, self.settings['pos_enc_dim'], self.settings['pe_init'])
        graph = add_eig_vec(graph, self.settings['pos_enc_dim'])
        
        label = answers[0]

        chunk_idx = (idx // 10000) * 10000
        with open(self.out_root + f'{chunk_idx}/{idx}.pkl', 'wb') as f:
            pickle.dump((graph, label), f)

        print(f"{idx} Processed!", end="\r", flush=True)

        return True
        

    def distance_matrix(self, obs, interp):
        n_obs, n_dim_obs = obs.shape
        n_interp, n_dim_interp = interp.shape
        # Expand dimensions to broadcast
        obs = obs[:, np.newaxis, :]
        interp = interp[np.newaxis, :, :]
        # Calculate differences along each dimension
        diffs = obs - interp
        # Calculate squared distances along each dimension
        squared_diffs = diffs ** 2
        # Sum along dimensions to get squared distance
        squared_distances = np.sum(squared_diffs, axis=2)
        # Take square root to get actual distance
        distances = np.sqrt(squared_distances)
        return distances
    
    
    def idw_interpolation(self, obs_coords, values, interp_coords, p=2):
        dist = self.distance_matrix(obs_coords, interp_coords)
        # In IDW, weights are 1 / distance
        weights = 1.0 / (dist + 1e-12) ** p
        # Make weights sum to one
        weights /= np.sum(weights, axis=0, keepdims=True)
        # Multiply the weights for each interpolated point by all observed Z-values
        interpolated_values = np.dot(values, weights)
        return interpolated_values


    def norm_fcol(self, df):
        rtn = df.copy()
        # Norm other columns
        for col, (min_val, max_val) in self.dataset_info["eu_col"].items():
            rtn[col] = (rtn[col] - min_val) / (max_val - min_val)
        for col, (min_val, max_val) in self.dataset_info["non_eu_col"].items():
            rtn[col] = (rtn[col] - min_val) / (max_val - min_val)
        return rtn


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        chunk_idx = (idx // 10000) * 10000
        with open(self.out_root + f'{chunk_idx}/{idx}.pkl', 'rb') as f:
            graph, label = pickle.load(f)

        return graph, label


    def collate_fn(self, samples):
        # The input samples is a list of pairs (graph, label).
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(np.array(labels))
        tab_sizes_n = [ graphs[i].number_of_nodes() for i in range(len(graphs))]
        tab_snorm_n = [ torch.FloatTensor(size,1).fill_(1./float(size)) for size in tab_sizes_n ]
        snorm_n = torch.cat(tab_snorm_n).sqrt()  
        batched_graph = dgl.batch(graphs)       
        
        return batched_graph, labels, snorm_n
        
