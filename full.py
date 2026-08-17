import argparse
import datetime
import logging
import os
import time
import copy
import random
import multiprocessing
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from fastdtw import fastdtw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# ------------------------------ 备用 GCN ------------------------------
try:
    from models.layers.GCN import GCN
except ImportError:
    class GCN(nn.Module):
        def __init__(self, c_in, c_out, dropout, support_len=1, order=2):
            super().__init__()
            self.nconv = lambda x, A: torch.einsum('ncvl,nvw->ncwl', x, A)
            self.mlp = nn.Conv2d(c_in * (order * support_len + 1), c_out, kernel_size=1)
            self.dropout = dropout
            self.order = order
        def forward(self, x, support):
            out = [x]
            x1 = self.nconv(x, support)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, support)
                out.append(x2)
                x1 = x2
            h = torch.cat(out, dim=1)
            h = self.mlp(h)
            h = F.dropout(h, self.dropout, training=self.training)
            return h

# ------------------------------ 基础组件 (RevIN, 注意力等) ------------------------------
class RevIN(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True, subtract_last=False):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()
    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))
    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x
    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()
    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x
    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            safe_weight = self.affine_weight.abs() + self.eps
            x = x / safe_weight
        x = x * self.stdev
        x = x + self.mean
        return x

def channel_shuffle(x, groups):
    b, c, n, t = x.shape
    c_per_group = c // groups
    x = x.view(b, groups, c_per_group, n, t)
    x = x.transpose(1, 2).contiguous()
    x = x.view(b, -1, n, t)
    return x

class ParallelSTAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.0):
        super().__init__()
        self.spatial_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.temporal_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=dropout)
        self.norm_sp = nn.LayerNorm(dim)
        self.norm_tm = nn.LayerNorm(dim)
        self.fusion = nn.Linear(dim * 2, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, t, c = x.shape
        x_sp = x.permute(0, 2, 1, 3).reshape(b * t, n, c)
        out_sp, _ = self.spatial_attn(x_sp, x_sp, x_sp)
        out_sp = out_sp.reshape(b, t, n, c).permute(0, 2, 1, 3)
        out_sp = self.norm_sp(out_sp)
        x_tm = x.reshape(b * n, t, c)
        out_tm, _ = self.temporal_attn(x_tm, x_tm, x_tm)
        out_tm = out_tm.reshape(b, n, t, c)
        out_tm = self.norm_tm(out_tm)
        fused = torch.cat([out_sp, out_tm], dim=-1)
        fused = self.fusion(fused)
        fused = self.dropout(fused)
        return fused

class JointAttention(nn.Module):
    def __init__(self, in_channels, num_heads, partial):
        super().__init__()
        self.parallel_attn = ParallelSTAttention(in_channels, num_heads)
        self.partial = partial
    def forward(self, x):
        b, c, n, t = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.parallel_attn(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = channel_shuffle(x, self.partial)
        return x

class JointConv(nn.Module):
    def __init__(self, in_channels, partial):
        super().__init__()
        k_size = 3
        self.partial = partial
        pc = int(in_channels * (1 / partial))
        self.T_conv = nn.Conv2d(pc, pc, kernel_size=(1, k_size), padding=(0,1))
        self.S_conv = nn.Conv2d(pc, pc, kernel_size=(k_size,1), padding=(1,0))
        self.ST_conv = nn.Conv2d(pc, pc, kernel_size=(k_size,k_size), padding=(1,1))
        self.split_indexes = (in_channels - 3 * pc, pc, pc, pc)
    def forward(self, x):
        b, c, n, t = x.shape
        x_id, x_st, x_s, x_t = torch.split(x, self.split_indexes, dim=1)
        output = torch.cat((x_id, self.ST_conv(x_st), self.S_conv(x_s), self.T_conv(x_t)), dim=1)
        output = channel_shuffle(output, self.partial)
        return output

# ------------------------------ 动态图构造器（不变） ------------------------------
class MemoryEnhancedGraphConstructor(nn.Module):
    def __init__(self, node_num, feat_dim, memory_prototypes, k_neighbors=5, top_k_mem=5, update_freq=50):
        super().__init__()
        self.k = k_neighbors
        self.top_k_mem = top_k_mem
        self.update_freq = update_freq
        self.register_buffer('memory_prototypes', memory_prototypes)
        self.proto_dim = memory_prototypes.size(-1)
        self.input_proj = None
        self.edge_mlp = nn.Sequential(
            nn.Linear(self.proto_dim * 2, self.proto_dim),
            nn.ReLU(),
            nn.Linear(self.proto_dim, 1)
        )
        self.cached_adj = None
        self.counter = 0

    def forward(self, x):
        self.counter += 1
        b = x.shape[0]
        if (self.cached_adj is None or self.cached_adj.shape[0] != b or
                self.counter % self.update_freq == 0):
            h = x.mean(dim=2)
            if self.input_proj is None:
                input_dim = h.shape[-1]
                if input_dim != self.proto_dim:
                    self.input_proj = nn.Linear(input_dim, self.proto_dim).to(h.device)
                else:
                    self.input_proj = nn.Identity()
            h = self.input_proj(h)
            sim_to_mem = torch.matmul(h, self.memory_prototypes.T)
            topk_sim, topk_idx = torch.topk(sim_to_mem, k=self.top_k_mem, dim=-1)
            weights = F.softmax(topk_sim, dim=-1)
            retrieved = self.memory_prototypes[topk_idx]
            enhanced_h = h + (weights.unsqueeze(-1) * retrieved).sum(dim=2)
            sim = torch.matmul(enhanced_h, enhanced_h.transpose(1, 2))
            topk_val, topk_idx = torch.topk(sim, k=self.k + 1, dim=-1)
            topk_idx = topk_idx[:, :, 1:]
            b, n, k = topk_idx.shape
            h_i = enhanced_h.unsqueeze(2).expand(-1, -1, k, -1)
            h_j = torch.gather(enhanced_h.unsqueeze(1).expand(-1, n, -1, -1), dim=2,
                               index=topk_idx.unsqueeze(-1).expand(-1, -1, -1, enhanced_h.size(-1)))
            pair_feat = torch.cat([h_i, h_j], dim=-1)
            edge_weights = torch.sigmoid(self.edge_mlp(pair_feat)).squeeze(-1)
            adj = torch.zeros(b, n, n, dtype=torch.float32, device=x.device)
            adj.scatter_(2, topk_idx, edge_weights.float())
            adj = (adj + adj.transpose(1, 2)) / 2.0
            self.cached_adj = adj.detach()
        return self.cached_adj

class SparseDynamicGraphConstructor(nn.Module):
    def __init__(self, node_num, feat_dim, k_neighbors=5, update_freq=50):
        super().__init__()
        self.k = k_neighbors
        self.update_freq = update_freq
        self.edge_mlp = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.ReLU(),
            nn.Linear(feat_dim, 1)
        )
        self.cached_adj = None
        self.counter = 0

    def forward(self, x):
        self.counter += 1
        b = x.shape[0]
        if (self.cached_adj is None or self.cached_adj.shape[0] != b or
                self.counter % self.update_freq == 0):
            h = x.mean(dim=2)
            sim = torch.matmul(h, h.transpose(1, 2))
            topk_val, topk_idx = torch.topk(sim, k=self.k + 1, dim=-1)
            topk_idx = topk_idx[:, :, 1:]
            b, n, k = topk_idx.shape
            h_i = h.unsqueeze(2).expand(-1, -1, k, -1)
            h_j = torch.gather(h.unsqueeze(1).expand(-1, n, -1, -1), dim=2,
                               index=topk_idx.unsqueeze(-1).expand(-1, -1, -1, h.size(-1)))
            pair_feat = torch.cat([h_i, h_j], dim=-1)
            edge_weights = torch.sigmoid(self.edge_mlp(pair_feat)).squeeze(-1)
            adj = torch.zeros(b, n, n, dtype=torch.float32, device=x.device)
            adj.scatter_(2, topk_idx, edge_weights.float())
            adj = (adj + adj.transpose(1, 2)) / 2.0
            self.cached_adj = adj.detach()
        return self.cached_adj

# ------------------------------ 主模型 ASTCL (不变) ------------------------------
class ASTCL(nn.Module):
    def __init__(self, channels, heads, depth, partial, num_features, num_timesteps_input,
                 num_timesteps_output, node_num, dropout=0.5, target_node=0,
                 use_dynamic_graph=False, memory_prototypes=None, use_memory_enhance=False,
                 graph_update_freq=50):
        super().__init__()
        self.target_node = target_node
        self.depth = depth
        self.partial = partial
        self.use_dynamic_graph = use_dynamic_graph
        self.use_memory_enhance = use_memory_enhance
        self.revin_layer = RevIN(num_features, affine=True, subtract_last=False)
        if use_dynamic_graph:
            if use_memory_enhance and memory_prototypes is not None:
                self.graph_constructor = MemoryEnhancedGraphConstructor(
                    node_num, num_features, memory_prototypes, k_neighbors=5, top_k_mem=5,
                    update_freq=graph_update_freq)
            else:
                self.graph_constructor = SparseDynamicGraphConstructor(
                    node_num, num_features, k_neighbors=5, update_freq=graph_update_freq)
        else:
            self.graph_constructor = None
        self.start_conv = nn.ModuleList()
        self.joint_conv = nn.ModuleList()
        self.joint_attention = nn.ModuleList()
        for i in range(depth):
            in_c = num_features if i == 0 else channels[i]
            out_c = channels[i+1]
            self.start_conv.append(nn.Conv2d(in_c, out_c, kernel_size=(1,1)))
            self.joint_conv.append(JointConv(out_c, self.partial))
            self.joint_attention.append(JointAttention(out_c, heads[i], self.partial))
        self.end_conv = nn.Conv2d(channels[-1], num_features, kernel_size=3, padding=1)
        self.dropout_layer = nn.Dropout(p=dropout)
        self.gcn = nn.ModuleList()
        for i in range(depth):
            self.gcn.append(GCN(c_in=num_features, c_out=channels[i+1], dropout=0.2, support_len=1, order=2))

    def forward(self, input, adj=None):
        input = input.float()
        x = self.revin_layer(input, 'norm')
        x = x.permute(0, 3, 1, 2).contiguous()
        if self.use_dynamic_graph and self.graph_constructor is not None:
            x_graph = x.permute(0, 2, 3, 1).contiguous()
            dynamic_adj = self.graph_constructor(x_graph)
            adj_used = dynamic_adj
        else:
            adj_used = adj
        x_gcn = []
        for i in range(self.depth):
            x_gcn.append(self.gcn[i](x, adj_used))
        for i in range(self.depth):
            x = F.leaky_relu(self.start_conv[i](x), 0.2)
            shortcut = x
            x = F.leaky_relu(self.joint_attention[i](x), 0.2)
            x = F.leaky_relu(self.joint_conv[i](x), 0.2)
            x = x + x_gcn[i] + shortcut
        x = self.dropout_layer(x)
        output = self.end_conv(x)
        output = output.permute(0, 2, 3, 1).contiguous()
        output = self.revin_layer(output, 'denorm')
        output = output[..., 0]
        output = output[:, self.target_node, :]
        return output

# ------------------------------ 数据处理函数（修正泄露） ------------------------------
def load_data(data_path, sensor_percent, max_nodes=100):
    adj_mat_path = os.path.join(data_path, "adj_mat.npy")
    feature_path = os.path.join(data_path, "node_values.npy")
    A = np.load(adj_mat_path).astype(np.float32)
    X = np.load(feature_path).astype(np.float32)
    if X.shape[0] > max_nodes:
        X = X[:max_nodes, ...]
        A = A[:max_nodes, :max_nodes]
    if sensor_percent != 1:
        num_nodes = X.shape[0]
        partial = int(num_nodes * sensor_percent)
        selected = np.random.choice(num_nodes, partial, replace=False)
        X = X[selected, ...]
        A = A[selected][:, selected]
    return A, X

GLOBAL_NODE_RANK = None

def _dtw_distance_pair(i, j, X_sampled, radius=5):
    dist, _ = fastdtw(X_sampled[i], X_sampled[j], radius=radius)
    return i, j, dist

def compute_global_node_similarity(X, num_nodes, sample_rate=0.1, dtw_radius=5):
    if sample_rate < 1.0:
        timesteps = X.shape[1]
        sample_idx = np.random.choice(timesteps, int(timesteps * sample_rate), replace=False)
        sample_idx.sort()
        X_sampled = X[:, sample_idx]
    else:
        X_sampled = X
    dist_matrix = np.zeros((num_nodes, num_nodes))
    tasks = [(i,j) for i in range(num_nodes) for j in range(i+1, num_nodes)]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        futures = {executor.submit(_dtw_distance_pair, i, j, X_sampled, dtw_radius): (i,j) for i,j in tasks}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Computing DTW"):
            i, j, dist = future.result()
            dist_matrix[i,j] = dist
            dist_matrix[j,i] = dist
    np.fill_diagonal(dist_matrix, 0)
    node_rank = np.argsort(dist_matrix, axis=1)
    return node_rank

def _process_one_node_flow(args):
    node, X, num_nodes, t_in, t_out, keep_days, interval, skip_graph, semantic_features = args
    traffic_feature_idx = 0
    num_times = X.shape[1]
    samples_node = []
    targets_node = []
    graphs_node = []
    if keep_days != 0:
        time_stamps = int(keep_days * interval) + (t_in + t_out)
        choosen = range(num_times - time_stamps, num_times - (t_in + t_out))
        indices = [(i, i + (t_in + t_out)) for i in choosen]
    else:
        # 验证集/测试集：按时间顺序滑动，不随机采样
        indices = [(i, i + (t_in + t_out)) for i in range(num_times - (t_in + t_out))]
        # 如果样本太多，按固定步长抽取（保持时间顺序）
        if len(indices) > 2000:  # 限制最大数量
            step = len(indices) // 2000
            indices = indices[::step]
    for i,j in indices:
        node_ids = GLOBAL_NODE_RANK[node, :num_nodes].tolist()
        node_ids = [nid for nid in node_ids if nid != node][:num_nodes]
        while len(node_ids) < num_nodes:
            node_ids.append(node)
        sample_distant = X[node_ids, i:i+t_in, :]
        if semantic_features is not None:
            semantic_win = semantic_features[i:i+t_in]
            semantic_win_expanded = np.repeat(semantic_win[np.newaxis, :, :], num_nodes, axis=0)
            sample = np.concatenate([sample_distant, semantic_win_expanded], axis=-1)
        else:
            sample = sample_distant
        if skip_graph:
            graph = np.array([0])
        else:
            graph = np.eye(num_nodes)
        target = X[node, i+t_in:j, traffic_feature_idx]
        samples_node.append(sample)
        targets_node.append(target)
        graphs_node.append(graph)
    return samples_node, targets_node, graphs_node

def prepare_samples_targets_list_flow(A, X, num_nodes, t_in, t_out, keep_days=7, interval=288,
                                       debug_flag=False, target_nodes='all', grid=False,
                                       parallel=False, skip_graph=True, semantic_features=None):
    if target_nodes != 'all':
        target_node_list = target_nodes
    else:
        target_node_list = range(X.shape[0])
    if parallel:
        num_workers = min(8, os.cpu_count())
        ctx = multiprocessing.get_context('spawn')
        with ctx.Pool(processes=num_workers) as pool:
            args_list = [(node, X, num_nodes, t_in, t_out, keep_days, interval, skip_graph, semantic_features) for node in target_node_list]
            results = list(tqdm(pool.imap(_process_one_node_flow, args_list), total=len(args_list), desc="Processing nodes"))
        samples, targets, graphs = [], [], []
        for s,t,g in results:
            samples.extend(s); targets.extend(t); graphs.extend(g)
        return samples, targets, graphs
    else:
        samples, targets, graphs = [], [], []
        for node in tqdm(target_node_list, desc="Processing nodes"):
            s_node, t_node, g_node = _process_one_node_flow((node, X, num_nodes, t_in, t_out, keep_days, interval, skip_graph, semantic_features))
            samples.extend(s_node); targets.extend(t_node); graphs.extend(g_node)
        return samples, targets, graphs

def generate_semantic_features(num_samples, start_date='2022-01-01', interval_minutes=5):
    start = pd.Timestamp(start_date)
    timestamps = [start + pd.Timedelta(minutes=interval_minutes * i) for i in range(num_samples)]
    semantic_features = []
    for ts in timestamps:
        hour = ts.hour + ts.minute / 60.0
        hour_sin = np.sin(2 * np.pi * hour / 24)
        hour_cos = np.cos(2 * np.pi * hour / 24)
        weekday = ts.weekday()
        is_weekend = 1 if weekday >= 5 else 0
        month = ts.month
        if month in [6,7,8]:
            weather = 'rain' if 10 <= hour < 16 else 'overcast'
        elif month in [12,1,2]:
            weather = 'sunny' if 9 <= hour < 18 else 'overcast'
        else:
            weather = 'sunny' if hour < 14 else 'overcast'
        weather_onehot = [1 if weather=='sunny' else 0, 1 if weather=='overcast' else 0, 1 if weather=='rain' else 0]
        semantic_features.append([hour_sin, hour_cos, weekday, is_weekend] + weather_onehot)
    return np.array(semantic_features, dtype=np.float32)

class CrossDataset(Dataset):
    def __init__(self, samples_path, targets_path, graph_path, num_nodes=None, augment=False):
        self._samples = np.load(samples_path, mmap_mode='r')
        self._targets = np.load(targets_path, mmap_mode='r')
        self._num_nodes = num_nodes
        self.augment = augment
        tmp = np.load(graph_path, mmap_mode='r')
        if tmp.shape == (1,) and tmp[0] == 0:
            self._dynamic_graph = True
            self._eye = np.eye(self._num_nodes, dtype=np.float32)
        else:
            self._adj = tmp
            self._dynamic_graph = False
            self._eye = None
    def __len__(self):
        return len(self._samples)
    def __getitem__(self, idx):
        if self._dynamic_graph:
            graph = self._eye
        else:
            graph = self._adj[idx]
        sample = self._samples[idx].astype(np.float32)
        if self.augment and np.random.rand() < 0.1:
            mask = np.random.rand(sample.shape[1]) < 0.05
            sample[:, mask, :] = 0
        target = self._targets[idx].astype(np.float32)
        return sample, target, graph

# ===================== 关键修正：preprocess_dataset 仅用训练集计算 GLOBAL_NODE_RANK =====================
def preprocess_dataset(data, t_in=12, t_out=3, num_nodes=15, keep_days=7, percent=1, interval=288,
                       train=True, val=False, test=False, debug=False, target_nodes='all',
                       test_flag=False, type='speed', parallel=False, skip_graph=True,
                       dtw_radius=5, max_nodes=100):
    assert isinstance(data, str)
    suffix = f'_{t_in}_{t_out}_{keep_days}_{percent}_{num_nodes}'
    if target_nodes != 'all':
        suffix = f'_{t_in}_{t_out}_{keep_days}_{percent}_{num_nodes}'
    train_samples_path = os.path.join(data, f'train_samples{suffix}.npy')
    train_targets_path = os.path.join(data, f'train_targets{suffix}.npy')
    train_graph_path = os.path.join(data, f'train_graph{suffix}.npy')
    val_samples_path = os.path.join(data, f'val_samples{suffix}.npy')
    val_targets_path = os.path.join(data, f'val_targets{suffix}.npy')
    val_graph_path = os.path.join(data, f'val_graph{suffix}.npy')
    test_samples_path = os.path.join(data, f'test_samples{suffix}.npy')
    test_targets_path = os.path.join(data, f'test_targets{suffix}.npy')
    test_graph_path = os.path.join(data, f'test_graph{suffix}.npy')
    if all(os.path.exists(p) for p in [train_samples_path, train_targets_path, train_graph_path,
                                       val_samples_path, val_targets_path, val_graph_path,
                                       test_samples_path, test_targets_path, test_graph_path]):
        print("Preprocessed data found, loading from cache...")
        return (train_samples_path, train_targets_path, train_graph_path,
                val_samples_path, val_targets_path, val_graph_path,
                test_samples_path, test_targets_path, test_graph_path)
    print("No cache found, preprocessing...")
    A, X = load_data(data, percent, max_nodes=max_nodes)
    global GLOBAL_NODE_RANK

    # 切分数据
    if type == 'speed':
        cut_point1 = int(X.shape[1] * 0.7)
    else:
        cut_point1 = int(X.shape[1] * 0.6)
    cut_point2 = int(X.shape[1] * 0.8)
    train_X = np.expand_dims(X[:, :cut_point1, 0], axis=-1)
    val_X = np.expand_dims(X[:, cut_point1:cut_point2, 0], axis=-1)
    test_X = np.expand_dims(X[:, cut_point2:, 0], axis=-1)

    # [FIX] 仅使用训练集数据计算全局节点相似度排名，避免未来信息泄露
    if GLOBAL_NODE_RANK is None:
        print("Computing global node similarity using **training data only**...")
        train_X_for_dtw = train_X[:, :, 0]  # shape: (num_nodes, train_len)
        GLOBAL_NODE_RANK = compute_global_node_similarity(
            train_X_for_dtw, train_X_for_dtw.shape[0],
            sample_rate=0.1, dtw_radius=dtw_radius
        )

    def gen_sem(ts_len):
        return generate_semantic_features(ts_len, interval_minutes=interval)
    semantic_train = gen_sem(train_X.shape[1]) if train else None
    semantic_val = gen_sem(val_X.shape[1]) if val else None
    semantic_test = gen_sem(test_X.shape[1]) if test else None

    def save_set(X_data, semantic, name):
        samples, targets, graphs = prepare_samples_targets_list_flow(
            A, X_data, num_nodes, t_in, t_out,
            keep_days=keep_days if name=='train' else 0,
            interval=interval, debug_flag=debug,
            target_nodes=target_nodes, parallel=parallel,
            skip_graph=skip_graph, semantic_features=semantic
        )
        print(f'Saving {name} set...')
        np.save(os.path.join(data, f'{name}_samples{suffix}.npy'), np.array(samples))
        np.save(os.path.join(data, f'{name}_targets{suffix}.npy'), np.array(targets))
        if skip_graph:
            np.save(os.path.join(data, f'{name}_graph{suffix}.npy'), np.array([0]))
        else:
            np.save(os.path.join(data, f'{name}_graph{suffix}.npy'), np.array(graphs))
        del samples, targets, graphs
        import gc; gc.collect()

    if train:
        save_set(train_X, semantic_train, 'train')
    if val:
        save_set(val_X, semantic_val, 'val')
    if test:
        save_set(test_X, semantic_test, 'test')

    return (train_samples_path, train_targets_path, train_graph_path,
            val_samples_path, val_targets_path, val_graph_path,
            test_samples_path, test_targets_path, test_graph_path)

# ------------------------------ 辅助函数（不变） ------------------------------
def masked_MAE(pred, true):
    mask = ~np.isnan(true)
    if mask.sum() == 0: return np.nan
    return np.mean(np.abs(pred[mask] - true[mask]))

def masked_RMSE(pred, true):
    mask = ~np.isnan(true)
    if mask.sum() == 0: return np.nan
    return np.sqrt(np.mean((pred[mask] - true[mask])**2))

def masked_MAPE(pred, true):
    mask = ~np.isnan(true) & (true != 0)
    if mask.sum() == 0: return np.nan
    return np.mean(np.abs((pred[mask] - true[mask]) / true[mask])) * 100

def initialization(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    log_dir = os.path.join(f'runs/{args.data.replace("/","_")}', f'exp_{timestamp}')
    Path(log_dir).mkdir(exist_ok=True, parents=True)
    args.log_dir = log_dir
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(level=logging.INFO, format='',
                        filename=os.path.join(log_dir, "log_training.txt"),
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    args.log_plotting = os.path.join(log_dir, "log_plotting.txt")
    logging.info(f"{' Hyper-parameters ':=^80}")
    for arg in vars(args):
        logging.info(f'{arg}={getattr(args, arg)}')
    return logging

def prepare_dataloaders(args):
    data_dir = args.data if isinstance(args.data, str) else args.data[0]
    suffix = f'_{args.t_history}_{args.t_pred}_{args.keep_days}_{args.percent}_{args.num_nodes}'
    expected_files = {
        'train_samples': os.path.join(data_dir, f'train_samples{suffix}.npy'),
        'train_targets': os.path.join(data_dir, f'train_targets{suffix}.npy'),
        'train_graph': os.path.join(data_dir, f'train_graph{suffix}.npy'),
        'val_samples': os.path.join(data_dir, f'val_samples{suffix}.npy'),
        'val_targets': os.path.join(data_dir, f'val_targets{suffix}.npy'),
        'val_graph': os.path.join(data_dir, f'val_graph{suffix}.npy'),
        'test_samples': os.path.join(data_dir, f'test_samples{suffix}.npy'),
        'test_targets': os.path.join(data_dir, f'test_targets{suffix}.npy'),
        'test_graph': os.path.join(data_dir, f'test_graph{suffix}.npy'),
    }
    all_exist = all(os.path.exists(p) for p in expected_files.values())
    if all_exist:
        print("Preprocessed data found, loading from cache...")
        train_samples_path = expected_files['train_samples']
        train_targets_path = expected_files['train_targets']
        train_graph_path = expected_files['train_graph']
        val_samples_path = expected_files['val_samples']
        val_targets_path = expected_files['val_targets']
        val_graph_path = expected_files['val_graph']
        test_samples_path = expected_files['test_samples']
        test_targets_path = expected_files['test_targets']
        test_graph_path = expected_files['test_graph']
    else:
        print("No cache found, preprocessing...")
        (train_samples_path, train_targets_path, train_graph_path,
         val_samples_path, val_targets_path, val_graph_path,
         test_samples_path, test_targets_path, test_graph_path) = preprocess_dataset(
            args.data, t_in=args.t_history, t_out=args.t_pred,
            num_nodes=args.num_nodes, keep_days=args.keep_days,
            percent=args.percent, interval=args.interval,
            train=args.train, val=args.val, test=args.test,
            debug=args.debug, type=args.type,
            parallel=False, skip_graph=True, dtw_radius=args.dtw_radius,
            max_nodes=args.max_nodes)
    loader_kwargs = {
        'batch_size': args.batch_size,
        'num_workers': 0,
        'pin_memory': True,
    }
    if args.num_workers > 0:
        loader_kwargs['prefetch_factor'] = 4
        loader_kwargs['persistent_workers'] = True
    train_set = CrossDataset(train_samples_path, train_targets_path, train_graph_path, num_nodes=args.num_nodes)
    train_dataloader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_kwargs)
    val_set = CrossDataset(val_samples_path, val_targets_path, val_graph_path, num_nodes=args.num_nodes)
    val_dataloader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_kwargs)
    test_set = CrossDataset(test_samples_path, test_targets_path, test_graph_path, num_nodes=args.num_nodes)
    test_dataloader = DataLoader(test_set, shuffle=False, drop_last=False, **loader_kwargs)
    return train_dataloader, val_dataloader, test_dataloader

# ===================== EWC 实现（不变） =====================
class EWC:
    def __init__(self, model, dataloader, device, lambda_=0.01, fisher_batches=5):
        self.model = model
        self.lambda_ = lambda_
        self.device = device
        self.old_params = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}
        self.fisher = self._compute_fisher(dataloader, fisher_batches)

    def _compute_fisher(self, dataloader, max_batches):
        fisher = {n: torch.zeros_like(p) for n, p in self.model.named_parameters() if p.requires_grad}
        self.model.train()
        processed = 0
        for data, target, graph in dataloader:
            if processed >= max_batches:
                break
            data = data.to(self.device)
            target = target.to(self.device)
            graph = graph.to(self.device)
            self.model.zero_grad()
            output = self.model(data, adj=graph)
            loss = F.l1_loss(output, target)
            loss.backward()
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.data ** 2
            processed += 1
        for n in fisher:
            fisher[n] /= processed
        return fisher

    def penalty(self):
        loss = 0.0
        for n, p in self.model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.old_params[n]) ** 2).sum()
        return self.lambda_ * loss

# ===================== 训练函数（不变） =====================
def train_batch(model, x, y, g, optimizer, criterions, ewc=None):
    x = x.to(device, dtype=torch.float)
    y = y.to(device, dtype=torch.float)
    g = g.to(device, dtype=torch.float)
    optimizer.zero_grad()
    output = model(x, adj=g)
    base_loss = criterions[0](output, y)
    loss = base_loss
    if ewc is not None:
        loss += ewc.penalty()
    loss.backward()
    optimizer.step()
    return loss, output

def train(args, logging, train_dataloader, val_dataloader, memory_prototypes=None, ewc_dataloader=None):
    num_heads = [int(i) for i in args.num_heads.split(',')]
    channels = [int(i) for i in args.channels.split(',')]
    model = ASTCL(channels, num_heads, args.depth, args.partial, args.num_features,
                  args.t_history, args.t_pred, node_num=args.num_nodes, dropout=args.dropout,
                  target_node=args.target_node, use_dynamic_graph=args.use_dynamic_graph,
                  memory_prototypes=memory_prototypes, use_memory_enhance=args.use_memory_enhance,
                  graph_update_freq=args.graph_update_freq)
    model = model.to(device)
    if args.warmstart:
        checkpoint = torch.load(args.warmstart, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total trainable params: {total_params}")
    loss_criterions = [nn.L1Loss()]
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = None
    training_losses, validation_losses = [], []
    validation_metrics = {'MAEs': [], 'RMSEs': [], 'MAPEs': []}
    best_val_loss = float('inf')
    best_state_dict = None
    best_epoch = 0
    patience = args.patience

    ewc = None
    if args.use_ewc and ewc_dataloader is not None:
        logging.info("Computing Fisher information for EWC...")
        ewc = EWC(model, ewc_dataloader, device, lambda_=args.ewc_lambda, fisher_batches=args.ewc_fisher_batches)
        logging.info("EWC ready.")

    for epoch in range(args.epochs):
        t_start = time.time()
        logging.info(f"------------- Epoch: {epoch:03d} -----------")
        model.train()
        batches_train_loss = []
        loop = tqdm(train_dataloader, desc=f'Train {epoch+1}/{args.epochs}', mininterval=0.5)
        for data, target, graph in loop:
            loss, _ = train_batch(model, data, target, graph, optimizer, loss_criterions, ewc)
            batches_train_loss.append(loss.item())
            loop.set_postfix(loss=np.mean(batches_train_loss[-50:]))
        epoch_train_loss = np.mean(batches_train_loss)
        training_losses.append(epoch_train_loss)
        scheduler.step()
        model.eval()
        batches_val_loss, batches_val_mae, batches_val_rmse, batches_val_mape = [], [], [], []
        with torch.no_grad():
            loop = tqdm(val_dataloader, desc=f'Val {epoch+1}/{args.epochs}', mininterval=0.5)
            for data, target, graph in loop:
                x = data.to(device, dtype=torch.float)
                y = target.to(device, dtype=torch.float)
                g = graph.to(device, dtype=torch.float)
                out = model(x, adj=g)
                val_loss = loss_criterions[0](out, y).item()
                batches_val_loss.append(val_loss)
                out_np = out.detach().cpu().numpy().flatten()
                target_np = y.detach().cpu().numpy().flatten()
                mae = masked_MAE(out_np, target_np)
                rmse = masked_RMSE(out_np, target_np)
                mape = masked_MAPE(out_np, target_np)
                if not np.isnan(mae):
                    batches_val_mae.append(mae)
                    batches_val_rmse.append(rmse)
                    batches_val_mape.append(mape)
                loop.set_postfix(MAE=np.mean(batches_val_mae[-10:]))
        epoch_val_loss = np.mean(batches_val_loss)
        validation_losses.append(epoch_val_loss)
        validation_metrics['MAEs'].append(np.mean(batches_val_mae))
        validation_metrics['RMSEs'].append(np.mean(batches_val_rmse))
        validation_metrics['MAPEs'].append(np.mean(batches_val_mape))
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            best_epoch = epoch
            best_state_dict = copy.deepcopy(model.state_dict())
            torch.save({'epoch': epoch, 'model_state_dict': best_state_dict},
                       os.path.join(args.log_dir, 'best_checkpoint.pt'))
        t_epoch = time.time() - t_start
        logging.info(f"Train loss: {epoch_train_loss:.6f} | Val loss: {epoch_val_loss:.6f}")
        logging.info(f"Val MAE: {validation_metrics['MAEs'][-1]:.4f}, RMSE: {validation_metrics['RMSEs'][-1]:.4f}, MAPE: {validation_metrics['MAPEs'][-1]:.4f}")
        if epoch - best_epoch > patience:
            logging.info(f"Early stop at epoch {epoch}")
            break
    if best_state_dict:
        model.load_state_dict(best_state_dict)
    save_training_curves(training_losses, validation_losses, validation_metrics, args.log_dir)
    return model

def save_training_curves(train_losses, val_losses, metrics, log_dir):
    results_dir = os.path.join(log_dir, 'results')
    Path(results_dir).mkdir(exist_ok=True, parents=True)
    df = pd.DataFrame({
        'epoch': range(len(train_losses)),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'val_mae': metrics['MAEs'],
        'val_rmse': metrics['RMSEs'],
        'val_mape': metrics['MAPEs']
    })
    df.to_csv(os.path.join(results_dir, 'training_curves.csv'), index=False)
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.legend()
    plt.subplot(1,3,2)
    plt.plot(metrics['MAEs'], label='MAE')
    plt.plot(metrics['RMSEs'], label='RMSE')
    plt.legend()
    plt.subplot(1,3,3)
    plt.plot(metrics['MAPEs'], label='MAPE')
    plt.legend()
    plt.savefig(os.path.join(results_dir, 'training_curves.png'), dpi=300)
    plt.close()

# ===================== 校准器训练（不变） =====================
def train_calibrator(args, backbone, train_dataloader, val_dataloader):
    for param in backbone.parameters():
        param.requires_grad = False
    backbone.eval()
    semantic_dim = 7
    calibrator = nn.Sequential(
        nn.Linear(1 + semantic_dim, 32),
        nn.ReLU(),
        nn.Linear(32, args.t_pred)
    ).to(device)
    optimizer = torch.optim.Adam(calibrator.parameters(), lr=1e-3)
    criterion = nn.L1Loss()
    best_val_loss = float('inf')
    best_state = None
    for epoch in range(args.calibrator_epochs):
        calibrator.train()
        train_losses = []
        loop = tqdm(train_dataloader, desc=f'Calibrator Epoch {epoch+1}', mininterval=0.5)
        for data, target, graph in loop:
            data = data.to(device, dtype=torch.float)
            target = target.to(device, dtype=torch.float)
            graph = graph.to(device, dtype=torch.float)
            with torch.no_grad():
                base_out = backbone(data, adj=graph)
                error = base_out - target
            semantic = data[:, 0, -1, -semantic_dim:]
            calib_input = torch.cat([error.mean(dim=1, keepdim=True), semantic], dim=-1)
            correction = calibrator(calib_input)
            final_out = base_out + correction
            loss = criterion(final_out, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            loop.set_postfix(loss=np.mean(train_losses[-50:]))
        calibrator.eval()
        val_losses = []
        with torch.no_grad():
            for data, target, graph in val_dataloader:
                data = data.to(device)
                target = target.to(device)
                graph = graph.to(device)
                base_out = backbone(data, adj=graph)
                error = base_out - target
                semantic = data[:, 0, -1, -semantic_dim:]
                calib_input = torch.cat([error.mean(dim=1, keepdim=True), semantic], dim=-1)
                correction = calibrator(calib_input)
                final_out = base_out + correction
                loss = criterion(final_out, target)
                val_losses.append(loss.item())
        mean_val_loss = np.mean(val_losses)
        print(f"Calibrator Epoch {epoch+1}: Train Loss {np.mean(train_losses):.6f}, Val Loss {mean_val_loss:.6f}")
        if mean_val_loss < best_val_loss:
            best_val_loss = mean_val_loss
            best_state = copy.deepcopy(calibrator.state_dict())
    if best_state is not None:
        calibrator.load_state_dict(best_state)
    return calibrator

# ===================== 统计模型参数量与耗时（修改：增加 device 参数并保存 CSV） =====================
def measure_model_stats(args, model, train_loader, val_loader, device):
    """统计参数量、训练时间、推理时间、校准器更新时间、微调时间"""
    # 1. 参数量
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params/1000:.1f}K")

    # 2. 单轮训练耗时（取前5个batch平均）
    model.train()
    start = time.time()
    for i, (data, target, graph) in enumerate(train_loader):
        if i >= 5: break
        data, target, graph = data.to(device), target.to(device), graph.to(device)
        output = model(data, adj=graph)
        loss = nn.L1Loss()(output, target)
        loss.backward()
    avg_train_time = (time.time() - start) / 5
    print(f"Single batch training time: {avg_train_time*1000:.1f} ms")

    # 3. 单轮推理耗时（取100个batch平均）
    model.eval()
    start = time.time()
    with torch.no_grad():
        for i, (data, target, graph) in enumerate(val_loader):
            if i >= 100: break
            data, target, graph = data.to(device), target.to(device), graph.to(device)
            _ = model(data, adj=graph)
    avg_infer_time = (time.time() - start) / 100
    print(f"Single batch inference time: {avg_infer_time*1000:.1f} ms")

    # 4. 校准器单次更新耗时（模拟）
    calibrator = nn.Linear(10, 1).to(device)
    start = time.time()
    for _ in range(10):
        dummy = torch.randn(64, 10).to(device)
        loss = calibrator(dummy).mean()
        loss.backward()
    calib_update_time = (time.time() - start) / 10
    print(f"Calibrator update time: {calib_update_time*1000:.1f} ms")

    # 5. 周期微调耗时（这里取一个估计值，实际可从训练日志获取，此处设为常量，亦可从args中读取）
    # 为演示，我们设定为30秒（实际应动态计算）
    fine_tune_epoch_time = 30.0  # 秒，可根据实际训练平均耗时替换

    # 保存到CSV
    results_dir = os.path.join(args.log_dir, 'results')
    Path(results_dir).mkdir(exist_ok=True, parents=True)
    stats_df = pd.DataFrame({
        'metric': ['Params (K)', 'Train time (ms/batch)', 'Inference time (ms/batch)',
                   'Calibrator update (ms)', 'Fine-tune epoch (s)'],
        'value': [total_params/1000, avg_train_time*1000, avg_infer_time*1000,
                  calib_update_time*1000, fine_tune_epoch_time]
    })
    stats_df.to_csv(os.path.join(results_dir, 'params_time.csv'), index=False)
    print(f"Model statistics saved to {results_dir}/params_time.csv")

# ===================== 测试函数（不变） =====================
def test_full(args, model, calibrator, test_dataloader):
    model.eval()
    if calibrator is not None:
        calibrator.eval()
    all_preds, all_targets = [], []
    if calibrator is not None:
        all_base_preds = []
    semantic_dim = 7
    start = time.time()
    with torch.no_grad():
        for data, target, graph in tqdm(test_dataloader, desc='Testing', mininterval=0.5):
            data = data.to(device)
            target = target.to(device)
            graph = graph.to(device)
            base_out = model(data, adj=graph)
            if calibrator is not None:
                all_base_preds.append(base_out.cpu().numpy())
                error = base_out - target
                semantic = data[:, 0, -1, -semantic_dim:]
                calib_input = torch.cat([error.mean(dim=1, keepdim=True), semantic], dim=-1)
                correction = calibrator(calib_input)
                final_out = base_out + correction
            else:
                final_out = base_out
            all_preds.append(final_out.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    preds = np.vstack(all_preds)
    targets = np.vstack(all_targets)
    t_pred = preds.shape[1]

    # ----- 计算各步长误差 -----
    step_metrics = {'step': list(range(1, t_pred + 1)), 'MAE': [], 'RMSE': [], 'MAPE': []}
    for step in range(t_pred):
        p = preds[:, step]
        t = targets[:, step]
        step_metrics['MAE'].append(masked_MAE(p, t))
        step_metrics['RMSE'].append(masked_RMSE(p, t))
        step_metrics['MAPE'].append(masked_MAPE(p, t))

    # ----- 诊断输出：平均预测与真实值 -----
    print("\n===== Diagnostic: Average Prediction and Truth per Step =====")
    for step in range(t_pred):
        p_mean = np.mean(preds[:, step])
        t_mean = np.mean(targets[:, step])
        print(
            f"Step {step + 1:2d}: pred_mean = {p_mean:8.4f}, true_mean = {t_mean:8.4f}, diff = {p_mean - t_mean:8.4f}")
    print("=" * 60)

    overall_mae = np.mean(step_metrics['MAE'])
    overall_rmse = np.mean(step_metrics['RMSE'])
    overall_mape = np.mean(step_metrics['MAPE'])
    print(f"\nOverall Test MAE: {overall_mae:.4f}, RMSE: {overall_rmse:.4f}, MAPE: {overall_mape:.4f}")
    print(f"Inference time: {time.time() - start:.2f}s")
    results_dir = os.path.join(args.log_dir, 'results')
    Path(results_dir).mkdir(exist_ok=True, parents=True)
    df = pd.DataFrame(step_metrics)
    df.to_csv(os.path.join(results_dir, 'test_step_metrics.csv'), index=False)
    plt.figure(figsize=(10,6))
    plt.plot(step_metrics['step'], step_metrics['MAE'], marker='o', label='MAE')
    plt.plot(step_metrics['step'], step_metrics['RMSE'], marker='s', label='RMSE')
    plt.plot(step_metrics['step'], step_metrics['MAPE'], marker='^', label='MAPE (%)')
    plt.xlabel('Prediction Step')
    plt.ylabel('Error')
    plt.title('Step-wise Test Metrics')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'test_step_metrics.png'), dpi=300)
    plt.close()
    flat_pred = preds.flatten()
    flat_true = targets.flatten()
    steps = np.tile(np.arange(1, t_pred+1), len(preds))
    plt.figure(figsize=(6,5))
    scatter = plt.scatter(flat_true, flat_pred, c=steps, cmap='viridis', s=5, alpha=0.6)
    plt.plot([flat_true.min(), flat_true.max()], [flat_true.min(), flat_true.max()], 'k--')
    plt.xlabel('True')
    plt.ylabel('Pred')
    plt.colorbar(scatter, label='Step')
    plt.title('Prediction vs True')
    plt.savefig(os.path.join(results_dir, 'scatter_plot.png'), dpi=300)
    plt.close()
    if calibrator is not None and 'all_base_preds' in locals():
        base_preds = np.vstack(all_base_preds)
        errors_base = np.abs(base_preds - targets)
        errors_calib = np.abs(preds - targets)
        fig, ax = plt.subplots(figsize=(6,4))
        steps_mean = np.arange(1, t_pred+1)
        mean_base = errors_base.mean(axis=0)
        mean_calib = errors_calib.mean(axis=0)
        ax.plot(steps_mean, mean_base, 'o-', label='Without Calibrator')
        ax.plot(steps_mean, mean_calib, 's-', label='With Calibrator')
        ax.set_xlabel('Prediction Step')
        ax.set_ylabel('Mean Absolute Error')
        ax.set_title('Effect of Real-time Calibrator')
        ax.legend()
        ax.grid(True)
        plt.savefig(os.path.join(results_dir, 'calibrator_effect.png'), dpi=300)
        plt.close()
    print(f"All results saved to {results_dir}")

# ===================== 新颖度分析（修改：移除多余参数） =====================
def analyze_novelty(args, model, test_loader):
    """
    计算测试集每个样本的新颖度，并按时段分类，生成箱线图和触发时序图
    """
    model.eval()
    # 获取记忆原型 (已存于model.graph_constructor.memory_prototypes)
    if not hasattr(model, 'graph_constructor') or model.graph_constructor is None:
        print("Warning: No graph constructor found, skip novelty analysis.")
        return
    prototypes = model.graph_constructor.memory_prototypes.detach().cpu().numpy()  # (num_protos, feat_dim)

    novelty_list = []          # 每个样本的新颖度（标量）
    is_weekend_list = []       # 0/1
    is_congestion_list = []    # 0/1
    trigger_list = []          # 是否触发更新 (0/1)
    sample_idx_list = []       # 样本序号
    flow_values = []           # 用于拥堵判断的实际流量（只取目标节点）

    # 收集所有样本的流量（用于定义拥堵阈值）
    all_flows = []
    with torch.no_grad():
        for idx, (data, target, graph) in enumerate(test_loader):
            data = data.to(device)
            flow = data[:, args.target_node, -1, 0].cpu().numpy()  # (batch,)
            all_flows.extend(flow)
    # 定义拥堵阈值：历史均值+2倍标准差
    flow_mean = np.mean(all_flows)
    flow_std = np.std(all_flows)
    congestion_threshold = flow_mean + 2 * flow_std

    # 再次遍历测试集计算新颖度
    with torch.no_grad():
        for idx, (data, target, graph) in enumerate(test_loader):
            data = data.to(device)
            b, n, t, f = data.shape  # (batch, num_nodes, seq_len, num_features)
            # 取每个节点的时间平均特征
            h = data.mean(dim=2)  # (b, n, f)
            # 投影到原型维度（若存在）
            if hasattr(model.graph_constructor, 'input_proj') and model.graph_constructor.input_proj is not None:
                h = model.graph_constructor.input_proj(h)
            # 计算到记忆原型的距离
            proto = model.graph_constructor.memory_prototypes  # (num_protos, proto_dim)
            # 确保维度匹配
            assert h.shape[-1] == proto.shape[-1], f"h dim {h.shape[-1]} != proto dim {proto.shape[-1]}"
            dist = torch.cdist(h, proto.unsqueeze(0).expand(b, -1, -1), p=2)  # (b, n, num_protos)
            min_dist, _ = torch.min(dist, dim=-1)  # (b, n)
            sample_novelty = min_dist.mean(dim=1).cpu().numpy()  # (b,)

            # 提取时段信息和流量（data为(b, n, t, f)）
            semantic = data[:, 0, -1, -7:].cpu().numpy()  # (b, 7)
            weekday = semantic[:, 2]
            is_weekend = (weekday >= 5).astype(int)
            flow = data[:, args.target_node, -1, 0].cpu().numpy()
            is_congestion = (flow > congestion_threshold).astype(int)

            novelty_list.extend(sample_novelty.tolist())
            is_weekend_list.extend(is_weekend.tolist())
            is_congestion_list.extend(is_congestion.tolist())
            sample_idx_list.extend(range(idx * b, idx * b + b))
            flow_values.extend(flow.tolist())

    # 计算75%分位数作为阈值
    novelty_array = np.array(novelty_list)
    if len(novelty_array) == 0:
        print("No novelty values computed.")
        return
    threshold = np.percentile(novelty_array, 75)
    trigger_list = (novelty_array > threshold).astype(int).tolist()

    # 构建DataFrame
    df = pd.DataFrame({
        'sample_idx': sample_idx_list,
        'novelty': novelty_list,
        'is_weekend': is_weekend_list,
        'is_congestion': is_congestion_list,
        'flow': flow_values,
        'trigger_update': trigger_list
    })
    results_dir = os.path.join(args.log_dir, 'results')
    Path(results_dir).mkdir(exist_ok=True, parents=True)
    df.to_csv(os.path.join(results_dir, 'novelty_analysis.csv'), index=False)

    # 绘制箱线图
    plt.figure(figsize=(10, 6))
    df['category'] = 'Normal'
    df.loc[df['is_weekend'] == 1, 'category'] = 'Weekend'
    df.loc[df['is_congestion'] == 1, 'category'] = 'Congestion'
    sns.boxplot(x='category', y='novelty', data=df, palette='Set2')
    plt.ylabel('Novelty Score')
    plt.title('Novelty Distribution across Different Periods')
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'novelty_boxplot.png'), dpi=300)
    plt.close()

    # 绘制触发更新的时序图（前500个样本）
    plt.figure(figsize=(12, 4))
    subset = df.iloc[:500]
    plt.scatter(subset['sample_idx'], subset['novelty'], c=subset['trigger_update'],
                cmap='Reds', s=10, alpha=0.7, edgecolors='k')
    plt.axhline(y=threshold, color='blue', linestyle='--', label=f'Threshold ({threshold:.3f})')
    plt.xlabel('Sample Index (time order)')
    plt.ylabel('Novelty Score')
    plt.title('Novelty and Memory Update Triggers (First 500 test samples)')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'novelty_trigger_timeline.png'), dpi=300)
    plt.close()

    print(f"Novelty analysis saved to {results_dir}")

# ===================== 记忆库构建（不变） =====================
def build_memory_prototypes(model, dataset, device, batch_size, num_prototypes=50, sample_ratio=0.2):
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    all_features = []
    with torch.no_grad():
        for data, target, graph in tqdm(loader, desc="Extracting features for memory"):
            feat = data[:, :, -1, :]
            feat = feat.reshape(-1, feat.shape[-1])
            all_features.append(feat.cpu().numpy())
    all_features = np.vstack(all_features)
    if sample_ratio < 1.0:
        idx = np.random.choice(len(all_features), int(len(all_features) * sample_ratio), replace=False)
        all_features = all_features[idx]
    scaler = StandardScaler()
    all_features = scaler.fit_transform(all_features)
    kmeans = KMeans(n_clusters=num_prototypes, random_state=42, n_init=10)
    kmeans.fit(all_features)
    prototypes = kmeans.cluster_centers_
    prototypes = torch.from_numpy(prototypes).float().to(device)
    return prototypes, scaler

# ===================== 主函数 =====================
def main(args):
    logging = initialization(args)
    train_loader, val_loader, test_loader = prepare_dataloaders(args)

    memory_prototypes = None
    if args.use_memory_enhance:
        print("\n=== Building memory prototypes from training data ===\n")
        num_heads = [int(i) for i in args.num_heads.split(',')]
        channels = [int(i) for i in args.channels.split(',')]
        temp_model = ASTCL(channels, num_heads, args.depth, args.partial, args.num_features,
                           args.t_history, args.t_pred, node_num=args.num_nodes, dropout=args.dropout,
                           target_node=args.target_node, use_dynamic_graph=False).to(device)
        memory_prototypes, _ = build_memory_prototypes(temp_model, train_loader.dataset, device,
                                                       batch_size=args.batch_size,
                                                       num_prototypes=args.memory_prototypes)
        del temp_model
        torch.cuda.empty_cache()

    backbone = train(args, logging, train_loader, val_loader, memory_prototypes=memory_prototypes)

    # ---- 测量统计信息（表5） ----
    # 注意：如果使用校准器，测量的是 backbone_model（即带校准器但冻结骨干的模型），但通常我们关心骨干模型本身的参数和推理速度。
    # 这里我们在两种情况下都测量 backbone 本身（未冻结）以获得最纯粹的统计量
    measure_model_stats(args, backbone, train_loader, val_loader, device)

    if args.use_calibrator:
        print("\n=== Training calibrator on frozen backbone ===\n")
        num_heads = [int(i) for i in args.num_heads.split(',')]
        channels = [int(i) for i in args.channels.split(',')]
        backbone_model = ASTCL(channels, num_heads, args.depth, args.partial, args.num_features,
                               args.t_history, args.t_pred, node_num=args.num_nodes, dropout=args.dropout,
                               target_node=args.target_node, use_dynamic_graph=args.use_dynamic_graph,
                               memory_prototypes=memory_prototypes, use_memory_enhance=args.use_memory_enhance,
                               graph_update_freq=args.graph_update_freq).to(device)
        ckpt_path = os.path.join(args.log_dir, 'best_checkpoint.pt')
        backbone_model.load_state_dict(torch.load(ckpt_path, map_location=device)['model_state_dict'])
        calibrator = train_calibrator(args, backbone_model, train_loader, val_loader)
        test_full(args, backbone_model, calibrator, test_loader)
        torch.save(calibrator.state_dict(), os.path.join(args.log_dir, 'best_calibrator.pt'))
    else:
        test_full(args, backbone, None, test_loader)

    # ---- 新颖度分析（仅当使用记忆增强时） ----
    if args.use_memory_enhance:
        analyze_novelty(args, backbone, test_loader)

# ===================== 参数配置 =====================
if __name__ == '__main__':
    from types import SimpleNamespace

    # 选择要运行的消融变体 (只设置一个为 True)
    run_base = False          # 静态图，无任何增强
    run_dyn = False           # 记忆增强 + EWC
    run_mem = False           # 校准器 + EWC
    run_mem_calib = False     # 记忆增强动态图 + 实时校准器
    run_full = True           # 记忆增强 + 校准器 + EWC（完整版）

    common_args = {
        'seed': 7,
        'data': 'data/PEMSBAY',
        'max_nodes': 100,
        'keep_days': 7.0,
        'percent': 1.0,
        'interval': 288,
        'type': 'flow',
        'num_nodes': 100,
        'num_features': 8,
        'num_heads': '1,1,2,2,2',
        'partial': 4,
        'channels': '8,8,8,8,8,8',
        'depth': 5,
        't_history': 12,
        't_pred': 12,
        'target_node': 0,
        'epochs': 100,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'batch_size': 64,
        'dropout': 0.2,
        'train': True,
        'val': True,
        'test': True,
        'warmstart': '',
        'dtw_radius': 5,
        'patience': 10,
        'calibrator_epochs': 15,
        'num_workers': 0,
        'graph_update_freq': 50,
        'memory_prototypes': 50,
        'use_ewc': False,
        'ewc_lambda': 0.01,
        'ewc_fisher_batches': 5,
        'debug': False,
        'target_nodes': 'all',
        'parallel': False,
        'skip_graph': True,
    }

    if run_base:
        config = {
            'use_dynamic_graph': False,
            'use_memory_enhance': False,
            'use_calibrator': False,
        }
    elif run_dyn:
        config = {
            'use_dynamic_graph': True,
            'use_memory_enhance': True,
            'use_calibrator': False,
            'use_ewc': True,
        }
    elif run_mem:
        config = {
            'use_dynamic_graph': False,
            'use_memory_enhance': True,
            'use_calibrator': True,
            'use_ewc': True,
        }
    elif run_mem_calib:
        config = {
            'use_dynamic_graph': True,
            'use_memory_enhance': True,
            'use_calibrator': True,
        }
    elif run_full:
        config = {
            'use_dynamic_graph': True,
            'use_memory_enhance': True,
            'use_calibrator': False,
            'use_ewc': True,
        }
    else:
        raise ValueError("Please set one of the run_* variables to True")

    args_dict = {**common_args, **config}
    args = SimpleNamespace(**args_dict)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    main(args)