import pickle
import torch
import torch.nn as nn
import networkx as nx
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv
from collections import defaultdict
import torch.nn.functional as F
import json
import os

def load_processed_data(filename):
    print(f"正在从 {filename} 加载处理后的数据...")
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    print("数据加载完成!")
    return data

def create_global_mappings(processed_data):
    node_to_id = defaultdict(lambda: len(node_to_id))
    edge_type_to_id = defaultdict(lambda: len(edge_type_to_id))
    node_types = set()
    edge_types = set()
    
    for patient_info in processed_data.values():
        graph = patient_info['时序异构图']
        for node, attr in graph.nodes(data=True):
            node_type = attr['type']
            node_types.add(node_type)
            node_to_id[f"{node_type}_{node}"]
        
        for _, _, attr in graph.edges(data=True):
            edge_type = attr['type']
            edge_type_str = str(edge_type)
            edge_types.add(edge_type_str)
            edge_type_to_id[edge_type_str]
    return dict(node_to_id), dict(edge_type_to_id), list(node_types), list(edge_types)

def save_node_to_index(node_to_index, save_path):
    """保存node_to_index字典到JSON文件"""
    with open(save_path, 'w') as f:
        json.dump(node_to_index, f)

def load_node_to_index(save_path):
    """从JSON文件加载node_to_index字典，确保键的类型正确"""
    with open(save_path, 'r') as f:
        return json.load(f)

def convert_to_hetero_data(graph, node_to_id, edge_type_to_id, node_types, edge_types, save_dir, device):
    data = HeteroData()
    
    # 创建双向映射
    local_node_to_index = {}
    local_index_to_node = {}
    local_index_to_type = {}
    current_idx = 0
    
    # 为每种节点类型记录原始ID和类型ID
    for node_type in set(nx.get_node_attributes(graph, 'type').values()):
        nodes = [node for node, attr in graph.nodes(data=True) if attr['type'] == node_type]
        if nodes:
            # 创建局部索引映射
            for node in nodes:
                if node not in local_node_to_index:
                    local_node_to_index[node] = current_idx
                    local_index_to_node[current_idx] = node
                    local_index_to_type[current_idx] = node_type
                    current_idx += 1
            
            # 存储原始ID和类型ID，并移动到指定设备
            node_ids = torch.tensor([node_to_id[f"{node_type}_{node}"] for node in nodes], device=device)
            node_type_ids = torch.tensor([node_types.index(node_type)] * len(nodes), device=device)
            
            # 使用 'x' 作为特征键名，存储原始特征
            data[node_type].x = torch.stack([node_ids, node_type_ids], dim=1)  # [num_nodes, 2]
    
    # 添加边关系
    for edge_type in set(nx.get_edge_attributes(graph, 'type').values()):
        edges = [(u, v) for (u, v, attr) in graph.edges(data=True) if attr['type'] == edge_type]
        if edges:
            src, dst = zip(*edges)
            src_type = graph.nodes[src[0]]['type']
            dst_type = graph.nodes[dst[0]]['type']
            
            src_index = [local_node_to_index[sn] for sn in src]
            dst_index = [local_node_to_index[dn] for dn in dst]
            
            edge_type_str = str(edge_type)
            data[src_type, edge_type_str, dst_type].edge_index = torch.tensor(
                [src_index, dst_index], 
                dtype=torch.long,
                device=device
            )
            
            # 确保边特征是2D张量 [num_edges, 1]
            edge_type_ids = torch.tensor(
                [edge_type_to_id[edge_type_str]] * len(edges),
                device=device
            ).view(-1, 1)  
            
            data[src_type, edge_type_str, dst_type].edge_attr = edge_type_ids
    
    return data

def get_node_info(data, node_index):
    """获取给定索引对应的节点信息"""
    if node_index in data.local_index_to_node:
        node = data.local_index_to_node[node_index]
        node_type = data.local_index_to_type[node_index]
        return f"{node_type}_{node}"
    return None

class CustomGATConv(GATConv):
    def __init__(self, in_channels, out_channels, heads=1, edge_dim=None, **kwargs):
        super().__init__(in_channels, out_channels, heads=heads, edge_dim=edge_dim, **kwargs)

    def forward(self, x, edge_index, edge_attr=None):
        # 确保所有输入都在同一设备上
        x = x.to(edge_index.device)
        if edge_attr is not None:
            edge_attr = edge_attr.to(edge_index.device)

        # 获取节点数量
        num_nodes = x.size(0)

        # 检查边索引的有效性
        if edge_index.numel() > 0:  # 如果有边
            assert edge_index.max() < num_nodes, f"Edge index {edge_index.max()} >= number of nodes {num_nodes}"

        # 使用父类的forward方法
        return super().forward(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)

class HeteroGNN(torch.nn.Module):
    def __init__(self, num_nodes, num_node_types, num_edge_types, hidden_channels, out_channels, num_layers, heads, device):
        super().__init__()
        
        self.device = device
        self.heads = heads
        self.hidden_channels = hidden_channels
        
        # 初始化嵌入层
        self.node_embedding = nn.Embedding(num_nodes, hidden_channels).to(device)
        self.node_type_embedding = nn.Embedding(num_node_types, hidden_channels).to(device)
        self.edge_type_embedding = nn.Embedding(num_edge_types, hidden_channels).to(device)
        
        # GAT层
        self.convs = torch.nn.ModuleList()
        self.convs.append(CustomGATConv(
            hidden_channels, 
            hidden_channels // heads, 
            heads=heads, 
            edge_dim=hidden_channels
        ).to(device))
        
        for _ in range(1, num_layers):
            self.convs.append(CustomGATConv(
                hidden_channels, 
                hidden_channels // heads, 
                heads=heads, 
                edge_dim=hidden_channels
            ).to(device))
        
        self.lin = nn.Linear(hidden_channels*2, out_channels).to(device)
        self.dropout = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.4)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        # 确保所有输入都在正确的设备上
        x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
        edge_index_dict = {k: v.to(self.device) for k, v in edge_index_dict.items()}
        edge_attr_dict = {k: v.to(self.device) for k, v in edge_attr_dict.items()}
        
        # 初始化节点表征
        transformed_x = {}
        for node_type, feat in x_dict.items():
            node_ids = feat[:, 0].long()
            type_ids = feat[:, 1].long()
            
            node_emb = self.node_embedding(node_ids)
            type_emb = self.node_type_embedding(type_ids)
            transformed_x[node_type] = self.dropout(self.lin(torch.cat([node_emb, type_emb], dim=1)))
        
        # 初始化边表征，确保所有边特征都是2D的
        transformed_edge_attr = {}
        for edge_key, edge_attr in edge_attr_dict.items():
            # 获取边类型嵌入并确保是2D张量 [num_edges, hidden_size]
            edge_type_emb = self.edge_type_embedding(edge_attr.squeeze())
            if edge_type_emb.dim() == 1:
                edge_type_emb = edge_type_emb.unsqueeze(0)
            transformed_edge_attr[edge_key] = self.dropout(edge_type_emb)
            
        
        # 合并所有节点的特征
        x = torch.cat([x for x in transformed_x.values()])
        edge_index = torch.cat([edge_index for edge_index in edge_index_dict.values()], dim=1)
        # x = self.dropout(x)
        
        # 确保所有边特征维度一致后再拼接
        edge_attr = torch.cat([attr for attr in transformed_edge_attr.values()], dim=0)
        
        # 存储每一层的边权重
        
        # 应用图卷积层
        for conv in self.convs:
            x_new, attention = conv(x, edge_index, edge_attr=edge_attr)
            # attention 是一个元组 (edge_index, alpha)，我们只需要 alpha
            e_index, alpha = attention
            x = F.elu(x_new)
            x = self.dropout(x_new)
        
        # 对所有节点的表征进行平均池化
        patient_repr = x.mean(dim=0)
        patient_repr = self.dropout2(patient_repr)
        final_edge_weights = alpha
        edge_index = e_index
        assert edge_index.size(1) == final_edge_weights.size(0), \
                f"Edge indices shape {edge_index.size(1)} doesn't match weights shape {final_edge_weights.size(0)}"
            
        
        return patient_repr, edge_index, final_edge_weights

# 在创建模型时，修改参数传递
def create_model(args):
    model = HeteroGNN(
        num_nodes=args.num_nodes,
        num_node_types=args.num_node_types,
        num_edge_types=args.num_edge_types,
        hidden_channels=args.hidden_size,
        out_channels=args.out_channels,
        num_layers=args.num_layers,
        heads=args.num_attention_heads,
        device=args.device
    ).to(args.device)
    return model

processed_data = load_processed_data('./data/mimic/processed_patient_data_with_snomed.pkl')
node_to_id, edge_to_id, node_types, edge_types = create_global_mappings(processed_data)

def get_seqG_embedding(pid, seqG_model):
    device = next(seqG_model.parameters()).device  # 获取模型所在的设备
    patient_graph = processed_data[pid]['时序异构图']
    hetero_data = convert_to_hetero_data(
        patient_graph, 
        node_to_id, 
        edge_to_id, 
        node_types, 
        edge_types, 
        './data/mimic/',
        device
    )
    
    patient_repr, edge_index, edge_weight = seqG_model(
        hetero_data.x_dict, 
        hetero_data.edge_index_dict, 
        hetero_data.edge_attr_dict
    )
    
    return patient_repr, edge_index, edge_weight

def check_node_ids(graph):
    """检查图中的节点ID格式"""
    for node, attr in graph.nodes(data=True):
        print(f"Node: {node}, Type: {attr['type']}")
        if isinstance(node, str) and node.startswith('V_'):
            print(f"Visit node found: {node}")
