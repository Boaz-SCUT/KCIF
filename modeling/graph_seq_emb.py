import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, GATConv
import torch_scatter
import numpy as np
import json

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# bert_model = BertModel.from_pretrained('./data/best_bert_model').to(device)
# tokenizer = BertTokenizer.from_pretrained('./data/best_bert_model')

# mimic iii中的icd9代码
# icd2id = np.load('./data/icd2idx.npy', allow_pickle=True).item()
# id2icd = {v: k for k, v in icd2id.items()}
# mimic iv
icd2id = np.load('./data/icd2idx.npy', allow_pickle=True).item()
id2icd = {v: k for k, v in icd2id.items()}

# 从文件中加载ICD代码的嵌入表示
with open("./data/code_embeddings.json") as f:
    code_embeddings = json.load(f)

class GraphEncoder2(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, num_node_types):
        super(GraphEncoder, self).__init__()

        self.embed_edge_type = torch.nn.Embedding(num_relations, in_channels)
        self.embed_node_type = torch.nn.Embedding(num_node_types, in_channels)
        self.fc = torch.nn.Linear(768, 768)
        self.gat1 = GATConv(in_channels, hidden_channels, heads=1)
        self.gat2 = GATConv(hidden_channels, out_channels, heads=1)
        self.drop_x = nn.Dropout(0.5)

    def forward(self, x, edge_index, edge_type, node_type):
        x = torch.randn((1, 768), device="cuda:0")
        x = x.repeat((x.size(0), 1))
        if edge_index.numel() == 0 or x.size(0) == 1:
            return x, None

        edge_embeddings = self.embed_edge_type(edge_type)
        node_embeddings = self.embed_node_type(node_type) + x

        # Add edge embeddings to node embeddings of source nodes
        src_node_embeddings = node_embeddings[edge_index[0, :], :]
        # tgt_node_embeddings = node_embeddings[edge_index[1, :], :]
        x = torch.cat([src_node_embeddings, edge_embeddings], dim=-1)

        # x = torch.cat([src_node_embeddings, tgt_node_embeddings], dim=-1)
        # x = src_node_embeddings - tgt_node_embeddings
        # x = src_node_embeddings
        x = self.fc(x.relu())
        # x = self.fc(x)
        x = self.gat1(x, edge_index)
        x = x.relu()
        x = self.drop_x(x)
        x, att = self.gat2(x, edge_index, return_attention_weights=True)
        x = x.relu()
        x = self.drop_x(x)
        return global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device)), att

class HeteroGATLayer(nn.Module):
    def __init__(self, in_channels, out_channels, edge_type_num):
        super(HeteroGATLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_type_num = edge_type_num
        self.node_type_embedding = torch.nn.Embedding(3, in_channels)
        self.drop_out = nn.Dropout(0.2)
        self.lin = nn.Linear(out_channels*2, out_channels)
        self.fc_single_node = nn.Linear(in_channels, out_channels)

        # 创建多个GATConv层，每个边类型对应一个GATConv层

        self.convs = nn.ModuleList([GATConv(in_channels, out_channels) for _ in range(edge_type_num)])
        self.drop2 = nn.Dropout(0.4)

    def forward(self, x, edge_index, edge_type, node_type):
        if edge_index.numel() == 0:
            if x.size(0) != 1:
                x_emb = x.unsqueeze(0)
                x = self.fc_single_node(x_emb)
            else:
                x = self.fc_single_node(x)
            return x, None
        node_embedding = self.node_type_embedding(node_type)
        x = x + node_embedding
        x = self.drop_out(x)
        outs = []
        attention_weights = []
        # 对于每种边类型，分别应用对应的GATConv层
        # try:
        for i, mask in enumerate((edge_type == etype).nonzero(as_tuple=False).flatten() for etype in range(self.edge_type_num)):
            out, attention_weight = self.convs[i](x, edge_index[:, mask], return_attention_weights=True)
            out = self.drop2(out)

            outs.append(out)
            attention_weights.append(attention_weight)
        # except:
        #     print('edge_index.shape: ', edge_index.shape)
        #     print('edge_type.shape: ', edge_type.shape)
        #     print('node_type.shape: ', node_type.shape)
        #     print('x.shape: ', x.shape)
        #     # 输出报错信息
        #     print('edge_type: ', edge_type)
        return sum(outs), attention_weights

class GraphEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_relations):
        super(GraphEncoder, self).__init__()
        self.drop_out = nn.Dropout(0.5)
        self.conv1 = HeteroGATLayer(in_channels, out_channels, num_relations)
        self.conv2 = HeteroGATLayer(out_channels, out_channels, num_relations)
        # self.conv3 = HeteroGATLayer(out_channels, out_channels, num_relations)

    def forward(self, x, edge_index, edge_type, node_type, max_length=500):

        # x是list，且在cpu上，需要放在GPU上
        # icd_list = [id2icd[x_id] for x_id in x]
        # x_emb_list = [code_embeddings[code] for code in icd_list if code in code_embeddings]
        # x = torch.stack([torch.tensor(embedding) for embedding in x_emb_list], dim=0).to(device)

        # MIMIC IV-- 因为GPU内存不够用，如果x的长度超过200，所以我们只取graph的60%的数据作为输入。
        # if x.size(0) > max_length:
        #     x = x[:int(max_length), :]
        #     mask = (edge_index[0, :] < max_length) & (edge_index[1, :] < max_length)
        #     edge_index = edge_index[:, mask]
        #     edge_type = edge_type[mask]
        #     node_type = node_type[:int(max_length)]

        # x是list，且在cpu上，需要放在GPU上

        # 随机初始化x
        # x = torch.randn((x.size(0), 768), device="cuda:0")
        x = torch.randn((len(x), 128), device="cuda:0")
        x,_ = self.conv1(x, edge_index, edge_type, node_type)
        x = torch.nn.functional.relu(x)
        # x = self.drop_out(x)
        x, att = self.conv2(x, edge_index, edge_type, node_type)
        # x, att = self.conv2(x, edge_index, edge_type, node_type)
        torch.nn.functional.relu(x)
        x = self.drop_out(x)

        return global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device)), att


    def forward0(self, x, edge_index, edge_type, node_type, max_length=500):

        # x是list，且在cpu上，需要放在GPU上
        icd_list = [id2icd[x_id] for x_id in x]
        x_emb_list = [code_embeddings[code] for code in icd_list if code in code_embeddings]
        x = torch.stack([torch.tensor(embedding) for embedding in x_emb_list], dim=0).to(device)

        # MIMIC IV-- 因为GPU内存不够用，如果x的长度超过200，所以我们只取graph的60%的数据作为输入。
        # if x.size(0) > max_length:
        #     x = x[:int(max_length), :]
        #     mask = (edge_index[0, :] < max_length) & (edge_index[1, :] < max_length)
        #     edge_index = edge_index[:, mask]
        #     edge_type = edge_type[mask]
        #     node_type = node_type[:int(max_length)]


        # MIMIC III不用采样

        # x = torch.randn((x.size(0), 768), device="cuda:0")
        x,_ = self.conv1(x, edge_index, edge_type, node_type)
        x = torch.nn.functional.relu(x)
        # x = self.drop_out(x)
        x, att = self.conv2(x, edge_index, edge_type, node_type)
        torch.nn.functional.relu(x)
        x = self.drop_out(x)

        return global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long, device=x.device)), att



class SequenceEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(SequenceEncoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.self_attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=4)

    def forward(self, x):
        x = x.squeeze(2)
        # 设置初始的隐藏状态和单元状态
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))

        # 自注意力操作需要输入尺寸为 (seq_len, batch, embed_dim)，所以我们需要对 out 进行转置操作
        out = out.transpose(0, 1)

        attn_output, _ = self.self_attention(out, out, out)

        # 我们使用最后一个时间步的输出
        out = attn_output[-1, :, :]

        return out

def batch_to_embedding(patient_data, graph_encoder, sequence_encoder):
    '''
    根据患者的seqgraph数据，得到患者的嵌入向量，这里的seqgraph数据是一个患者的数据
    '''
    # max_graph_num = max(len(patient_graphs) for patient_graphs in patient_data)  # 计算一个batch中最多的子图数量
    graph_embeddings = []
    for graph in patient_data:
        graph_embedding, att = graph_encoder(graph.x, graph.edge_index, graph.edge_type)
        graph_embeddings.append(graph_embedding)
    # 在graph_embeddings 的 dim=0维度上添加一个维度
    patient_embedding = sequence_encoder(torch.stack(graph_embeddings).unsqueeze(0))  # 将所有子图的嵌入合并成一个张量


    return patient_embedding

def get_pid2graph_emb(pid):

    pid2graph_data = torch.load('../data/mimic/statement/pid2graph_data.pt')
    graph_datas = pid2graph_data[pid]
    # instantiate your models
    num_node_types = 3  # adjust this according to your data
    num_edge_types = 3  # adjust this according to your data
    in_channels = 32  # adjust this according to your data
    out_channels = 32  # adjust this according to your task
    graph_encoder = GraphEncoder(in_channels, out_channels, num_edge_types)
    sequence_encoder = SequenceEncoder(input_size=out_channels, hidden_size=in_channels, num_layers=1)

    batch_embedding = batch_to_embedding(graph_datas, graph_encoder, sequence_encoder)
    print(batch_embedding)
    print(batch_embedding.shape)

def get_grap2emb(graph_data):
    num_node_types = 3  # adjust this according to your data
    num_edge_types = 3  # adjust this according to your data
    in_channels = 32  # adjust this according to your data
    out_channels = 32  # adjust this according to your task
    graph_encoder = GraphEncoder(in_channels, out_channels, num_edge_types)
    sequence_encoder = SequenceEncoder(input_size=out_channels, hidden_size=in_channels, num_layers=1)

    batch_embedding = batch_to_embedding(graph_data, graph_encoder, sequence_encoder)
    print(batch_embedding)
    print(batch_embedding.shape)

import torch.nn.functional as F
class RelationalGraphConvolution(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_node_types, num_edge_types):
        super(RelationalGraphConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_edge_types = num_edge_types
        self.node_type_embedding = torch.nn.Embedding(num_node_types, in_channels)
        self.edge_type_embedding = torch.nn.Embedding(num_edge_types, in_channels)
        self.W_N = torch.nn.Linear(in_channels*2, out_channels)
        self.W_R = torch.nn.Linear(in_channels, out_channels) # Linear transformation for relation embedding
        self.attention = torch.nn.Linear(2 * out_channels, 1)
        self.drop_out = nn.Dropout(0.7)


    def forward(self, x, edge_index, node_type, edge_type):
        x = torch.randn((1, 768), device="cuda:0")
        x = x.repeat((x.size(0), 1))
        if edge_index.numel() == 0 or x.size(0) == 1:
            return x, None

        node_type_embeddings = self.node_type_embedding(node_type)
        edge_type_embeddings = self.edge_type_embedding(edge_type)

        src_node_embeddings = x[edge_index[0, :], :]
        tgt_node_embeddings = x[edge_index[1, :], :]

        source_node_embedding = x + node_type_embeddings #
        edge_embedding = source_node_embedding - edge_type_embeddings
        x_combined = torch.cat([source_node_embedding, edge_embedding], dim=-1) # phi(X)
        edge_embedding = self.W_R(edge_embedding) # Update relation embedding #
        aggregated_edge_embeddings = torch_scatter.scatter(edge_embedding, edge_index[1, :], reduce="mean")
        x = self.W_N(x_combined) + aggregated_edge_embeddings # 聚合信息
        x = self.drop_out(x)
        attention_weights = self.attention(torch.cat([x[edge_index[0, :]], x[edge_index[1, :]]], dim=1)).squeeze(1)
        attention_weights = F.softmax(attention_weights, dim=0)
        # Get graph representation by averaging node embeddings
        graph_embedding = torch.mean(x, dim=0)
        return graph_embedding, attention_weights




class GATGraphEncoder(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_node_types, num_relations, heads=2):
        super(GATGraphEncoder, self).__init__()

        self.node_type_embedding = torch.nn.Embedding(num_node_types, in_channels)
        self.convs = torch.nn.ModuleList()
        for _ in range(2):  # Add two layers
            self.convs.append(
                torch.nn.ModuleList(
                    [GATConv((2 if _ == 0 else heads) * in_channels, hidden_channels, heads=heads)
                     for _ in range(num_relations)]
                )
            )
        self.conv_out = GATConv(hidden_channels*heads, out_channels, concat=False)

    def forward(self, x, node_type, edge_index, edge_type):
        node_embedding = self.node_type_embedding(node_type)
        x = torch.cat([x, node_embedding], dim=-1)

        for convs in self.convs:
            x = sum(convs[edge_type[i]](x, edge_index[:, mask]) for i, mask in enumerate(edge_type == i))
            x = x.relu()

        x, edge_weights = self.conv_out(x, edge_index)
        return global_mean_pool(x, torch.zeros(x.size(0), dtype=torch.long)), edge_weights




# if __name__ == '__main__':
#     # get_pid2graph_emb(pid = 11)
#     pid2graph_data = torch.load('../data/mimic/statement/pid2graph_data.pt')
#     graph_data = pid2graph_data[95561]
#     get_grap2emb(graph_data)