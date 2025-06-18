import torch

from modeling.modeling_encoder import TextEncoder, MODEL_NAME_TO_CLASS
from utils.data_hita import *
from utils.layers import *
from modeling.transformer_kcif import TransformerTime


# 图神经网络内的消息传递
class QAGNN_Message_Passing(nn.Module):
    def __init__(self, args, k, n_ntype, n_etype, input_size, hidden_size, output_size,
                 dropout=0.1):
        super().__init__()
        assert input_size == output_size
        self.args = args
        self.n_ntype = n_ntype
        self.n_etype = n_etype

        assert input_size == hidden_size
        self.hidden_size = hidden_size

        self.emb_node_type = nn.Linear(self.n_ntype, hidden_size // 2)

        self.basis_f = 'sin'  # ['id', 'linact', 'sin', 'none']
        if self.basis_f in ['id']:
            self.emb_score = nn.Linear(1, hidden_size // 2)
        elif self.basis_f in ['linact']:
            self.B_lin = nn.Linear(1, hidden_size // 2)
            self.emb_score = nn.Linear(hidden_size // 2, hidden_size // 2)
        elif self.basis_f in ['sin']:
            self.emb_score = nn.Linear(hidden_size // 2, hidden_size // 2)

        self.edge_encoder = torch.nn.Sequential(
            torch.nn.Linear(n_etype + 1 + n_ntype * 2, hidden_size),
            torch.nn.BatchNorm1d(hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size))

        self.k = k
        self.gnn_layers = nn.ModuleList(
            [GATConvE(args, hidden_size, n_ntype, n_etype, self.edge_encoder) for _ in range(k)])

        self.Vh = nn.Linear(input_size, output_size)
        self.Vx = nn.Linear(hidden_size, output_size)

        self.activation = GELU()
        self.dropout = nn.Dropout(dropout)
        self.dropout_rate = dropout

    def mp_helper(self, _X, edge_index, edge_type, _node_type, _node_feature_extra, return_attention_weights=True):
        all_gnn_attn = []
        all_edge_map = []
        for _ in range(self.k):
            if return_attention_weights:
                _X, (edge_idx, edge_weight) = self.gnn_layers[_](_X, edge_index, edge_type, _node_type,
                                                                 _node_feature_extra)
                # 取最后一个注意力头的注意力分数
                gnn_attn = edge_weight[:, - 1]
                edge_map = edge_idx

                gnn_attn = gnn_attn[0:500]
                edge_map = edge_map[:, 0:500]
                # gnn_attn = gnn_attn
                # edge_map = edge_map

                all_gnn_attn.append(gnn_attn)
                all_edge_map.append(edge_map)
            else:
                _X = self.gnn_layers[_](_X, edge_index, edge_type, _node_type, _node_feature_extra)
            _X = self.activation(_X)
            _X = F.dropout(_X, self.dropout_rate, training=self.training)
        if return_attention_weights:
            return _X, (all_edge_map, all_gnn_attn)
        else:
            return _X

    def forward(self, H, A, node_type, node_score, cache_output=False, return_attention_weights=True):
        """
        H: tensor of shape (batch_size, n_node, d_node)
            node features from the previous layer
        A: (edge_index, edge_type)
        node_type: long tensor of shape (batch_size, n_node)
            0 == question entity; 1 == answer choice entity; 2 == other node; 3 == context node
        node_score: tensor of shape (batch_size, n_node, 1)
        """
        _batch_size, _n_nodes = node_type.size()

        # Embed type
        T = make_one_hot(node_type.view(-1).contiguous(), self.n_ntype).view(_batch_size, _n_nodes, self.n_ntype)
        node_type_emb = self.activation(self.emb_node_type(T))  # [batch_size, n_node, dim/2]

        # Embed score
        if self.basis_f == 'sin':
            js = torch.arange(self.hidden_size // 2).unsqueeze(0).unsqueeze(0).float().to(
                node_type.device)  # [1,1,dim/2]
            js = torch.pow(1.1, js)  # [1,1,dim/2]
            B = torch.sin(js * node_score)  # [batch_size, n_node, dim/2]
            node_score_emb = self.activation(self.emb_score(B))  # [batch_size, n_node, dim/2]
        elif self.basis_f == 'id':
            B = node_score
            node_score_emb = self.activation(self.emb_score(B))  # [batch_size, n_node, dim/2]
        elif self.basis_f == 'linact':
            B = self.activation(self.B_lin(node_score))  # [batch_size, n_node, dim/2]
            node_score_emb = self.activation(self.emb_score(B))  # [batch_size, n_node, dim/2]

        X = H
        edge_index, edge_type = A  # edge_index: [2, total_E]   edge_type: [total_E, ]  where total_E is for the batched graph
        _X = X.view(-1, X.size(2)).contiguous()  # [`total_n_nodes`, d_node] where `total_n_nodes` = b_size * n_node
        _node_type = node_type.view(-1).contiguous()  # [`total_n_nodes`, ]
        _node_feature_extra = torch.cat([node_type_emb, node_score_emb], dim=2).view(_node_type.size(0),
                                                                                     -1).contiguous()  # [`total_n_nodes`, dim]

        if return_attention_weights:
            _X, (all_gnn_atten, all_edge_map) = self.mp_helper(_X, edge_index, edge_type, _node_type,
                                                               _node_feature_extra)
        else:
            _X = self.mp_helper(_X, edge_index, edge_type, _node_type, _node_feature_extra)

        X = _X.view(node_type.size(0), node_type.size(1), -1)  # [batch_size, n_node, dim]

        output = self.activation(self.Vh(H) + self.Vx(X))
        output = self.dropout(output)

        if return_attention_weights:
            return output, (all_gnn_atten, all_edge_map)
        else:
            return output


class QAGNN(nn.Module):
    def __init__(self, args, pre_dim, k, n_ntype, n_etype, sent_dim,
                 n_concept, concept_dim, concept_in_dim, n_attention_head,
                 fc_dim, n_fc_layer, p_emb, p_gnn, p_fc,
                 pretrained_concept_emb=None, freeze_ent_emb=True,
                 init_range=0, gram_dim=768):
        super().__init__()
        self.pre_dim = pre_dim
        self.init_range = init_range
        # token Embedding
        self.concept_emb = CustomizedEmbedding(concept_num=n_concept, concept_out_dim=concept_dim,
                                               use_contextualized=False, concept_in_dim=concept_in_dim,
                                               pretrained_concept_emb=pretrained_concept_emb,
                                               freeze_ent_emb=freeze_ent_emb)
        # 句子（sentence）映射到概念（token，词汇，也可称为concept）
        # self.svec2nvec = nn.Linear((sent_dim), concept_dim)  # 现在使用的是 DL 模型和
        self.svec2nvec = nn.Linear((sent_dim+44)*3, concept_dim)  # 现在使用的是 DL 模型和
        # self.svec2nvec = nn.Linear((sent_dim+44), concept_dim)

        self.concept_dim = concept_dim

        self.activation = GELU()
        # 图神经网络
        self.gnn = QAGNN_Message_Passing(args, k=k, n_ntype=n_ntype, n_etype=n_etype,
                                         input_size=concept_dim, hidden_size=concept_dim, output_size=concept_dim,
                                         dropout=p_gnn)


        # self.pooler = MultiheadAttPoolLayer(n_attention_head, (sent_dim)+44, concept_dim)  # 混合多层注意力机制
        self.pooler = MultiheadAttPoolLayer(n_attention_head, ((sent_dim)+44)*3, concept_dim)  # 混合多层注意力机制

        # self.fc = MLP( (sent_dim ), fc_dim, self.pre_dim, n_fc_layer, p_fc,layer_norm=True)
        # self.fc = MLP(sent_dim*2, fc_dim, self.pre_dim, n_fc_layer, p_fc, layer_norm=True) #不加入graph的embedding和预训练模型
        # self.fc = MLP(sent_dim * 2 + 44, fc_dim, self.pre_dim, n_fc_layer, p_fc, layer_norm=True)  # 不加入graph的embedding和预训练模型
        # self.fc = MLP(sent_dim * 3 + 44, fc_dim, self.pre_dim, n_fc_layer, p_fc, layer_norm=True)  # 不加入graph的embedding和预训练模型
        self.fc = MLP((sent_dim+44)*3, fc_dim, self.pre_dim, n_fc_layer, p_fc, layer_norm=True) #不加入graph的embedding和预训练模型
        self.is_pretrain = False
        if self.is_pretrain:
            self.fc = MLP((sent_dim+44), fc_dim, self.pre_dim, n_fc_layer, p_fc, layer_norm=True)
        else:
            self.use_graph = True
            if self.use_graph:
                self.fc = MLP((sent_dim+44)*2+768, fc_dim, self.pre_dim, n_fc_layer, p_fc, layer_norm=True)
                # self.fc = MLP((sent_dim+44)+768, fc_dim, self.pre_dim, n_fc_layer, p_fc, layer_norm=True)
            else:
                self.fc = MLP((sent_dim+44)*2, fc_dim, self.pre_dim, n_fc_layer, p_fc, layer_norm=True)
        # self.fc2 = MLP((sent_dim + 44) , fc_dim, self.pre_dim, n_fc_layer, p_fc, layer_norm=True)
        self.fc2 = MLP((sent_dim + 44)*3, fc_dim, self.pre_dim, n_fc_layer, p_fc, layer_norm=True)
        
        # self.fc2 = MLP((sent_dim + 44) * 2 + 768+200, fc_dim, self.pre_dim, n_fc_layer, p_fc, layer_norm=True)
        # self.fc2 = MLP((sent_dim + 44) * 2, fc_dim, self.pre_dim, n_fc_layer, p_fc, layer_norm=True)
        self.dropout_e = nn.Dropout(p_emb)
        self.dropout_fc = nn.Dropout(p_fc)
        self.dropout_g = nn.Dropout(0.9)  # mimic iii
        self.dropout_z = nn.Dropout(0.9)
        # 为了使得各个张量的最后一个维度一致，我们需要使用线性变换
        self.linear_seq_graph = torch.nn.Linear(768, 256)  # 用于将graph_seq_emb的尺寸从768变换到812
        self.linear_graph = torch.nn.Linear(100, 256)  # 用于将graph_seq_emb的尺寸从768变换到812
        self.linear_Z = torch.nn.Linear(100, 256)  # 用于将graph_seq_emb的尺寸从768变换到812
        self.linear_sent_vec = torch.nn.Linear(812, 256)  # 用于将sent_vecs的尺寸从812变换到256
        self.linear_dl_vec = torch.nn.Linear(812, 256)
        self.linear_concat = torch.nn.Linear(768+812+200, 812)

        self.activateOut = torch.nn.Sigmoid()

        if init_range > 0:
            self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, sent_vecs, dl_vec, concept_ids, node_type_ids, node_scores, adj_lengths, adj, emb_data=None,
                cache_output=False, return_attention_weights=True, return_P_emb=False,simp_emb=None, graph_seq_emb=None, isPretrain=False):
        """
        用 hita 的编码作为上下文编码

        """
        # 改
        gnn_input0 = self.activation(self.svec2nvec(sent_vecs)).unsqueeze(1)  # (batch_size, 1, dim_node)
        gnn_input1 = self.concept_emb(concept_ids[:, 1:] - 1, emb_data)  # (batch_size, n_node-1, dim_node) # 重点！！
        gnn_input1 = gnn_input1.to(node_type_ids.device)
        gnn_input = self.dropout_e(torch.cat([gnn_input0, gnn_input1], dim=1))  # (batch_size, n_node, dim_node)

        # Normalize node sore (use norm from Z)
        _mask = (torch.arange(node_scores.size(1), device=node_scores.device) < adj_lengths.unsqueeze(
            1)).float()  # 0 means masked out #[batch_size, n_node]
        node_scores = -node_scores
        node_scores = node_scores - node_scores[:, 0:1, :]  # [batch_size, n_node, 1]
        node_scores = node_scores.squeeze(2)  # [batch_size, n_node]
        node_scores = node_scores * _mask
        mean_norm = (torch.abs(node_scores)).sum(dim=1) / adj_lengths  # [batch_size, ]
        node_scores = node_scores / (mean_norm.unsqueeze(1) + 1e-05)  # [batch_size, n_node]
        node_scores = node_scores.unsqueeze(2)  # [batch_size, n_node, 1]

        if return_attention_weights:
            gnn_output, (edge_idx, edge_weight) = self.gnn(gnn_input, adj, node_type_ids, node_scores)
        else:
            gnn_output = self.gnn(gnn_input, adj, node_type_ids, node_scores)

        Z_vecs = gnn_output[:, 0]  # (batch_size, dim_node)

        mask = torch.arange(node_type_ids.size(1), device=node_type_ids.device) >= adj_lengths.unsqueeze(
            1)  # 1 means masked out

        mask = mask | (node_type_ids == 3)  # pool over all KG nodes
        mask[mask.all(1), 0] = 0  # a temporary solution to avoid zero node

        sent_vecs_for_pooler = sent_vecs
        graph_vecs, pool_attn = self.pooler(sent_vecs_for_pooler, gnn_output, mask)

        if cache_output:
            self.concept_ids = concept_ids
            self.adj = adj
            self.pool_attn = pool_attn

        concat = self.dropout_fc(sent_vecs)  # 删除 graph embedidng 和 预训练模型
        # concat = sent_vecs  # 复现hita的结果
        
        if isPretrain:
            logits = self.fc(concat)  # bs*961
            logits = self.activateOut(logits)  # bs*5985
            if return_attention_weights:
                return logits, pool_attn, (edge_idx, edge_weight)
            else:
                return logits, pool_attn


        if return_P_emb:
            return concat, pool_attn
        else:
            # 1. 使用相似患者的编码
            if graph_seq_emb is not None:
                concat = self.dropout_fc(torch.cat((sent_vecs, dl_vec, graph_seq_emb), 1))  # Z_vecs, graph_vecs:  torch.Size([16, 100])
            else:
                # concat = self.dropout_fc(torch.cat((sent_vecs, dl_vec), 1))  # Z_vecs, graph_vecs:  torch.Size([16, 100])
                
                # 不加入 dl_vec
                concat = self.dropout_fc(sent_vecs)  # Z_vecs, graph_vecs:  torch.Size([16, 100])
                # concat = sent_vecs  # Z_vecs, graph_vecs:  torch.Size([16, 100])

            logits = self.fc2(concat)  # bs*961
            logits = self.activateOut(logits) # bs*5985
            if return_attention_weights:
                return logits, pool_attn, (edge_idx, edge_weight)
            else:
                return logits, pool_attn

class FeatureFusion(nn.Module):
    def __init__(self, input_dims, output_dim):
        super(FeatureFusion, self).__init__()
        self.input_dims = input_dims
        self.output_dim = output_dim

        # 线性变换层，用于生成查询（Query）、键（Key）和值（Value）
        self.query_layers = nn.ModuleList([nn.Linear(in_dim, output_dim) for in_dim in input_dims])
        self.key_layers = nn.ModuleList([nn.Linear(in_dim, output_dim) for in_dim in input_dims])
        self.value_layers = nn.ModuleList([nn.Linear(in_dim, output_dim) for in_dim in input_dims])

    def forward(self, *tensors):
        # 确保输入的张量数量与初始化时相同
        assert len(tensors) == len(
            self.input_dims), "Number of input tensors must match number of input dimensions."

        # 生成查询、键和值
        queries = torch.stack([layer(tensor) for tensor, layer in zip(tensors, self.query_layers)])
        keys = torch.stack([layer(tensor) for tensor, layer in zip(tensors, self.key_layers)])
        values = torch.stack([layer(tensor) for tensor, layer in zip(tensors, self.value_layers)])

        # 计算注意力分数：点积后应用 softmax
        attention_scores = torch.matmul(queries.transpose(0, 1), keys.transpose(0, 1).transpose(1, 2))
        attention_scores = attention_scores / (self.output_dim ** 0.5)  # 缩放点积
        attention_weights = F.softmax(attention_scores, dim=-1)

        # 使用注意力权重对值进行加权
        weighted_values = torch.matmul(attention_weights, values.transpose(0, 1))

        # 按特征维度将加权的张量拼接起来
        concatenated_output = weighted_values.reshape(weighted_values.size(0), -1)

        return concatenated_output, attention_weights

class AttentionConcat(nn.Module):
    def __init__(self, sent_vec_dim, other_vec_dims):
        super(AttentionConcat, self).__init__()
        self.align_dim = sent_vec_dim
        self.dropout = nn.Dropout(0.7)

        # 为维度不同的张量创建线性层
        self.align_layers = nn.ModuleDict()
        for i, dim in enumerate(other_vec_dims):
            if dim != sent_vec_dim:
                self.align_layers[str(i)] = nn.Linear(dim, sent_vec_dim)

    def attention(self, query, key):
        scores = torch.matmul(query, key.transpose(-2, -1)) / (key.size(-1) ** 0.5)
        return F.softmax(scores, dim=-1)

    def forward(self, sent_vecs, other_vecs):
        final_out_feature = 0

        for i, vec in enumerate(other_vecs):
            # 如果需要，对张量进行尺寸调整
            if str(i) in self.align_layers:
                vec = self.align_layers[str(i)](vec)

            # 计算加权求和
            final_out_feature += self.attention(sent_vecs, vec) @ vec

        final_out_feature = self.dropout(final_out_feature/4)
        # 拼接 final_out_feature 和 sent_vecs
        return torch.cat([final_out_feature, sent_vecs], dim=-1)


class LM_QAGNN3(nn.Module):
    def __init__(self, args, pre_dim, model_name, k, n_ntype, n_etype,
                 n_concept, concept_dim, concept_in_dim, n_attention_head,
                 fc_dim, n_fc_layer, p_emb, p_gnn, p_fc,
                 pretrained_concept_emb=None, freeze_ent_emb=True,
                 init_range=0.0, encoder_config={}, hita_config={}):
        super().__init__()
        # 采用SciBert的预训练模型，作为TextEncoder
        self.encoder_PreTrain = TextEncoder(model_name, **encoder_config)
        self.encoder_HITA = TransformerTime(**hita_config)

        # self.encoder_HITA = TextEncoder(model_name, **encoder_config) # 在QA_GNN中用hita代替预训练模型编码，然后将预训练模型编码结果作为单独的表征cat到QA_GNN的编码中
        # self.encoder_PreTrain = TransformerTime(**hita_config)
        self.decoder = QAGNN(args, pre_dim, k, n_ntype, n_etype,
                             self.encoder_PreTrain.sent_dim,
                             n_concept, concept_dim, concept_in_dim, n_attention_head,
                             fc_dim, n_fc_layer, p_emb, p_gnn, p_fc,
                             pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb,
                             init_range=init_range)
        self.use_gram_emb = True

        self.fc = MLP(768 + 44, fc_dim, 5985, n_fc_layer, p_fc,
                      layer_norm=True)  # 不加入graph的embedding和预训练模型

        self.activateOut = torch.nn.Sigmoid()

        self.dropout_att = nn.Dropout(0.2)
        self.dropout_fc = nn.Dropout(0.2)

        self.w1 = torch.nn.Parameter(torch.randn(768+44, 768+44))
        self.cross_attention = CrossAttention(768+44, 768+44)

        self.pretrained_model = torch.load('saved_models/best_qa_hita_model.pt')

        # self.w1 = torch.nn.Parameter(torch.randn(768+44+768, 768+44+768))
        # self.cross_attention = CrossAttention(768+44+768, 768+44+768)

    def forward(self, simPatients, main_codes, sub_codes1, sub_codes2, ages, genders, ethnics,
                diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths,
                seq_time_step2, *inputs, layer_id=-1, cache_output=False, detail=False,
                return_attention_weights=True, return_hita_attention=True, return_P_emb=False,
                return_emb=True,simp_emb=None, use_graph=False, isPretrain=False):

        P_emb = self.pretrained_model(simPatients, main_codes, sub_codes1, sub_codes2, ages, genders, ethnics,
                diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths,
                seq_time_step2, *inputs, layer_id=-1, cache_output=False, detail=False,
                return_attention_weights=True, return_hita_attention=True, return_P_emb=True,
                return_emb=True,simp_emb=None, use_graph=False)


        if return_emb:
            return self.get_pretrain_emb(simPatients, main_codes, sub_codes1, sub_codes2, ages, genders, ethnics,
                    diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths,
                    seq_time_step2, *inputs, layer_id=-1, cache_output=False, detail=False,
                    return_attention_weights=True, return_hita_attention=True, return_P_emb=return_P_emb,
                                         simp_emb=None, use_graph=use_graph, isPretrain=isPretrain)
        else:
            return_P_emb = False
            if use_graph:
                use_graph = True
            else:
                use_graph = False
            simp_emb = simp_emb
            return self.get_pretrain_emb(simPatients, main_codes, sub_codes1, sub_codes2, ages, genders, ethnics,
                    diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths,
                    seq_time_step2, *inputs, layer_id=-1, cache_output=False, detail=False,
                    return_attention_weights=True, return_hita_attention=True, return_P_emb=return_P_emb,
                                         simp_emb=simp_emb, use_graph=use_graph, isPretrain=isPretrain)

    def multi_head_attention_forward(self, query, key, value, num_heads, dropout):
        # 获得维度信息
        batch_size = query.shape[0]
        query_len, key_len, value_len = query.shape[1], key.shape[1], value.shape[1]
        head_dim = query.shape[2] // num_heads
        assert head_dim * num_heads == query.shape[2], "Embedding size must be divisible by num_heads"

        # 重塑QKV以便我们有多个头
        query = query.reshape(batch_size, query_len, num_heads, head_dim)
        key = key.reshape(batch_size, key_len, num_heads, head_dim)
        value = value.reshape(batch_size, value_len, num_heads, head_dim)

        # 将QKV转换为多头形式，然后计算点积注意力
        scales = query @ key.transpose(-2, -1) / (head_dim ** 0.5)
        attention = torch.softmax(scales, dim=-1)
        attention = F.dropout(attention, p=dropout, training=True)
        out = attention @ value

        # 重塑输出以合并头部
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, num_heads * head_dim)

        return out

    def sum_sim_patients2(self, sim_P, P_emb, num_heads=4, dropout=0.2):
        P_emb_unsqueezed = P_emb.unsqueeze(1)

        # 应用多头注意力机制
        attention_output = self.multi_head_attention_forward(
            P_emb_unsqueezed, sim_P, sim_P, num_heads, dropout
        )

        print('attention_output: ',attention_output.shape)
        attention_output = attention_output.sum(dim=1)

        return attention_output

    def sum_sim_patients(self, sim_P, P_emb):
        attention_weight = self.cross_attention(P_emb, sim_P) # [bsz, 1, 7]
        # print('attention_weight: ',attention_weight.shape)
        # print('sim_P: ',sim_P.shape)
        attended_sim_P = attention_weight.squeeze().unsqueeze(2) * sim_P
        attended_sim_P = attended_sim_P.sum(dim=1)
        return attended_sim_P

    def batch_graph(self, edge_index_init, edge_type_init, n_nodes):
        # edge_index_init: list of (n_examples, ). each entry is torch.tensor(2, E)
        # edge_type_init:  list of (n_examples, ). each entry is torch.tensor(E, )
        n_examples = len(edge_index_init)
        edge_index = [edge_index_init[_i_] + _i_ * n_nodes for _i_ in range(n_examples)]
        edge_index = torch.cat(edge_index, dim=1)  # [2, total_E]
        edge_type = torch.cat(edge_type_init, dim=0)  # [total_E, ]
        return edge_index, edge_type

class LM_QAGNN(nn.Module):
    def __init__(self, args, pre_dim, model_name, k, n_ntype, n_etype,
                 n_concept, concept_dim, concept_in_dim, n_attention_head,
                 fc_dim, n_fc_layer, p_emb, p_gnn, p_fc,
                 pretrained_concept_emb=None, freeze_ent_emb=True,
                 init_range=0.0, encoder_config={}, hita_config={}):
        super().__init__()
        # 采用SciBert的预训练模型，作为TextEncoder
        self.encoder_PreTrain = TextEncoder(model_name, **encoder_config)
        self.encoder_HITA = TransformerTime(**hita_config)

        # self.encoder_HITA = TextEncoder(model_name, **encoder_config) # 在QA_GNN中用hita代替预训练模型编码，然后将预训练模型编码结果作为单独的表征cat到QA_GNN的编码中
        # self.encoder_PreTrain = TransformerTime(**hita_config)
        self.decoder = QAGNN(args, pre_dim, k, n_ntype, n_etype,
                             self.encoder_PreTrain.sent_dim,
                             n_concept, concept_dim, concept_in_dim, n_attention_head,
                             fc_dim, n_fc_layer, p_emb, p_gnn, p_fc,
                             pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb,
                             init_range=init_range)
        self.use_gram_emb = True


        self.fc = MLP(768 + 44, fc_dim, 5985, n_fc_layer, p_fc,
                      layer_norm=True)  # 不加入graph的embedding和预训练模型

        self.activateOut = torch.nn.Sigmoid()

        self.dropout_att = nn.Dropout(0.2)
        self.dropout_fc = nn.Dropout(0.2)

        self.w1 = torch.nn.Parameter(torch.randn(768+44, 768+44))
        self.cross_attention = CrossAttention(768+44, 768+44)
        self.dropout_hv = nn.Dropout(p=0.3)  # 0.3
        self.dropout_seqG = nn.Dropout(p=0.7)  # 0.7
        self.dropout_sent = nn.Dropout(p=0.2) # h
        # self.dropout_sent = nn.Dropout(p=0.2) # e
        
        # 交互层 - 使用多头注意力机制
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=768+44,
            num_heads=2, # 2
            dropout=0.7 # 0.7
        )
        
        # 非线性变换层，生成最终的hv
        self.hv_generator0 = nn.Sequential(
            nn.Linear((768+44) * 2, 768+44),
            # nn.ReLU(),
            nn.Dropout(p=0.2),
            # nn.Linear(768, 768),
            # nn.LayerNorm(768)
        )
        # 非线性变换层，生成最终的hv
        self.hv_generator = nn.Sequential(
            nn.Linear((768+44) * 3, 768+44),
            # nn.ReLU(),
            nn.Dropout(p=0.2),
            # nn.Linear(768, 768),
            # nn.LayerNorm(768)
        )
        self.proj_lab = nn.Linear(128, 768+44)
        
    
    
    def generate_hv(self, vecs_hita, lab_embeddings, seqG_embeddings):
        """
        动态生成互补信息向量hv
        """
        # 将三个表征堆叠在一起
        combined = torch.stack([vecs_hita, lab_embeddings, seqG_embeddings], dim=0)  # [3, bs, dim]
        
        device = combined.device
        self.cross_attention = self.cross_attention.to(device)
        
        # 交叉注意力
        attn_output, _ = self.cross_attention(
            combined, combined, combined
        )  # [3, bs, dim]
        
        # sum后，第一个维度压缩
        hv = attn_output.mean(dim=0).squeeze(0)
        
        return hv
    
    def generate_hv0(self, vecs_hita, lab_embeddings):
        """
        动态生成互补信息向量hv
        """
        # 将三个表征堆叠在一起
        combined = torch.stack([vecs_hita, lab_embeddings], dim=0)  # [3, bs, dim]
        
        device = combined.device
        self.cross_attention = self.cross_attention.to(device)
        
        # 交叉注意力
        attn_output, _ = self.cross_attention(
            combined, combined, combined
        )  # [3, bs, dim]
        
        # sum后，第一个维度压缩
        hv = attn_output.mean(dim=0).squeeze(0)

        return hv
        
       
    def generate_hv2(self, vecs_hita, lab_embeddings, seqG_embeddings):
        """
        动态生成互补信息向量hv
        """
        combined1 = torch.stack([vecs_hita, lab_embeddings], dim=0)  # [2, bs, dim]
        combined2 = torch.stack([vecs_hita, seqG_embeddings], dim=0)  # [2, bs, dim]
        
        device = combined1.device
        self.cross_attention1 = self.cross_attention.to(device)
        self.cross_attention2 = self.cross_attention.to(device)

        # 交叉注意力
        attn_output1, _ = self.cross_attention1(
            combined1, combined1, combined1
        )  # [2, bs, dim]
        attn_output2, _ = self.cross_attention2(
            combined2, combined2, combined2
        )  # [2, bs, dim]
        
        # sum后，第一个维度压缩
        hv1 = attn_output1.mean(dim=0).squeeze(0)
        hv2 = attn_output2.mean(dim=0).squeeze(0)
        
        # 将注意力的输出展平并连接
        # attn_output = attn_output.transpose(0, 1)  # [bs, 3, dim]
        # flat_output = attn_output.reshape(attn_output.size(0), -1)  # [bs, 3*dim]
        
        # # 生成动态hv
        # hv = self.hv_generator(flat_output)  # [bs, hv_dim]
        
        return hv1, hv2
    
    
    def diversity_entropy_loss(self, hv1, hv2):
        # 基于余弦相似度计算互补信息的多样性. 
        hv1_norm = F.normalize(hv1, dim=1)
        hv2_norm = F.normalize(hv2, dim=1)  

        cos = torch.cosine_similarity(hv1_norm, hv2_norm, dim=1)  # [bs]    
        diversity_ortho = torch.abs(cos).mean()
        
        return diversity_ortho
    
    def orthogonal_loss(self, hv, vec1, vec2):
        # 确保向量归一化
        hv_norm = F.normalize(hv, dim=1)
        vec1_norm = F.normalize(vec1, dim=1)
        vec2_norm = F.normalize(vec2, dim=1)
        
        # 计算余弦相似度，期望接近0（正交）
        cos1 = torch.cosine_similarity(hv_norm, vec1_norm, dim=1) 
        cos2 = torch.cosine_similarity(hv_norm, vec2_norm, dim=1) 
        loss = torch.abs(cos1) + torch.abs(cos2)
        
        return loss.mean()
    
    def total_loss(self, vecs_hita, lab_embeddings, seqG_embeddings, hv1, hv2):
        # 正交约束损失
        ortho_loss = self.orthogonal_loss(hv1, vecs_hita, lab_embeddings) + \
                    self.orthogonal_loss(hv2, vecs_hita, seqG_embeddings)
        
        # 多样性损失
        div_loss = self.diversity_entropy_loss(hv1, hv2)  # 或使用其他多样性损失
        
        # 组合所有损失
        total = 0.1*ortho_loss + div_loss
        return total
    
    def contrastive_loss0(self, patient_embeddings, sent_vec):
        # 确保输入在同一设备上
        device = sent_vec.device
        patient_embeddings = patient_embeddings.to(device)
        
        # 归一化向量
        patient_embeddings = F.normalize(patient_embeddings, dim=1)
        sent_vec = F.normalize(sent_vec, dim=1)
        
        # 计算相似度矩阵
        logits = torch.mm(patient_embeddings, sent_vec.t())
        # 对角线上的元素应该最大(正样本对)
        labels = torch.arange(logits.size(0), device=device)
        
        # 计算交叉熵损失
        loss = F.cross_entropy(logits, labels)
        
        return loss

    def contrastive_loss(self, patient_embeddings, sent_vec, lambda_weight=0.7, temperature=1.0):

        # 确保嵌入是归一化的
        patient_embeddings = F.normalize(patient_embeddings, p=2, dim=1)
        sent_vec = F.normalize(sent_vec, p=2, dim=1)

        # 计算相似度矩阵，形状为 (batch_size, batch_size)
        similarity_matrix = torch.matmul(patient_embeddings, sent_vec.t()) / temperature

        # 第一部分损失：log [exp(s_ii) / sum_k exp(s_ik)]
        log_prob = F.log_softmax(similarity_matrix, dim=1)
        loss_part1 = -log_prob.diag().sum()  # 取对角线元素的和并取负号

        # 第二部分损失：log [exp(s_ii) / sum_{k != i} exp(s_ik)]
        exp_sim = torch.exp(similarity_matrix)
        sum_exp_neg = exp_sim.sum(dim=1) - exp_sim.diag()  # 减去 exp(s_ii)
        sum_exp_neg = sum_exp_neg.clamp(min=1e-10)  # 防止除以零
        loss_part2 = -torch.log(torch.exp(similarity_matrix.diag()) / sum_exp_neg).sum()

        # 组合损失并归一化
        loss = (loss_part1 + lambda_weight * loss_part2) / patient_embeddings.size(0)

        return loss

    def compute_orthogonality_loss(self, hv, vec1, vec2, vec3):
        """
        Compute soft orthogonality constraints using squared inner products
        """     
        # Normalize vectors to make constraints scale-invariant
        hv_norm = F.normalize(hv, dim=-1)
        vec1_norm = F.normalize(vec1, dim=-1)
        vec2_norm = F.normalize(vec2, dim=-1)
        vec3_norm = F.normalize(vec3, dim=-1)
        
        # 计算余弦相似度,direct=1确保计算batch内每对向量的相似度
        cos1 = torch.cosine_similarity(hv_norm, vec1_norm, dim=1)  # [bs]
        cos2 = torch.cosine_similarity(hv_norm, vec2_norm, dim=1)  # [bs]
        cos3 = torch.cosine_similarity(hv_norm, vec3_norm, dim=1)  # [bs]
        
        # 取绝对值,因为我们希望cos值接近0(正交)
        loss_ortho1 = torch.mean(torch.abs(cos1))
        loss_ortho2 = torch.mean(torch.abs(cos2))
        loss_ortho3 = torch.mean(torch.abs(cos3))
        
        # ortho_loss = loss_ortho1 + loss_ortho2 + loss_ortho3
        
        ortho_loss = loss_ortho1 + loss_ortho2
        
        return ortho_loss
        
    # 非定义，直接上的是forward函数，即这部分
    def get_pretrain_emb(self, phrase, lab_embeddings, seqG_embeddings, simPatients, main_codes, sub_codes1, sub_codes2, ages, genders, ethnics,
                diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths,
                seq_time_step2, *inputs, layer_id=-1, cache_output=False, detail=False,
                return_attention_weights=True, return_hita_attention=True, return_P_emb=False,
                         simp_emb=None, use_graph=False, isPretrain=False):
        bs, nc = inputs[0].size(0), inputs[0].size(1)
        device = inputs[0].device
        
        edge_index_orig, edge_type_orig = inputs[-2:]  # 边相关数据
        _inputs = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[:-6]] + [
            x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[-6:-2]] + [sum(x, []) for x in inputs[-2:]]

        *lm_inputs, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type = _inputs  # concept_ids来源 这里的edge_index 的 bsz=16
        edge_index, edge_type = self.batch_graph(edge_index, edge_type, concept_ids.size(1))
        adj = (edge_index.to(node_type_ids.device),
               edge_type.to(node_type_ids.device))  # edge_index: [2, total_E]   edge_type: [total_E, ]

        vecs_hita, visit_att, self_att= self.encoder_HITA(simPatients,
                                                                main_codes, sub_codes1, sub_codes2, ages, genders,
                                                                ethnics,
                                                                diagnosis_codes, seq_time_step,
                                                                mask_mult, mask_final, mask_code,
                                                                lengths, seq_time_step2,
                                                                return_hita_attention, use_graph=use_graph)  # torch.Size([16, 768])
        
         # Expand hv to batch size
        self.proj_lab = self.proj_lab.to(device)
        self.hv_generator = self.hv_generator.to(device)
        lab_proj = self.proj_lab(lab_embeddings)  # [bs, 768]
        hv = self.generate_hv(vecs_hita, lab_proj, seqG_embeddings)
        # hv = self.generate_hv0(vecs_hita, lab_proj)

        # self.hv_generator2 = self.hv_generator2.to(device)
        # hv1, hv2 = self.generate_hv2(vecs_hita, lab_proj, seqG_embeddings) 
        
        
        if phrase == 'train':
            # 1. 正交约束损失
            ortho_loss = self.compute_orthogonality_loss(hv, vecs_hita, lab_proj, seqG_embeddings)
            # 2. 正交损失
            contrastive_loss = self.contrastive_loss(seqG_embeddings, vecs_hita)
            
            # 3. 正交损失 + 对比损失
            contrastive_loss = ortho_loss + contrastive_loss            
            
            # 4. 不用对比损失
            # contrastive_loss = ortho_loss
            
            # #4. contrastive_loss = 0
            # contrastive_loss = 0
        else:
            contrastive_loss = 0
            
        #  #============================================================================================================
        # # 复现hita的结果
        # sent_vec = vecs_hita
        # #============================================================================================================
        
        
        #============================================================================================================
        # 本文的结果
        
        # # if 'm':
        seqG_embeddings = self.dropout_seqG(seqG_embeddings)
        hv = self.dropout_hv(hv)
        hv = hv.unsqueeze(0) if len(hv.shape) == 1 else hv
        vecs_hita = vecs_hita.unsqueeze(0) if len(vecs_hita.shape) == 1 else vecs_hita
        seqG_embeddings = seqG_embeddings.unsqueeze(0) if len(seqG_embeddings.shape) == 1 else seqG_embeddings
        hv = hv.unsqueeze(0) if len(hv.shape) == 1 else hv
        sent_vec = torch.cat((vecs_hita, seqG_embeddings, hv), -1) # 患者表征 + 相同语意空间的SeqGemb + 互补信息 hv
        
        # w/o seqG_embeddings
        # sent_vec = torch.cat((vecs_hita, hv), -1) # 患者表征  +  互补信息 hv
        
        # w/o lab_test: 删除互补信息, 且仅保留对比损失
        # sent_vec = torch.cat((vecs_hita, seqG_embeddings), -1) # 患者表征 + 相同语意空间的SeqGemb

        
        # # m1 w/o hv消融,注意，需要将正交损失部分给删除，因为不使用互补信息了
        # sent_vec =  torch.cat((vecs_hita, seqG_embeddings), -1)
        
        # m2 w/o  hv+对比损失。（将对比损失和正交损失都删除）
        # 删除 对比损失，。将 contrastive_loss = 0

        # if task = 'h':
        # sent_vec = torch.cat((vecs_hita, seqG_embeddings, hv), -1) # 患者表征 + 相同语意空间的SeqGemb + 互补信息 hv
        # sent_vec = self.dropout_sent(sent_vec)
        
        # # if 2:
        # hv1 = self.dropout_hv(hv1)
        # hv2 = self.dropout_hv(hv2)
        # sent_vec = torch.cat((vecs_hita, hv1, hv2), -1)
        
        if return_attention_weights:
            logits, attn, (edge_idx, edge_weight) = self.decoder(sent_vec.to(node_type_ids.device),
                                                                    simp_emb,
                                                                    concept_ids,
                                                                    node_type_ids, node_scores, adj_lengths, adj,
                                                                    emb_data=None, cache_output=cache_output, return_P_emb=return_P_emb,simp_emb=simp_emb)
        else:
            logits, attn = self.decoder(sent_vec.to(node_type_ids.device),
                                        vecs_hita.to(node_type_ids.device),
                                        concept_ids,
                                        node_type_ids, node_scores, adj_lengths, adj,
                                        emb_data=None, cache_output=cache_output, return_P_emb=return_P_emb,simp_emb=simp_emb)

        if not detail:
            if return_attention_weights: # 训练的时候用到的是这个判断
                # return logits, attn, (edge_idx, edge_weight), visit_att, self_att, attention_weights, simP_att
                return contrastive_loss, logits, attn, (edge_idx, edge_weight), visit_att, self_att # 2024.12.05改
            else:
                return logits, attn
        else:
            if return_attention_weights:
                return contrastive_loss, logits, attn, concept_ids.view(bs, nc, -1), \
                    node_type_ids.view(bs, nc, -1), edge_index_orig, \
                    edge_type_orig, (edge_idx, edge_weight)
            else:
                return contrastive_loss, logits, attn, concept_ids.view(bs, nc, -1), \
                    node_type_ids.view(bs, nc, -1), edge_index_orig, \
                    edge_type_orig

    # 非定义，直接上的是forward函数，即这部分
    def forward(self, simPatients, main_codes, sub_codes1, sub_codes2, ages, genders, ethnics,
                diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths,
                seq_time_step2, *inputs, layer_id=-1, cache_output=False, detail=False,
                return_attention_weights=True, return_hita_attention=True, return_P_emb=False,
                return_emb=True,simp_emb=None, use_graph=False, isPretrain=False):
        if isPretrain:
            return self.get_pretrain_emb(simPatients, main_codes, sub_codes1, sub_codes2, ages, genders, ethnics,
                                         diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths,
                                         seq_time_step2, *inputs, layer_id=-1, cache_output=False, detail=False,
                                         return_attention_weights=True, return_hita_attention=True,
                                         return_P_emb=False,
                                         simp_emb=None, use_graph=False, isPretrain=True)

        if return_emb:
            return self.get_pretrain_emb(simPatients, main_codes, sub_codes1, sub_codes2, ages, genders, ethnics,
                    diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths,
                    seq_time_step2, *inputs, layer_id=-1, cache_output=False, detail=False,
                    return_attention_weights=True, return_hita_attention=True, return_P_emb=return_P_emb,
                                         simp_emb=None, use_graph=use_graph, isPretrain=isPretrain)
        else:
            return_P_emb = False
            if use_graph:
                use_graph = True
            else:
                use_graph = False
            simp_emb = simp_emb
            return self.get_pretrain_emb(simPatients, main_codes, sub_codes1, sub_codes2, ages, genders, ethnics,
                    diagnosis_codes, seq_time_step, mask_mult, mask_final, mask_code, lengths,
                    seq_time_step2, *inputs, layer_id=-1, cache_output=False, detail=False,
                    return_attention_weights=True, return_hita_attention=True, return_P_emb=return_P_emb,
                                         simp_emb=simp_emb, use_graph=use_graph, isPretrain=isPretrain)

    def multi_head_attention_forward(self, query, key, value, num_heads, dropout):
        # 获得维度信息
        batch_size = query.shape[0]
        query_len, key_len, value_len = query.shape[1], key.shape[1], value.shape[1]
        head_dim = query.shape[2] // num_heads
        assert head_dim * num_heads == query.shape[2], "Embedding size must be divisible by num_heads"

        # 重塑QKV以便我们有多个头
        query = query.reshape(batch_size, query_len, num_heads, head_dim)
        key = key.reshape(batch_size, key_len, num_heads, head_dim)
        value = value.reshape(batch_size, value_len, num_heads, head_dim)

        # 将QKV转换为多头形式，然后计算点积注意力
        scales = query @ key.transpose(-2, -1) / (head_dim ** 0.5)
        attention = torch.softmax(scales, dim=-1)
        attention = F.dropout(attention, p=dropout, training=True)
        out = attention @ value

        # 重塑输出以合并头部
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, num_heads * head_dim)

        return out

    def sum_sim_patients2(self, sim_P, P_emb, num_heads=4, dropout=0.2):
        # sim_P: [bsz, seq_len, embed_size], P_emb: [bsz, embed_size]
        # 将P_emb增加一个seq_len维度，以便我们可以使用它作为多头注意力的查询
        P_emb_unsqueezed = P_emb.unsqueeze(1)

        # 应用多头注意力机制
        attention_output = self.multi_head_attention_forward(
            P_emb_unsqueezed, sim_P, sim_P, num_heads, dropout
        )

        print('attention_output: ',attention_output.shape)
        attention_output = attention_output.sum(dim=1)

        # 将多头注意力的输出结果压缩回原来的维度
        # sum_P = attention_output.squeeze(1)
        return attention_output

    def batch_graph(self, edge_index_init, edge_type_init, n_nodes):
        # edge_index_init: list of (n_examples, ). each entry is torch.tensor(2, E)
        # edge_type_init:  list of (n_examples, ). each entry is torch.tensor(E, )
        n_examples = len(edge_index_init)
        edge_index = [edge_index_init[_i_] + _i_ * n_nodes for _i_ in range(n_examples)]
        edge_index = torch.cat(edge_index, dim=1)  # [2, total_E]
        edge_type = torch.cat(edge_type_init, dim=0)  # [total_E, ]
        return edge_index, edge_type

class CrossAttention(nn.Module):
    """交叉注意力模块"""
    def __init__(self, patient_rep_dim=768+44, similar_patients_rep_dim=768+44):
        super(CrossAttention, self).__init__()
        self.key_fc = nn.Linear(similar_patients_rep_dim, patient_rep_dim)
        self.query_fc = nn.Linear(patient_rep_dim, patient_rep_dim)
        self.scale = 1.0 / math.sqrt(patient_rep_dim)
        self.tanh = nn.Tanh()

    def forward(self, query, keys):
        # query: 当前患者表征, 形状 (batch_size, patient_rep_dim)
        # keys: 相似患者表征, 形状 (batch_size, num_similar_patients, similar_patients_rep_dim)
        query = self.query_fc(query).unsqueeze(1) # 转换查询，并增加一个维度以便广播
        # keys = self.key_fc(keys) # 转换键值
        energy = torch.bmm(query, keys.transpose(1, 2)) * self.scale # 计算能量值
        # energy = torch.bmm(query, keys.transpose(1, 2))
        attention_weights = F.softmax(energy, dim=2) # 应用softmax得到权重
        return attention_weights

class LM_QAGNN_DataLoader(object):

    def __init__(self, args, train_statement_path, train_adj_path,
                 dev_statement_path, dev_adj_path,
                 test_statement_path, test_adj_path,
                 batch_size, eval_batch_size, device, model_name, max_node_num=200, max_seq_length=20,
                 is_inhouse=False, inhouse_train_qids_path=None,
                 subsample=1.0, use_cache=True):
        super().__init__()  # 调用超类初始化
        self.args = args
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device0, self.device1 = device  # device=(device0, device1)
        self.is_inhouse = is_inhouse  # 是否选用内定数据，若有的话

        # 此name为编码器，encoder='cambridgeltl/SapBERT-from-PubMedBERT-fulltext'
        # 对应的model_type为‘bert’
        model_type = MODEL_NAME_TO_CLASS[model_name]  # 类型与可调度模型名字的映射
        (self.train_qids, self.train_HF_labels, self.train_Diag_labels, \
            self.train_main_codes, self.train_sub_code1s, self.train_sub_code2s, \
            self.train_ages, self.train_genders, self.train_ethnicities, \
            self.train_diagnosis_codes, self.train_seq_time_step, self.train_mask_mult, \
            self.train_mask_final, self.train_mask_code, self.train_lengths, self.train_seq_time_step2,
         self.train_main_diagnose_list, *self.train_encoder_data) = \
            load_input_tensors(train_statement_path, model_type, model_name, max_seq_length)

        (self.dev_qids, self.dev_HF_labels, self.dev_Diag_labels, \
            self.dev_main_codes, self.dev_sub_code1s, self.dev_sub_code2s, \
            self.dev_ages, self.dev_genders, self.dev_ethnicities, \
            self.dev_diagnosis_codes, self.dev_seq_time_step, self.dev_mask_mult, \
            self.dev_mask_final, self.dev_mask_code, self.dev_lengths, self.dev_seq_time_step2,
         self.main_diagnose_list, *self.dev_encoder_data) = \
            load_input_tensors(dev_statement_path, model_type, model_name, max_seq_length)

        # 选项数目
        num_choice = self.train_encoder_data[0].size(1)
        self.num_choice = num_choice
        print('num_choice:', num_choice)
        *self.train_decoder_data, self.train_adj_data = load_sparse_adj_data_with_contextnode(train_adj_path,
                                                                                              max_node_num, num_choice,
                                                                                              args)
        *self.dev_decoder_data, self.dev_adj_data = load_sparse_adj_data_with_contextnode(dev_adj_path, max_node_num,
                                                                                          num_choice, args)

        if test_statement_path is not None:
            (self.test_qids, self.test_HF_labels, self.test_Diag_labels, \
                self.test_main_codes, self.test_sub_code1s, self.test_sub_code2s, \
                self.test_ages, self.test_genders, self.test_ethnicities, \
                self.test_diagnosis_codes, self.test_seq_time_step, self.test_mask_mult, \
                self.test_mask_final, self.test_mask_code, self.test_lengths, self.test_seq_time_step2,
             self.test_main_diagnose_list, *self.test_encoder_data) = \
                load_input_tensors(test_statement_path, model_type, model_name, max_seq_length)
            *self.test_decoder_data, self.test_adj_data = load_sparse_adj_data_with_contextnode(test_adj_path,
                                                                                                max_node_num,
                                                                                                num_choice, args)
            # assert all(len(self.test_qids) == len(self.test_adj_data[0]) == x.size(0) for x in [self.test_labels] + self.test_encoder_data + self.test_decoder_data)

        # self.is_inhouse = 0 # 不要，不会改。修掉修掉！
        if self.is_inhouse:
            with open(inhouse_train_qids_path, 'r') as fin:
                inhouse_qids = set(line.strip() for line in fin)
            self.inhouse_train_indexes = torch.tensor(
                [i for i, qid in enumerate(self.train_qids) if qid in inhouse_qids])
            self.inhouse_test_indexes = torch.tensor(
                [i for i, qid in enumerate(self.train_qids) if qid not in inhouse_qids])

        assert 0. < subsample <= 1.
        if subsample < 1.:
            n_train = int(self.train_size() * subsample)
            assert n_train > 0
            self.train_qids = self.train_qids[:n_train]
            self.train_HF_labels = self.train_HF_labels[:n_train]
            self.train_Diag_labels = self.train_Diag_labels[:n_train]
            self.train_diagnosis_codes = self.train_diagnosis_codes[:n_train]
            self.train_seq_time_step = self.train_seq_time_step[:n_train]
            self.train_seq_time_step2 = self.train_seq_time_step2[:n_train]
            self.train_main_diagnose_list = self.train_main_diagnose_list[:n_train]
            self.train_mask_mult = self.train_mask_mult[:n_train]
            self.train_mask_final = self.train_mask_final[:n_train]
            self.train_mask_code = self.train_mask_code[:n_train]
            self.train_lengths = self.train_lengths[:n_train]
            self.train_encoder_data = [x[:n_train] for x in self.train_encoder_data]
            self.train_decoder_data = [x[:n_train] for x in self.train_decoder_data]
            self.train_adj_data = self.train_adj_data[:n_train]
            # assert all(len(self.train_qids) == len(self.train_adj_data[0]) == x.size(0) for x in [self.train_labels] + self.train_encoder_data + self.train_decoder_data)
            assert self.train_size() == n_train

    # 训练集大小（假如有内定，则以内定大小为主；否则，返回外部导入的数据长度
    def train_size(self):
        return self.inhouse_train_indexes.size(0) if self.is_inhouse else len(self.train_qids)

    # 验证集大小（未设定，自认不合理
    def dev_size(self):
        return len(self.dev_qids)

    # 测试集大小
    def test_size(self):
        if self.is_inhouse:
            return self.inhouse_test_indexes.size(0)
        else:
            return len(self.test_qids) if hasattr(self, 'test_qids') else 0
    def train_all_data(self):
        return

    def dev_all_data(self):
        return MultiGPUSparseAdjDataBatchGenerator(self.args, 'dev',
                                                   self.device0,
                                                   self.device1,
                                                   self.batch_size,
                                                   dev_indexes,
                                                   self.dev_qids,
                                                   self.dev_HF_labels,
                                                   self.dev_Diag_labels,
                                                   self.dev_main_codes,
                                                   self.dev_sub_code1s,
                                                   self.dev_sub_code2s,
                                                   self.dev_ages,
                                                   self.dev_genders,
                                                   self.dev_ethnicities,
                                                   self.dev_diagnosis_codes,
                                                   self.dev_seq_time_step,
                                                   self.dev_mask_mult,
                                                   self.dev_mask_final,
                                                   self.dev_mask_code,
                                                   self.dev_lengths,
                                                   self.dev_seq_time_step2,
                                                   self.dev_main_diagnose_list,
                                                   tensors0=self.dev_encoder_data,
                                                   tensors1=self.dev_decoder_data,
                                                   adj_data=self.dev_adj_data)

    def test_all_data(self):
        return MultiGPUSparseAdjDataBatchGenerator(self.args, 'test',
                                                   self.device0,
                                                   self.device1,
                                                   self.batch_size,
                                                   test_indexes,
                                                   self.test_qids,
                                                   self.test_HF_labels,
                                                   self.test_Diag_labels,
                                                   self.test_main_codes,
                                                   self.test_sub_code1s,
                                                   self.test_sub_code2s,
                                                   self.test_ages,
                                                   self.test_genders,
                                                   self.test_ethnicities,
                                                   self.test_diagnosis_codes,
                                                   self.test_seq_time_step,
                                                   self.test_mask_mult,
                                                   self.test_mask_final,
                                                   self.test_mask_code,
                                                   self.test_lengths,
                                                   self.test_seq_time_step2,
                                                   self.test_main_diagnose_list,
                                                   tensors0=self.test_encoder_data,
                                                   tensors1=self.test_decoder_data,
                                                   adj_data=self.test_adj_data)

    def train(self):
        if self.is_inhouse:
            n_train = self.inhouse_train_indexes.size(0)
            train_indexes = self.inhouse_train_indexes[torch.randperm(n_train)]
        else:
            train_indexes = torch.randperm(len(self.train_qids))
        return MultiGPUSparseAdjDataBatchGenerator(self.args, 'train',
                                                   self.device0,
                                                   self.device1,
                                                   self.batch_size,
                                                   train_indexes,
                                                   self.train_qids,
                                                   self.train_HF_labels,
                                                   self.train_Diag_labels,
                                                   self.train_main_codes,
                                                   self.train_sub_code1s,
                                                   self.train_sub_code2s,
                                                   self.train_ages,
                                                   self.train_genders,
                                                   self.train_ethnicities,
                                                   self.train_diagnosis_codes,
                                                   self.train_seq_time_step,
                                                   self.train_mask_mult,
                                                   self.train_mask_final,
                                                   self.train_mask_code,
                                                   self.train_lengths,
                                                   self.train_seq_time_step2,
                                                   self.train_main_diagnose_list,
                                                   tensors0=self.train_encoder_data,
                                                   tensors1=self.train_decoder_data,
                                                   adj_data=self.train_adj_data)

    def dev(self):
        return MultiGPUSparseAdjDataBatchGenerator(self.args, 'eval',
                                                   self.device0, self.device1,
                                                   self.eval_batch_size,
                                                   torch.arange(len(self.dev_qids)),
                                                   self.dev_qids,
                                                   self.dev_HF_labels,
                                                   self.dev_Diag_labels,
                                                   self.dev_main_codes,
                                                   self.dev_sub_code1s,
                                                   self.dev_sub_code2s,
                                                   self.dev_ages,
                                                   self.dev_genders,
                                                   self.dev_ethnicities,
                                                   self.dev_diagnosis_codes,
                                                   self.dev_seq_time_step,
                                                   self.dev_mask_mult,
                                                   self.dev_mask_final,
                                                   self.dev_mask_code,
                                                   self.dev_lengths,
                                                   self.dev_seq_time_step2,
                                                   self.dev_main_diagnose_list,
                                                   tensors0=self.dev_encoder_data,
                                                   tensors1=self.dev_decoder_data,
                                                   adj_data=self.dev_adj_data)

    def test(self):
        return MultiGPUSparseAdjDataBatchGenerator(self.args, 'eval', self.device0, self.device1,
                                                   self.eval_batch_size,
                                                   torch.arange(len(self.test_qids)),
                                                   self.test_qids,
                                                   self.test_HF_labels,
                                                   self.test_Diag_labels,
                                                   self.test_main_codes,
                                                   self.test_sub_code1s,
                                                   self.test_sub_code2s,
                                                   self.test_ages,
                                                   self.test_genders,
                                                   self.test_ethnicities,
                                                   self.test_diagnosis_codes,
                                                   self.test_seq_time_step,
                                                   self.test_mask_mult,
                                                   self.test_mask_final,
                                                   self.test_mask_code,
                                                   self.test_lengths,
                                                   self.test_seq_time_step2,
                                                    self.test_main_diagnose_list,
                                                   tensors0=self.test_encoder_data,
                                                   tensors1=self.test_decoder_data,
                                                   adj_data=self.test_adj_data)


###############################################################################
############################### GNN architecture ##############################
###############################################################################

from torch.autograd import Variable


def make_one_hot(labels, C):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        (N, ), where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.
    Returns : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C, where C is class number. One-hot encoded.
    '''
    labels = labels.unsqueeze(1)
    one_hot = torch.FloatTensor(labels.size(0), C).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    target = Variable(target)
    return target


from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter


# 在GAT中，沿着边的信息传递
class GATConvE(MessagePassing):
    """
    Args:
        emb_dim (int): dimensionality of GNN hidden states
        n_ntype (int): number of node types (e.g. 4)
        n_etype (int): number of edge relation types (e.g. 38)
    """

    def __init__(self, args, emb_dim, n_ntype, n_etype, edge_encoder, head_count=4, aggr="add"):
        super(GATConvE, self).__init__(aggr=aggr)
        self.args = args

        assert emb_dim % 2 == 0
        self.emb_dim = emb_dim

        self.n_ntype = n_ntype;
        self.n_etype = n_etype
        self.edge_encoder = edge_encoder

        # For attention（注意力部分
        self.head_count = head_count
        assert emb_dim % head_count == 0
        self.dim_per_head = emb_dim // head_count
        self.linear_key = nn.Linear(3 * emb_dim, head_count * self.dim_per_head)
        self.linear_msg = nn.Linear(3 * emb_dim, head_count * self.dim_per_head)
        self.linear_query = nn.Linear(2 * emb_dim, head_count * self.dim_per_head)

        self._alpha = None

        # For final MLP
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim),
                                       torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))

    def forward(self, x, edge_index, edge_type, node_type, node_feature_extra, return_attention_weights=True):
        # x: [N, emb_dim]
        # edge_index: [2, E]
        # edge_type [E,] -> edge_attr: [E, 39] / self_edge7+attr: [N, 39]
        # node_type [N,] -> headtail_attr [E, 8(=4+4)] / self_headtail_attr: [N, 8]
        # node_feature_extra [N, dim]

        # Prepare edge feature
        edge_vec = make_one_hot(edge_type, self.n_etype + 1)  # [E, 39]
        self_edge_vec = torch.zeros(x.size(0), self.n_etype + 1).to(edge_vec.device)
        self_edge_vec[:, self.n_etype] = 1

        head_type = node_type[edge_index[0]]  # [E,] #head=src
        tail_type = node_type[edge_index[1]]  # [E,] #tail=tgt
        head_vec = make_one_hot(head_type, self.n_ntype)  # [E,4]
        tail_vec = make_one_hot(tail_type, self.n_ntype)  # [E,4]
        headtail_vec = torch.cat([head_vec, tail_vec], dim=1)  # [E,8]
        self_head_vec = make_one_hot(node_type, self.n_ntype)  # [N,4]
        self_headtail_vec = torch.cat([self_head_vec, self_head_vec], dim=1)  # [N,8]

        edge_vec = torch.cat([edge_vec, self_edge_vec], dim=0)  # [E+N, ?]
        headtail_vec = torch.cat([headtail_vec, self_headtail_vec], dim=0)  # [E+N, ?]
        edge_embeddings = self.edge_encoder(torch.cat([edge_vec, headtail_vec], dim=1))  # [E+N, emb_dim]
        # edge_vec:torch.Size([18926, 35]) headtail_vec: torch.Size([22126, 8])
        # Add self loops to edge_index
        loop_index = torch.arange(0, x.size(0), dtype=torch.long, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, loop_index], dim=1)  # [2, E+N]

        x = torch.cat([x, node_feature_extra], dim=1)
        x = (x, x)
        aggr_out = self.propagate(edge_index, x=x, edge_attr=edge_embeddings)  # [N, emb_dim]
        out = self.mlp(aggr_out)

        alpha = self._alpha
        self._alpha = None

        if return_attention_weights:
            assert alpha is not None
            return out, (edge_index, alpha)
        else:
            return out

    def message(self, edge_index, x_i, x_j, edge_attr):  # i: tgt, j:src
        # print ("edge_attr.size()", edge_attr.size()) #[E, emb_dim]
        # print ("x_j.size()", x_j.size()) #[E, emb_dim]
        # print ("x_i.size()", x_i.size()) #[E, emb_dim]
        assert len(edge_attr.size()) == 2
        assert edge_attr.size(1) == self.emb_dim
        assert x_i.size(1) == x_j.size(1) == 2 * self.emb_dim
        assert x_i.size(0) == x_j.size(0) == edge_attr.size(0) == edge_index.size(1)

        key = self.linear_key(torch.cat([x_i, edge_attr], dim=1)).view(-1, self.head_count,
                                                                       self.dim_per_head)  # [E, heads, _dim]
        msg = self.linear_msg(torch.cat([x_j, edge_attr], dim=1)).view(-1, self.head_count,
                                                                       self.dim_per_head)  # [E, heads, _dim]
        query = self.linear_query(x_j).view(-1, self.head_count, self.dim_per_head)  # [E, heads, _dim]

        query = query / math.sqrt(self.dim_per_head)
        scores = (query * key).sum(dim=2)  # [E, heads]
        src_node_index = edge_index[0]  # [E,]
        alpha = softmax(scores, src_node_index)  # [E, heads] #group by src side node
        self._alpha = alpha

        # adjust by outgoing degree of src
        E = edge_index.size(1)  # n_edges
        N = int(src_node_index.max()) + 1  # n_nodes
        ones = torch.full((E,), 1.0, dtype=torch.float).to(edge_index.device)
        src_node_edge_count = scatter(ones, src_node_index, dim=0, dim_size=N, reduce='sum')[src_node_index]  # [E,]
        assert len(src_node_edge_count.size()) == 1 and len(src_node_edge_count) == E
        alpha = alpha * src_node_edge_count.unsqueeze(1)  # [E, heads]

        out = msg * alpha.view(-1, self.head_count, 1)  # [E, heads, _dim]
        return out.view(-1, self.head_count * self.dim_per_head)  # [E, emb_dim]
