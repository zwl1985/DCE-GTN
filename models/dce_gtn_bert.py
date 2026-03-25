import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers import TransformerLayer

class AspectAwareDynamicGCN(nn.Module):
    def __init__(self, hidden_dim, dropout_ratio, num_heads, edge_dim, norm):
        super().__init__()
        self.mem_dim = hidden_dim
        self.attention_heads = num_heads
        self.dropout = nn.Dropout(dropout_ratio)
        self.edge_dim = edge_dim

        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.Wxx = nn.Linear(hidden_dim, hidden_dim)

        edge_input_dim = num_heads + 2 * hidden_dim + (edge_dim if edge_dim > 0 else 0)
        self.Wx = nn.Linear(edge_input_dim, num_heads)

    def forward(self, x, weight_adj, e=None, src_mask=None):
        B, L, d = x.size()
        H = weight_adj.size(1)

        adj_avg = weight_adj.mean(dim=1)

        Ax = torch.bmm(adj_avg, x)
        Ax = self.W(Ax)
        gcn_outputs = F.relu(Ax)

        node1 = gcn_outputs.unsqueeze(1).expand(B, L, L, d)
        node2 = node1.permute(0, 2, 1, 3).contiguous()
        node_pair = torch.cat([node1, node2], dim=-1)

        weight_adj_perm = weight_adj.permute(0, 2, 3, 1).contiguous()

        if e is not None and self.edge_dim > 0:
            edge_feat = torch.cat([weight_adj_perm, node_pair, e], dim=-1)
        else:
            edge_feat = torch.cat([weight_adj_perm, node_pair], dim=-1)

        new_adj = self.Wx(edge_feat)
        new_adj = self.dropout(new_adj)
        new_adj = new_adj.permute(0, 3, 1, 2).contiguous()

        out = self.Wxx(gcn_outputs)
        return out, new_adj


class DCE_GTN(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.bert = bert

        self.deprel_emb = nn.Embedding(
            opt.deprel_size + 1, opt.deprel_dim, padding_idx=0
        ) if hasattr(opt, 'deprel_dim') and opt.deprel_dim > 0 else None

        self.linear_in = nn.Linear(opt.bert_dim, opt.hidden_dim)
        self.linear_out = nn.Linear(opt.hidden_dim + opt.bert_dim, opt.polarities_dim)

        self.bert_drop = nn.Dropout(opt.bert_dropout if hasattr(opt, 'bert_dropout') else 0.1)
        self.pooled_drop = nn.Dropout(opt.bert_dropout if hasattr(opt, 'bert_dropout') else 0.1)
        self.ffn_dropout = opt.ffn_dropout

        self.graph_convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.transformer_layers = nn.ModuleList()

        norm_class = nn.LayerNorm if opt.norm == 'ln' else nn.BatchNorm1d

        for _ in range(opt.num_layers):
            # NOTE: Only dec-gcn (dynamic graph convolution) is kept.
            graph_conv = AspectAwareDynamicGCN(
                hidden_dim=opt.hidden_dim,
                num_heads=opt.graph_conv_attention_heads,
                dropout_ratio=opt.ffn_dropout,
                norm=opt.norm,
                edge_dim=opt.deprel_dim
            )

            self.graph_convs.append(graph_conv)
            self.norms.append(norm_class(opt.hidden_dim))
            self.transformer_layers.append(
                TransformerLayer(
                    opt.hidden_dim,
                    opt.attention_heads,
                    attn_dropout_ratio=opt.attn_dropout,
                    ffn_dropout_ratio=opt.ffn_dropout,
                    norm=opt.norm
                )
            )

        self.attn = MultiHeadAttention(opt.attention_heads, opt.hidden_dim)
        self.attention_heads = opt.attention_heads

    def forward(self, inputs):
        if len(inputs) != 6:
            raise ValueError(f"Expected 6 inputs, got {len(inputs)}")

        text_bert_indices, bert_segments_ids, attention_mask, adj_dep, src_mask, aspect_mask = inputs

        token_type_vocab_size = self.bert.embeddings.token_type_embeddings.num_embeddings
        bert_segments_ids = torch.clamp(bert_segments_ids, 0, token_type_vocab_size - 1)

        device = next(self.bert.parameters()).device
        text_bert_indices = text_bert_indices.to(device)
        bert_segments_ids = bert_segments_ids.to(device)
        attention_mask = attention_mask.to(device)

        outputs = self.bert(
            input_ids=text_bert_indices,
            attention_mask=attention_mask
        )

        sequence_output, pooled_output = outputs.last_hidden_state, outputs.pooler_output

        gcn_inputs = self.bert_drop(sequence_output)
        pooled_output = self.pooled_drop(pooled_output)

        h = self.linear_in(gcn_inputs)
        B, L, d = h.size()

        aspect_outs = (h * aspect_mask.unsqueeze(-1)).sum(dim=1) / \
                      aspect_mask.sum(dim=1, keepdim=True)

        range_tensor = torch.arange(L, device=h.device).float()
        dist_mat = torch.abs(range_tensor.unsqueeze(0) - range_tensor.unsqueeze(1))
        short_bias = -dist_mat.unsqueeze(0).unsqueeze(0)
        short_bias = short_bias.expand(B, -1, -1, -1)

        weight_adj = self.attn(h, h, src_mask.unsqueeze(-2), aspect_outs, short_bias)

        e = self.deprel_emb(adj_dep) if self.deprel_emb is not None else None

        for i in range(self.opt.num_layers):
            h0 = h
            h, weight_adj = self.graph_convs[i](h, weight_adj, e, src_mask)
            h = self.norms[i](h)
            h = h.relu()
            h = F.dropout(h, self.ffn_dropout, training=self.training)

            h = self.transformer_layers[i](h, src_mask)
            h = h + h0

        aspect_words_num = aspect_mask.sum(dim=1).unsqueeze(-1)
        graph_out = (h * aspect_mask.unsqueeze(-1)).sum(dim=1) / aspect_words_num

        out = torch.cat([graph_out, pooled_output], dim=-1)
        out = self.linear_out(out)
        return out


def attention(query, key, aspect, weight_m, bias_m, mask, dropout, short):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    batch = scores.size(0)
    p = weight_m.size(0)
    max_dim = weight_m.size(1)

    weight_m = weight_m.unsqueeze(0).expand(batch, p, max_dim, max_dim)

    aspect_scores = torch.tanh(
        torch.add(torch.matmul(torch.matmul(aspect, weight_m), key.transpose(-2, -1)), bias_m)
    )
    scores = scores + aspect_scores

    if short is not None:
        scores = scores + short

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model

        self.weight_m = nn.Parameter(torch.randn(self.h, self.d_k, self.d_k))
        self.bias_m = nn.Parameter(torch.ones(1))
        self.dense = nn.Linear(d_model, self.d_k)

    def forward(self, query, key, mask, aspect, short):
        mask = mask[:, :, :query.size(1)]
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)
        query, key = [
            l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for l, x in zip(self.linears, (query, key))
        ]

        batch, aspect_dim = aspect.size(0), aspect.size(1)
        aspect = aspect.unsqueeze(1).expand(batch, self.h, aspect_dim)
        aspect = self.dense(aspect)
        aspect = aspect.unsqueeze(2).expand(batch, self.h, query.size(2), self.d_k)

        attn = attention(
            query, key, aspect, self.weight_m, self.bias_m, mask, self.dropout, short
        )
        return attn


DCE_GTN = DCE_GTN