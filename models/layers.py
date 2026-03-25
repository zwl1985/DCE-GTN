import math 
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from typing import Any 
from torch import Tensor 
from torch_scatter import scatter

class MHCABlock(nn.Module):
    def __init__(self, hidden_dim, attention_heads, attn_dropout_ratio, ffn_dropout_ratio, norm='ln'): 
        super().__init__() 
        self.hidden_dim = hidden_dim 
        self.attention_heads = attention_heads 
        self.attn = nn.MultiheadAttention(hidden_dim, attention_heads, dropout=attn_dropout_ratio, batch_first=True)
        self.ffn = nn.Sequential( 
            nn.Linear(hidden_dim, 2*hidden_dim), 
            nn.ReLU(), 
            nn.Dropout(ffn_dropout_ratio), 
            nn.Linear(2*hidden_dim, hidden_dim), 
            nn.Dropout(ffn_dropout_ratio) 
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def forward(self, q, v, k, src_mask):
        q_res = q
        q, _ = self.attn(q, k, v, src_mask)
        q = self.norm1(q + q_res)
        out = self.ffn(q) + q
        out = self.norm2(out)
        return out

class TransformerLayer(nn.Module): 
    def __init__(self, hidden_dim, attention_heads, attn_dropout_ratio, ffn_dropout_ratio, norm='ln'): 
        super().__init__() 
        self.hidden_dim = hidden_dim 
        self.attention_heads = attention_heads 
        self.attn_dropout = nn.Dropout(attn_dropout_ratio) 
        self.linear_q = nn.Linear(hidden_dim, hidden_dim) 
        self.linear_k = nn.Linear(hidden_dim, hidden_dim) 
        self.linear_v = nn.Linear(hidden_dim, hidden_dim) 
        self.linear_attn_out = nn.Sequential( 
            nn.Linear(hidden_dim, hidden_dim), 
            nn.Dropout(ffn_dropout_ratio) 
            ) 
        norm_class = None 
        if norm == 'ln': 
            norm_class = nn.LayerNorm 
        elif norm == 'bn': 
            norm_class = nn.BatchNorm1d 
        self.norm1 = norm_class(hidden_dim) 
        self.norm2 = norm_class(hidden_dim) 
        self.ffn = nn.Sequential( 
            nn.Linear(hidden_dim, 2*hidden_dim), 
            nn.ReLU(), 
            nn.Dropout(ffn_dropout_ratio), 
            nn.Linear(2*hidden_dim, hidden_dim), 
            nn.Dropout(ffn_dropout_ratio) 
        ) 
    
    def forward(self, x, src_mask):
        q = self.linear_q(x) 
        k = self.linear_k(x) 
        v = self.linear_v(x) 
        dim_split = self.hidden_dim // self.attention_heads 
        q_heads = torch.cat(q.split(dim_split, 2), dim=0) 
        k_heads = torch.cat(k.split(dim_split, 2), dim=0) 
        v_heads = torch.cat(v.split(dim_split, 2), dim=0) 
        attention_score = q_heads.bmm(k_heads.transpose(1, 2)) 
        attention_score = attention_score / math.sqrt(self.hidden_dim // self.attention_heads) 
        inf_mask = (~src_mask).unsqueeze(1).to(dtype=torch.float) * -1e9
        inf_mask = torch.cat([inf_mask for _ in range(self.attention_heads)], 0) 
        A = torch.softmax(attention_score + inf_mask, -1) 
        A = self.attn_dropout(A) 
        attn_out = torch.cat((A.bmm(v_heads)).split(q.size(0), 0), 2) 
        attn_out = self.linear_attn_out(attn_out) 
        attn_out = attn_out + x 
        attn_out = self.norm1(attn_out) 
        out = self.ffn(attn_out) + attn_out 
        out = self.norm2(out) 
        return out 

class TransformerLayer_v(nn.Module): 
    def __init__(self, hidden_dim, attention_heads, attn_dropout_ratio, ffn_dropout_ratio, norm='ln'): 
        super().__init__() 
        self.hidden_dim = hidden_dim 
        self.attention_heads = attention_heads 
        self.attn_dropout = nn.Dropout(attn_dropout_ratio) 
        self.linear_q = nn.Linear(hidden_dim, hidden_dim) 
        self.linear_k = nn.Linear(hidden_dim, hidden_dim) 
        self.linear_v = nn.Linear(hidden_dim, hidden_dim) 
        self.linear_attn_out = nn.Sequential( 
            nn.Linear(hidden_dim, hidden_dim), 
            nn.Dropout(ffn_dropout_ratio) 
            ) 
        norm_class = None 
        if norm == 'ln': 
            norm_class = nn.LayerNorm 
        elif norm == 'bn': 
            norm_class = nn.BatchNorm1d 
        self.norm1 = norm_class(hidden_dim) 
        self.norm2 = norm_class(hidden_dim) 
        self.ffn = nn.Sequential( 
            nn.Linear(hidden_dim, 2*hidden_dim), 
            nn.ReLU(), 
            nn.Dropout(ffn_dropout_ratio), 
            nn.Linear(2*hidden_dim, hidden_dim), 
            nn.Dropout(ffn_dropout_ratio) 
        ) 
    
    def forward(self, x, src_mask):
        q = self.linear_q(x) 
        k = self.linear_k(x) 
        v = self.linear_v(x) 
        dim_split = self.hidden_dim // self.attention_heads 
        q_heads = torch.cat(q.split(dim_split, 2), dim=0) 
        k_heads = torch.cat(k.split(dim_split, 2), dim=0) 
        v_heads = torch.cat(v.split(dim_split, 2), dim=0) 
        attention_score = q_heads.bmm(k_heads.transpose(1, 2)) 
        attention_score = attention_score / math.sqrt(self.hidden_dim // self.attention_heads) 
        inf_mask = (~src_mask).unsqueeze(1).to(dtype=torch.float) * -1e9
        inf_mask = torch.cat([inf_mask for _ in range(self.attention_heads)], 0) 
        A = torch.softmax(attention_score + inf_mask, -1) 
        words_attn = A.clone().detach()
        A = self.attn_dropout(A) 
        attn_out = torch.cat((A.bmm(v_heads)).split(q.size(0), 0), 2) 
        attn_out = self.linear_attn_out(attn_out) 
        attn_out = attn_out + x 
        attn_out = self.norm1(attn_out) 
        out = self.ffn(attn_out) + attn_out 
        out = self.norm2(out) 
        return out, words_attn 

class TextGATConv_mod(nn.Module):
    def __init__(
        self, 
        hidden_dim,
        num_heads: int = 1,
        negative_slope: float = 0.2,
        attn_dropout: float = 0.0,
        ffn_dropout: float = 0.0, 
        edge_dim: int = None, 
        norm='ln'): 

        super().__init__() 
        self.hidden_dim = hidden_dim 
        self.num_heads = num_heads 
        self.negative_slope = negative_slope 
        self.attn_dropout = attn_dropout 
        self.ffn_dropout = ffn_dropout 
        self.edge_dim = edge_dim 
        self.linear_q = nn.Linear(hidden_dim, hidden_dim, bias=False) 
        self.linear_kv = nn.Linear(hidden_dim, hidden_dim, bias=False) 
        self.linear_e = nn.Linear(edge_dim, hidden_dim, bias=False) 
        norm_class = None 
        if norm == 'ln': 
            norm_class = nn.LayerNorm
        elif norm == 'bn': 
            norm_class = nn.BatchNorm1d 
        self.ffn = nn.Sequential( 
            nn.Linear(hidden_dim, hidden_dim), 
            norm_class(hidden_dim), 
            nn.ReLU(), 
            nn.Dropout(ffn_dropout) 
        ) 
    
    def attention_matrix(self, q, kv, adj): 
        C = q.size(-1) 
        alpha = (q * kv).sum(dim=-1) / math.sqrt(C)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = alpha.permute(0, 2, 3, 1)
        inf = torch.ones_like(alpha) * -1e9 
        alpha = torch.where((adj != 0).unsqueeze(-1), alpha, inf)
        alpha = alpha.permute(0, 3, 1, 2)
        alpha = torch.softmax(alpha, dim=-1) 
        alpha = F.dropout(alpha, p=self.attn_dropout, training=self.training) 
        return alpha 

    def forward(self, x, adj, e):
        B, L = x.size(0), x.size(1) 
        H, C = self.num_heads, self.hidden_dim // self.num_heads 
        q = self.linear_q(x).view(B, L, H, C).transpose(1, 2)
        kv = self.linear_kv(x).view(B, L, H, C).transpose(1, 2)
        e = self.linear_e(e).view(B, L, L, H, C).permute(0, 3, 1, 2, 4)
        q = q.unsqueeze(3)
        kv = kv.unsqueeze(2)
        kv = kv + e
        alpha = self.attention_matrix(q, kv, adj)
        alpha = alpha.unsqueeze(-1)
        out = (alpha * kv).sum(dim=-2)
        out = out.permute(0, 2, 1, 3)
        out = out.contiguous().view(B, L, H * C)
        out = self.ffn(out) 
        return out