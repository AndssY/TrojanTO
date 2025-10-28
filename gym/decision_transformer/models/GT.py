import torch
from torch import nn
from torch.nn import Parameter
import torch.nn.functional as F
import math
from transformers.modeling_utils import (
    Conv1D,
)

class GraphTransformer(nn.Module):

    def __init__(self, depths, embed_dim, ff_embed_dim, num_heads, dropout, model_type, weights_dropout=True):
        super(GraphTransformer, self).__init__()
        self.layers = nn.ModuleList()
        self.model_type = model_type

        for _ in range(depths):
            self.layers.append(GraphTransformerLayer(embed_dim, ff_embed_dim, num_heads, dropout, weights_dropout))

        if "rwd" in model_type:
            self.local_global_proj = nn.Sequential(
                nn.Linear(embed_dim, 2 * embed_dim, bias=False),
            )

    def forward(self, x, relation, kv=None,
                self_padding_mask=None, self_attn_mask=None):
        """ Input shape: Time x Batch x Channel
            relation:  tgt_len x src_len x bsz x dim
            key_padding_mask: Time x batch
            attn_mask:  tgt_len x src_len
        """
        for idx, layer in enumerate(self.layers):
            x, _ = layer(x, relation, kv, self_padding_mask, self_attn_mask)

        if "rwd" in self.model_type:
            y = x[::3].transpose(0, 1)
            y = self.local_global_proj(y)
        else:
            y = torch.cat([x[1::3], x[::3]], dim=-1).transpose(0, 1)

        return x, y

    def get_attn_weights(self, x, relation, kv=None,
                         self_padding_mask=None, self_attn_mask=None):
        attns = []
        for idx, layer in enumerate(self.layers):
            x, attn = layer(x, relation, kv, self_padding_mask, self_attn_mask, need_weights=True)
            attns.append(attn)
        attn = torch.stack(attns)
        return attn


class GraphTransformerPlus(nn.Module):

    def __init__(self, depths, embed_dim, ff_embed_dim, num_heads, dropout, model_type, state_dim, max_T, weights_dropout=True):
        super(GraphTransformerPlus, self).__init__()
        self.depths = depths
        self.layers = nn.ModuleList()
        self.global_layers = nn.ModuleList()
        self.model_type = model_type

        for _ in range(depths):
            self.layers.append(GraphTransformerLayer(embed_dim, ff_embed_dim, num_heads, dropout, weights_dropout))
            self.global_layers.append(Block(N_head=num_heads, D=2 * embed_dim))

        if "rwd" in model_type:
            self.local_global_proj = nn.Sequential(
                nn.Linear(embed_dim, 2 * embed_dim),
            )
        else:
            self.local_global_proj = nn.Sequential(
                nn.LayerNorm(embed_dim * 2),
                nn.Linear(2 * embed_dim, 2 * embed_dim)
            )

        self.embed_state = torch.nn.Linear(state_dim, 2 * embed_dim)
        self.temporal_embed = nn.Parameter(torch.zeros(1, max_T, 2 * embed_dim))

    def forward(self, x, relation, state, kv=None,
                self_padding_mask=None, self_attn_mask=None):
        """
            x: tgt_len x bsz x dim
            relation:  tgt_len x src_len x bsz x dim
            key_padding_mask: Time x batch
            attn_mask:  tgt_len x src_len
            state_embedding: Batch x Time x dim
        """
        if "rwd" in self.model_type:
            state_embedding = x[::3].transpose(0, 1)
        else:
            state_embedding = x[1::3].transpose(0, 1)
        bs, seq_length, dim = state_embedding.size()
        device = x.device

        mask = torch.tril(torch.ones(seq_length * 2, seq_length * 2)).to(device=device)
        for i in range(0, seq_length * 2, 2):
            mask[i, i + 1] = 1
        mask = mask.unsqueeze(0).unsqueeze(0)

        token_embeddings = torch.zeros((bs, seq_length * 2, 2 * dim), dtype=torch.float32, device=device)
        token_embeddings[:, 0::2] = self.embed_state(state) + self.temporal_embed[:, :seq_length]

        for idx in range(self.depths):
            x, _ = self.layers[idx](x, relation, kv, self_padding_mask, self_attn_mask)

            if "rwd" in self.model_type:
                y = x[::3].transpose(0, 1)
                y = self.local_global_proj(y)
            else:
                y = torch.cat([x[1::3], x[::3]], dim=-1).transpose(0, 1)
                # y = x[1::3].transpose(0, 1)
                y = self.local_global_proj(y)

            token_embeddings[:, 1::2] = token_embeddings[:, 1::2] + y
            token_embeddings, _ = self.global_layers[idx](token_embeddings, mask=mask)

        return x, token_embeddings[:, 1::2]


class GraphTransformerLayer(nn.Module):

    def __init__(self, embed_dim, ff_embed_dim, num_heads, dropout, weights_dropout=True):
        super(GraphTransformerLayer, self).__init__()
        self.self_attn = RelationMultiheadAttention(embed_dim, num_heads, dropout, weights_dropout)
        self.fc1 = Conv1D(ff_embed_dim, embed_dim)
        self.fc2 = Conv1D(embed_dim, ff_embed_dim)
        self.attn_layer_norm = nn.LayerNorm(embed_dim)
        self.ff_layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.02)
        nn.init.constant_(self.fc1.bias, 0.)
        nn.init.constant_(self.fc2.bias, 0.)

    def forward(self, x, relation, kv=None,
                self_padding_mask=None, self_attn_mask=None,
                need_weights=False):
        # x: seq_len x bsz x embed_dim
        residual = x
        if kv is None:
            x, self_attn = self.self_attn(query=x, key=x, value=x, relation=relation,
                                          key_padding_mask=self_padding_mask, attn_mask=self_attn_mask,
                                          need_weights=need_weights)
        else:
            x, self_attn = self.self_attn(query=x, key=kv, value=kv, relation=relation,
                                          key_padding_mask=self_padding_mask, attn_mask=self_attn_mask,
                                          need_weights=need_weights)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.attn_layer_norm(residual + x)

        residual = x
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.ff_layer_norm(residual + x)
        return x, self_attn


class RelationMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., weights_dropout=True):
        super(RelationMultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.in_proj_weight = Parameter(torch.Tensor(3 * embed_dim, embed_dim))
        self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        self.relation_in_proj = Conv1D(2 * embed_dim, embed_dim)

        self.out_proj = Conv1D(embed_dim, embed_dim)
        self.weights_dropout = weights_dropout
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.in_proj_weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.normal_(self.relation_in_proj.weight, std=0.02)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, relation, key_padding_mask=None, attn_mask=None, need_weights=False):
        """ Input shape: Time x Batch x Channel
            relation:  tgt_len x src_len x bsz x dim
            key_padding_mask: Time x batch
            attn_mask:  tgt_len x src_len
        """
        qkv_same = query.data_ptr() == key.data_ptr() == value.data_ptr()
        kv_same = key.data_ptr() == value.data_ptr()

        tgt_len, bsz, embed_dim = query.size()
        src_len = key.size(0)
        assert key.size() == value.size()

        if qkv_same:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif kv_same:
            # encoder-decoder attention
            q = self.in_proj_q(query)
            k, v = self.in_proj_kv(key)
        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim)
        k = k.contiguous().view(src_len, bsz * self.num_heads, self.head_dim)
        v = v.contiguous().view(src_len, bsz * self.num_heads, self.head_dim)

        ra, rb = self.relation_in_proj(relation).chunk(2, dim=-1)
        ra = ra.contiguous().view(tgt_len, src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        rb = rb.contiguous().view(tgt_len, src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        q = q.unsqueeze(1) + ra
        k = k.unsqueeze(0) + rb
        q *= self.scaling
        # q: tgt_len x src_len x bsz*heads x dim
        # k: tgt_len x src_len x bsz*heads x dim
        # v: src_len x bsz*heads x dim

        attn_weights = torch.einsum('ijbn,ijbn->ijb', [q, k]) * (1.0 / math.sqrt(k.size(-1)))
        assert list(attn_weights.size()) == [tgt_len, src_len, bsz * self.num_heads]

        if attn_mask is not None:
            attn_weights.masked_fill_(
                attn_mask.unsqueeze(-1),
                float('-inf')
            )

        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(tgt_len, src_len, bsz, self.num_heads)
            attn_weights.masked_fill_(
                key_padding_mask.unsqueeze(0).unsqueeze(-1),
                float('-inf')
            )
            attn_weights = attn_weights.view(tgt_len, src_len, bsz * self.num_heads)

        attn_weights = F.softmax(attn_weights, dim=1)

        if self.weights_dropout:
            attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        # attn_weights: tgt_len x src_len x bsz*heads
        # v: src_len x bsz*heads x dim
        attn = torch.einsum('ijb,jbn->bin', [attn_weights, v])
        if not self.weights_dropout:
            attn = F.dropout(attn, p=self.dropout, training=self.training)

        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]

        attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            # maximum attention weight over heads
            attn_weights = attn_weights.view(tgt_len, src_len, bsz, self.num_heads)
        else:
            attn_weights = None

        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_kv(self, key):
        return self._in_proj(key, start=self.embed_dim).chunk(2, dim=-1)

    def in_proj_q(self, query):
        return self._in_proj(query, end=self.embed_dim)

    def in_proj_k(self, key):
        return self._in_proj(key, start=self.embed_dim, end=2 * self.embed_dim)

    def in_proj_v(self, value):
        return self._in_proj(value, start=2 * self.embed_dim)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

        return output


class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return F.gelu(input)


class CausalSelfAttention(nn.Module):
    def __init__(self, N_head=6, D=128, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        assert D % N_head == 0

        self.N_head = N_head

        # self.key = nn.Linear(D, D)
        # self.query = nn.Linear(D, D)
        # self.value = nn.Linear(D, D)
        self.key = Conv1D(D, D)
        self.query = Conv1D(D, D)
        self.value = Conv1D(D, D)

        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resd_drop = nn.Dropout(resid_pdrop)

        self.proj = Conv1D(D, D)

    def forward(self, x, mask=None):
        # x: B * N * D
        B, N, D = x.size()

        q = self.query(x.view(B * N, -1)).view(B, N, self.N_head, D // self.N_head).transpose(1, 2)
        k = self.key(x.view(B * N, -1)).view(B, N, self.N_head, D // self.N_head).transpose(1, 2)
        v = self.value(x.view(B * N, -1)).view(B, N, self.N_head, D // self.N_head).transpose(1, 2)

        A = q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1)))
        if mask is not None:
            A = A.masked_fill(mask[:, :, :N, :N] == 0, float('-inf'))
        A = F.softmax(A, dim=-1)
        A_drop = self.attn_drop(A)
        y = (A_drop @ v).transpose(1, 2).contiguous().view(B, N, D)
        y = self.resd_drop(self.proj(y))
        return y, A


class Block(nn.Module):
    def __init__(self, N_head, D, attn_pdrop=0.1, resid_pdrop=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(D)
        self.ln2 = nn.LayerNorm(D)
        self.attn = CausalSelfAttention(N_head=N_head, D=D, attn_pdrop=attn_pdrop, resid_pdrop=resid_pdrop)
        # self.mlp = nn.Sequential(
        #     nn.Linear(D, 4 * D),
        #     GELU(),
        #     nn.Linear(4 * D, D),
        #     nn.Dropout(resid_pdrop),
        # )
        self.mlp = nn.Sequential(
            Conv1D(4 * D, D),
            GELU(),
            Conv1D(D, 4 * D),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x, mask=None):
        y, att = self.attn(self.ln1(x), mask=mask)
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x, att