import torch
from torch_geometric.nn import Linear



class MLP(torch.nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_layers, dropout, act='relu'):
        super(MLP, self).__init__()

        self.model = torch.nn.ModuleList()
        self.model.append(Linear(in_dim, hid_dim))

        for _ in range(num_layers - 2):
            self.model.append(Linear(hid_dim, hid_dim))

        self.model.append(Linear(hid_dim, out_dim))
        self.dropout = torch.nn.Dropout(dropout)

        if act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'prelu':
            self.act = torch.nn.PReLU()
        elif act == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def reset_parameters(self):
        for layer in self.model:
            layer.reset_parameters()

    def forward(self, x):
        for layer in self.model[:-1]:
            x = self.dropout(self.act(layer(x)))

        x = self.model[-1](x)

        return x


class FFN(torch.nn.Module):
    def __init__(self, in_dim, hid_dim):
        super(FFN, self).__init__()

        self.layer1 = torch.nn.Linear(in_dim, hid_dim)
        self.gelu = torch.nn.GELU()
        self.layer2 = torch.nn.Linear(hid_dim, in_dim)

    def reset_parameters(self):
        self.layer1.reset_parameters()
        self.layer2.reset_parameters()

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)

        return x


class GT_SA(torch.nn.Module):
    def __init__(self, in_dim, att_dim, out_dim, num_heads, att_dropout, att_bias=0.5):
        super(GT_SA, self).__init__()

        self.Q = Linear(in_dim, num_heads * att_dim)
        self.K = Linear(in_dim, num_heads * att_dim)
        self.V = Linear(in_dim, num_heads * out_dim)
        self.output_layer = Linear(num_heads * out_dim, out_dim)

        self.act = torch.nn.Sigmoid()
        # self.act = torch.nn.Tanh()
        self.att_dropout = torch.nn.Dropout(att_dropout)

        self.num_heads = num_heads
        self.att_dim = att_dim
        self.out_dim = out_dim
        self.att_bias = att_bias

    def reset_parameters(self):
        self.Q.reset_parameters()
        self.K.reset_parameters()
        self.V.reset_parameters()
        self.output_layer.reset_parameters()

    def att_scores(self, Q, K, adj):  # input_size: [num_fea, in_dim]
        Q = self.Q(Q).view(-1, self.num_heads, self.att_dim)  # [num_fea, num_heads, d_k]
        K = self.K(K).view(-1, self.num_heads, self.att_dim)  # [num_fea, num_heads, d_k]

        Q = Q.permute(1, 0, 2).contiguous()  # [num_heads, num_fea, d_k]
        K = K.permute(1, 2, 0).contiguous()  # [num_heads, d_k, num_fea]

        # softmax((Q * K^T) / sqrt(d_k))
        scores = torch.matmul(Q * (self.att_dim ** -0.5), K)  # [num_heads, num_fea, num_fea]
        scores = self.act(scores)
        scores = scores - self.att_bias

        adj = torch.cat([adj.unsqueeze(0) for _ in range(self.num_heads)], dim=0)
        scores = torch.softmax(scores + 0.1*adj, dim=2)  # [num_heads, num_fea, num_fea]

        return scores

    def forward(self, Q, K, V, adj):  # input_size: [num_fea, in_dim]
        scores = self.att_scores(Q, K, adj)  # [num_heads, num_fea, num_fea]

        V = self.V(V).view(-1, self.num_heads, self.out_dim)  # [num_fea, num_heads, d_v]
        V = V.permute(1, 0, 2).contiguous()  # [num_heads, num_fea, d_v]

        out = torch.matmul(self.att_dropout(scores), V)  # [num_heads, num_fea, d_v]
        out = out.permute(1, 0, 2).contiguous()  # [num_fea, num_heads, d_v]
        out = out.view(-1, self.num_heads * self.out_dim)  # [num_fea, num_heads * d_v]
        out = self.output_layer(out)  # [num_fea, d_v]

        return out


class GT_layer(torch.nn.Module):
    def __init__(self, in_dim, att_dim, num_heads, att_dropout, dropout):
        super(GT_layer, self).__init__()

        self.att_norm = torch.nn.LayerNorm(in_dim)
        self.att_model = GT_SA(in_dim, att_dim, in_dim, num_heads, att_dropout)

        self.ffn_norm = torch.nn.LayerNorm(in_dim)
        self.ffn = FFN(in_dim, in_dim)

        self.dropout = torch.nn.Dropout(dropout)

    def reset_parameters(self):
        self.att_norm.reset_parameters()
        self.att_model.reset_parameters()
        self.ffn_norm.reset_parameters()
        self.ffn.reset_parameters()

    def forward(self, global_fea, global_adj):
        att_x = self.att_norm(global_fea)
        att_x = self.att_model(att_x, att_x, att_x, global_adj)
        att_x = self.dropout(att_x)
        global_fea = global_fea + att_x

        ffn_x = self.ffn_norm(global_fea)
        ffn_x = self.ffn(ffn_x)
        ffn_x = self.dropout(ffn_x)
        out = global_fea + ffn_x

        return out


class FT_SA(torch.nn.Module):
    def __init__(self, in_dim, att_dim, out_dim, num_heads, att_dropout, att_bias=0.5):
        super(FT_SA, self).__init__()

        self.Q = Linear(in_dim, num_heads * att_dim)
        self.K = Linear(in_dim, num_heads * att_dim)
        self.V = Linear(in_dim, num_heads * out_dim)
        self.output_layer = Linear(num_heads * out_dim, out_dim)

        self.act = torch.nn.Sigmoid()
        # self.act = torch.nn.Tanh()
        self.att_dropout = torch.nn.Dropout(att_dropout)

        self.num_heads = num_heads
        self.att_dim = att_dim
        self.out_dim = out_dim
        self.att_bias = att_bias

    def reset_parameters(self):
        self.Q.reset_parameters()
        self.K.reset_parameters()
        self.V.reset_parameters()
        self.output_layer.reset_parameters()

    def att_scores(self, batch_size, batched_q, batched_k):
        # input_size: [b, num_fea, in_dim]

        batched_q = self.Q(batched_q).view(batch_size, -1, self.num_heads, self.att_dim)
        # [b, num_fea, num_heads, d_k]
        batched_k = self.K(batched_k).view(batch_size, -1, self.num_heads, self.att_dim)
        # [b, num_fea, num_heads, d_k]

        batched_q = batched_q.permute(0, 2, 1, 3).contiguous()
        # [b, num_heads, num_fea, d_k]
        batched_k = batched_k.permute(0, 2, 3, 1).contiguous()
        # [b, num_heads, d_k, num_fea]

        # softmax((Q * K^T) / sqrt(d_k))
        scores = torch.matmul(batched_q * (self.att_dim ** -0.5), batched_k)  # [b, num_heads, num_fea, num_fea]
        scores = self.act(scores)
        scores = scores - self.att_bias
        scores = torch.softmax(scores, dim=3)  # [b, num_heads, num_fea, num_fea]

        # torch.save(scores, 'scores_.pt')

        return scores

    def forward(self, batched_q, batched_k, batched_v):
        # input_size: [b, num_fea, in_dim]
        batch_size = batched_q.size(0)

        scores = self.att_scores(batch_size, batched_q, batched_k)  # [b, num_heads, num_fea, num_fea]

        batched_v = self.V(batched_v).view(batch_size, -1, self.num_heads, self.out_dim)
        # [b, num_fea, num_heads, d_v]
        batched_v = batched_v.permute(0, 2, 1, 3).contiguous()
        # [b, num_heads, num_fea, d_v]

        out = torch.matmul(self.att_dropout(scores), batched_v)  # [b, num_heads, num_fea, d_v]
        out = out.permute(0, 2, 1, 3).contiguous()  # [b, num_fea, num_heads, d_v]
        out = out.view(batch_size, -1, self.num_heads * self.out_dim)  # [b, num_fea, num_heads * d_v]
        out = self.output_layer(out)  # [b, num_fea, out_dim]

        return out


class FT_layer(torch.nn.Module):
    def __init__(self, in_dim, att_dim, num_heads, att_dropout, dropout):
        super(FT_layer, self).__init__()

        self.att_norm = torch.nn.LayerNorm(in_dim)
        self.att_model = FT_SA(in_dim, att_dim, in_dim, num_heads, att_dropout)

        self.ffn_norm = torch.nn.LayerNorm(in_dim)
        self.ffn = FFN(in_dim, in_dim)

        self.dropout = torch.nn.Dropout(dropout)

    def reset_parameters(self):
        self.att_norm.reset_parameters()
        self.att_model.reset_parameters()
        self.ffn_norm.reset_parameters()
        self.ffn.reset_parameters()

    def forward(self, batched_x):
        att_x = self.att_norm(batched_x)
        att_x = self.att_model(att_x, att_x, att_x)
        att_x = self.dropout(att_x)
        batched_x = batched_x + att_x

        ffn_x = self.ffn_norm(batched_x)
        ffn_x = self.ffn(ffn_x)
        ffn_x = self.dropout(ffn_x)
        out = batched_x + ffn_x

        return out


class Biformer(torch.nn.Module):
    def __init__(self, fea_dim, gt_layers, lt_layers, att_dim, num_heads, att_dropout, hid_dim, out_dim, pred_layers, dropout):
        super(Biformer, self).__init__()

        self.gt = torch.nn.ModuleList(
            [GT_layer(fea_dim, att_dim, num_heads, att_dropout, dropout) for _ in range(gt_layers)])

        self.ft = torch.nn.ModuleList(
            [FT_layer(fea_dim, att_dim, num_heads, att_dropout, dropout) for _ in range(lt_layers)])

        self.pred = MLP(fea_dim, hid_dim, out_dim, pred_layers, dropout)

    def reset_parameters(self):
        self.pred.reset_parameters()
        for layer in self.gt:
            layer.reset_parameters()
        for layer in self.ft:
            layer.reset_parameters()

    def forward(self, global_fea, global_adj, batched_local_feas, cat_mask):
        for layer in self.gt:
            global_fea = layer(global_fea, global_adj)

        x = torch.cat([global_fea[cat_mask].unsqueeze(1), batched_local_feas], dim=1)
        for layer in self.ft:
            x = layer(x)

        att_out = x.sum(dim=1)
        out = self.pred(att_out)

        return out
