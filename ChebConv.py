import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import numpy as np
import scipy.sparse as sp


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """
    Convert a scipy sparse matrix to a torch sparse tensor.
    """
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def adj_mx_from_edges(num_pts, edges, sparse=True):
    edges = np.array(edges, dtype=np.int32)
    data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
    adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)
    adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
    adj_mx = normalize(adj_mx + sp.eye(adj_mx.shape[0]))

    if sparse:
        adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
    else:
        adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)
    return adj_mx


class ChebConv(nn.Module):
    """
    The ChebNet convolution operation.
    :param in_c: int, number of input channels. 
    :param out_c: int, number of output channels. 
    :param K: int, the order of Chebyshev Polynomial. 
    """
    def __init__(self, in_c, out_c, K, bias=True, normalize=True):
        super(ChebConv, self).__init__()
        self.normalize = normalize
        self.weight = nn.Parameter(torch.Tensor(K + 1, 1, in_c, out_c))  # [K+1, 1, in_c, out_c]
        init.xavier_normal_(self.weight)

        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_c))
            init.zeros_(self.bias)
        else:
            self.register_parameter("bias", None)

        self.K = K + 1

    def forward(self, inputs: torch.Tensor, graph) -> torch.Tensor:
        """
        :param inputs: the input data, [B, N, C], 
        :param graph: the graph structure, [N, N]
        :return: convolution result, [B, N, D]
        """
        B, C, h, w = inputs.shape
        inputs = inputs.flatten(2).transpose(1, 2)  # [B, N, C]
        graph = graph.to(inputs.device)
        L = ChebConv.get_laplacian(graph, self.normalize)  # [N, N] or [B, N, N]
        mul_L = self.cheb_polynomial(L)
        if mul_L.dim() == 3:
            result = torch.einsum('knm,bmc->kbnc', mul_L, inputs)
        else:
            result = torch.einsum('kbnm,bmc->kbnc', mul_L, inputs)

        result = torch.matmul(result, self.weight)  # [K, B, N, D]
        result = torch.sum(result, dim=0) + self.bias  # [B, N, D]

        return result

    def cheb_polynomial(self, laplacian):
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian.
        :param laplacian: the graph laplacian
        :return: the multi order Chebyshev laplacian
        """
        if laplacian.dim() == 2:
            N = laplacian.size(0)  # [N, N]
            multi_order_laplacian = torch.zeros([self.K, N, N], device=laplacian.device,
                                                dtype=laplacian.dtype)  # [K, N, N]
            multi_order_laplacian[0] = torch.eye(N, device=laplacian.device, dtype=laplacian.dtype)
        else:
            B, N, _ = laplacian.shape  # [B, N, N]
            multi_order_laplacian = torch.zeros([self.K, B, N, N], device=laplacian.device,
                                                dtype=laplacian.dtype)  # [K, B, N, N]
            multi_order_laplacian[0] = torch.eye(N, device=laplacian.device,
                                                 dtype=laplacian.dtype).unsqueeze(0)

        if self.K == 1:
            return multi_order_laplacian
        else:
            multi_order_laplacian[1] = laplacian
            if self.K == 2:
                return multi_order_laplacian
            else:
                for k in range(2, self.K):
                    multi_order_laplacian[k] = 2 * torch.matmul(laplacian, multi_order_laplacian[k-1]) - \
                                               multi_order_laplacian[k-2]

        return multi_order_laplacian

    @staticmethod
    def get_laplacian(graph, normalize):
        """
        return the laplacian of the graph.
        :param graph: the graph structure without self loop, [N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        if normalize:
            degree = torch.sum(graph, dim=-1).clamp_min(1e-6)
            if graph.dim() == 2:
                D = torch.diag(degree ** (-1 / 2))
                L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph), D)
            else:
                D = degree ** (-1 / 2)
                norm_graph = graph * D.unsqueeze(-1) * D.unsqueeze(-2)
                eye = torch.eye(graph.size(-1), device=graph.device, dtype=graph.dtype).unsqueeze(0)
                L = eye - norm_graph
        else:
            degree = torch.sum(graph, dim=-1)
            if graph.dim() == 2:
                D = torch.diag(degree)
                L = D - graph
            else:
                D = torch.diag_embed(degree)
                L = D - graph
        return L


class _GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, p_dropout=None):
        super(_GraphConv, self).__init__()

        self.gconv = ChebConv(input_dim, output_dim, K=2)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x, adj):
        x = self.gconv(x, adj)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))
        else:
            pass
        return x


class _ResChebGC(nn.Module):
    def __init__(self, input_dim, hid_dim, n_seq, p_dropout, top_k=8, temperature=0.2):
        super(_ResChebGC, self).__init__()
        self.gconv1 = _GraphConv(input_dim, hid_dim, p_dropout)
        self.top_k = top_k
        self.temperature = temperature

    def build_dynamic_adj(self, x):
        B, C, h, w = x.shape
        N = h * w
        k = min(self.top_k, N - 1)
        nodes = x.flatten(2).transpose(1, 2)  # [B, N, C]
        nodes = F.normalize(nodes, p=2, dim=-1)
        sim = torch.matmul(nodes, nodes.transpose(1, 2))  # [B, N, N]

        eye_mask = torch.eye(N, device=x.device, dtype=torch.bool).unsqueeze(0)
        sim = sim.masked_fill(eye_mask, -float('inf'))
        index = sim.topk(k=k, dim=-1).indices
        weight = torch.gather(sim, dim=-1, index=index)
        weight = torch.softmax(weight / self.temperature, dim=-1)

        adj = torch.zeros(B, N, N, device=x.device, dtype=x.dtype)
        adj.scatter_(-1, index, weight)
        adj = 0.5 * (adj + adj.transpose(1, 2))
        adj = adj + torch.eye(N, device=x.device, dtype=x.dtype).unsqueeze(0)
        return adj

    def forward(self, x):
        adj = self.build_dynamic_adj(x)
        out = self.gconv1(x, adj)
        return out
