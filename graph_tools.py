import numpy as np
import torch
import torch_geometric.typing as pyg_typing
from torch_sparse import SparseTensor
from torch_geometric.utils import sort_edge_index, remove_self_loops, coalesce
from torch_geometric.utils.sparse import index2ptr



def GCN_kernel(adj, self_loops=True):
    adj = SparseTensor(row=adj[0], col=adj[1])
    adj = adj.set_diag() if self_loops else adj

    deg = adj.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    adj_sym = deg_inv_sqrt.view(-1, 1) * adj * deg_inv_sqrt.view(1, -1)
    adj_sym = adj_sym.to_scipy(layout='csr')

    return adj_sym


def PPR_kernel(edge_index, edge_attr):
    ppr = SparseTensor(row=edge_index[0], col=edge_index[1], value=edge_attr)
    ppr = ppr.to_scipy(layout='csr')

    return ppr


def metis_divide(edge_index, num_nodes: int, num_parts: int, recursive: bool=False):
    # set recursive to True if num_parts < 7
    print('Computing METIS partitioning...')
    row, col = sort_edge_index(edge_index, num_nodes=num_nodes)
    rowptr = index2ptr(row, size=num_nodes)
    cluster = None

    if pyg_typing.WITH_TORCH_SPARSE:
        try:
            cluster = torch.ops.torch_sparse.partition(
                rowptr.cpu(), col.cpu(), None, num_parts, recursive).to(edge_index.device)
        except (AttributeError, RuntimeError):
            pass

    if cluster is None and pyg_typing.WITH_METIS:
        cluster = pyg_typing.pyg_lib.partition.metis(
            rowptr.cpu(), col.cpu(), num_parts, recursive=recursive).to(edge_index.device)

    cluster = cluster.numpy()

    divide_list = [((np.where(cluster == i))[0]).tolist() for i in range(num_parts)]

    print('Done!')

    return divide_list, torch.from_numpy(cluster)


def get_local_feas(data, k1, k2, k3):
    gcn_kernel = GCN_kernel(data.adj)
    ppr_kernel = PPR_kernel(data.edge_index, data.edge_attr)

    local_feas = [data.x]
    x1, x2, x3 = data.x, data.x, data.x

    for _ in range(k1):
        x1 = torch.from_numpy(gcn_kernel @ x1)
        local_feas.append(x1)

    for _ in range(k2):
        x2 = 0.5 * x2 - torch.from_numpy(gcn_kernel @ x2)
        local_feas.append(x2)

    for _ in range(k3):
        x3 = torch.from_numpy(ppr_kernel @ x3)
        local_feas.append(x3)

    local_feas = torch.cat(local_feas, dim=1)
    local_feas = local_feas.view(data.x.size(0), 1 + k1 + k2 + k3, -1)

    return local_feas


def get_global_feas(data, num_parts):
    divide_list, cluster = metis_divide(data.adj, data.x.size(0), num_parts)

    global_fea = torch.cat([data.x[batch].mean(dim=0, keepdim=True) for batch in divide_list], dim=0)

    global_adj = cluster[data.adj.view(-1)].view(2, -1)
    edge_index2, _ = remove_self_loops(global_adj)
    global_adj = coalesce(global_adj)
    global_adj = SparseTensor(row=global_adj[0], col=global_adj[1]).to_dense()

    return global_fea, global_adj, cluster


def get_local_feas2(data, k1, k2, k3, alpha=0.2):
    gcn_kernel = GCN_kernel(data.adj)

    local_feas = [data.x]
    x1, x2, x3 = data.x, data.x, data.x

    for _ in range(k1):
        x1 = torch.from_numpy(gcn_kernel @ x1)
        local_feas.append(x1)

    for _ in range(k2):
        x2 = 0.5 * x2 - torch.from_numpy(gcn_kernel @ x2)
        local_feas.append(x2)

    for _ in range(k3):
        x3 = (1 - alpha) * torch.from_numpy(gcn_kernel @ x3) + alpha * data.x
        local_feas.append(x3)

    local_feas = torch.cat(local_feas, dim=1)
    local_feas = local_feas.view(data.x.size(0), 1 + k1 + k2 + k3, -1)

    return local_feas
