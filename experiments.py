import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.utils import mask_to_index

from training_tools import Logger, set_seed
from graph_tools import get_local_feas, get_global_feas, get_local_feas2
from model import Biformer



def data_loader(mask, batch_size, shuffle):
    idx = mask_to_index(mask)
    loader = DataLoader(idx, batch_size=batch_size, shuffle=shuffle)

    return loader


def train(train_loader, local_feas, global_feas, global_adj, cluster, true_y, model, optimizer, device):
    model.train()

    total_loss = 0.
    for batch in train_loader:
        optimizer.zero_grad()
        x1 = global_feas
        x2 = local_feas[batch]
        cat_mask = cluster[batch]
        out = model(x1.to(device), global_adj.to(device), x2.to(device), cat_mask)
        out = torch.log_softmax(out, dim=-1)
        loss = F.nll_loss(out, true_y[batch].to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(train_loader)


@torch.no_grad()
def test(loader, local_feas, global_feas, global_adj, cluster, true_y, model, device):
    model.eval()

    total_correct, total_num = 0., 0.

    for batch in loader:
        x1 = global_feas
        x2 = local_feas[batch]
        cat_mask = cluster[batch]
        out = model(x1.to(device), global_adj.to(device), x2.to(device), cat_mask)
        pred_y = out.argmax(dim=-1)
        total_correct += int((pred_y.cpu() == true_y[batch]).sum())
        total_num += len(batch)

    acc = total_correct / total_num

    return acc


def run_experiments(data, k1, k2, k3, pool_ratio, gt_layers, lt_layers, num_heads, dropout, att_dropout, lr, weight_decay,
                    epochs, batch_size, runs, eval_steps, cuda, log=True):

    masks = [data.train_mask, data.val_mask, data.test_mask]
    [print(torch.sum(mask)) for mask in masks] if log else None

    set_seed()
    device = torch.device(f'cuda:{cuda}')

    # local_feas = get_local_feas(data, k1, k2, k3)
    local_feas = get_local_feas2(data, k1, k2, k3)

    num_parts = int(data.x.size(0) / pool_ratio)

    global_fea, global_adj, cluster = get_global_feas(data, num_parts)

    train_loader = data_loader(masks[0], batch_size, shuffle=True)
    val_loader = data_loader(masks[1], batch_size, shuffle=False)
    test_loader = data_loader(masks[2], batch_size, shuffle=False)

    model = Biformer(
        data.x.size(-1), gt_layers, lt_layers, 512, num_heads, att_dropout, 512, len(data.y.unique()), 3, dropout).to(device)

    logger = Logger(runs)
    for run in range(runs):
        print('============================================') if log else None
        model.reset_parameters()

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        for epoch in range(1, 1 + epochs):
            loss = train(train_loader, local_feas, global_fea, global_adj, cluster, data.y, model, optimizer, device)
            print(f'Run: {run + 1:02d}, Epoch: {epoch:02d}, Loss: {loss:.4f}') if log else None

            if epoch > 9 and epoch % eval_steps == 0:
                train_acc = test(train_loader, local_feas, global_fea, global_adj, cluster, data.y, model, device)
                val_acc = test(val_loader, local_feas, global_fea, global_adj, cluster, data.y, model, device)
                test_acc = test(test_loader, local_feas, global_fea, global_adj, cluster, data.y, model, device)
                print(f'Train: {train_acc*100:.2f}, Val: {val_acc*100:.2f}, Test: {test_acc*100:.2f}') if log else None
                result = train_acc, val_acc, test_acc
                logger.add_result(run, result)

        logger.print_statistics(run)
    test_acc = logger.print_statistics()
    print('============================================')

    return test_acc
