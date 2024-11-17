import argparse
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import RandomNodeSplit, Compose, GDC
import optuna

from experiments import run_experiments



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--k1', type=int, default=5)
    parser.add_argument('--k2', type=int, default=2)
    parser.add_argument('--k3', type=int, default=4)
    parser.add_argument('--gt_layers', type=int, default=1)
    parser.add_argument('--lt_layers', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=3)
    parser.add_argument('--num_parts', type=int, default=100)

    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--att_dropout', type=float, default=0.5)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--weight_decay', type=float, default=0.000005)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=600)
    parser.add_argument('--runs', type=int, default=3)
    parser.add_argument('--eval_steps', type=int, default=1)
    parser.add_argument('--cuda', type=int, default=2)
    args = parser.parse_args()
    print(args)

    T = Compose([RandomNodeSplit('train_rest', num_val=0.2, num_test=0.2)])

    dataset = Planetoid(root='/home/zq2/data', name='Cora', transform=T)
    data = dataset[0]
    data.num_features = data.x.size(1)
    data.adj = data.edge_index.clone()
    gdc = GDC(diffusion_kwargs=dict(method='ppr', alpha=0.05), sparsification_kwargs=dict(method='topk', k=128, dim=0))
    data = gdc(data)

    run_experiments(
        data, args.k1, args.k2, args.k3, args.num_parts, args.gt_layers, args.lt_layers, args.num_heads, args.dropout,
        args.att_dropout, args.lr, args.weight_decay, args.epochs, args.batch_size, args.runs, args.eval_steps,
        args.cuda)

    print('============================================')
    print(args)


    # def objective(trial):
    #     # Define hyperparameters to tune
    #     k1 = trial.suggest_int("k1", 0, 6)
    #     k2 = trial.suggest_int("k2", 0, 6)
    #     k3 = trial.suggest_int("k3", 0, 6)
    #     gt_layers = trial.suggest_int("gt_layers", 1, 4)
    #     lt_layers = trial.suggest_int("lt_layers", 1, 4)
    #
    #     acc = run_experiments(
    #         data, k1, k2, k3, args.num_parts, gt_layers, lt_layers, args.num_heads, args.dropout, args.att_dropout,
    #         args.lr, args.weight_decay, args.epochs, args.batch_size, args.runs, args.eval_steps, args.cuda, False)
    #
    #     return acc
    #
    # study = optuna.create_study(direction="maximize")
    # study.optimize(objective, n_trials=300)
    # best_params = study.best_params
    # print("Best hyperparameters:", best_params)


if __name__ == "__main__":
    main()
