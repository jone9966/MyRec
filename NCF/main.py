import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import argparse

from dataset import NCFDataset
from model import NCF
from train import *


def arg_parser():
    parser = argparse.ArgumentParser(description="Train and evaluate NCF model on data")
    parser.add_argument("data_path", type=str, help="path to the data file")
    parser.add_argument("--k_gmf", type=int, default=20, help="number of latent factor for gmf")
    parser.add_argument("--k_mlp", type=int, default=20, help="number of latent factor for mlp")
    parser.add_argument("--negative_sample", type=int, default=5, help="number of negative samples")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate for optimizer")
    parser.add_argument("--batch_size", type=int, default=1024, help="batch size for training")
    parser.add_argument("--num_epochs", type=int, default=50, help="number of epochs for training")
    parser.add_argument("--test_ratio", type=float, default=0.2, help="ratio of data to use for testing")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="device to use for training and evaluation")
    return parser.parse_args()


if __name__ == '__main__':
    args = arg_parser()

    # Load data
    data = pd.read_csv(args.data_path)
    unique_row = np.sort(data['row'].unique())
    unique_col = np.sort(data['col'].unique())

    rows = data['row'].astype('category').cat.codes
    cols = data['col'].astype('category').cat.codes

    # Split data into train and test sets
    train_df, _ = train_test_split_random(rows.values, cols.values, ratio=args.test_ratio)
    train_u, train_v = train_df['u'].values, train_df['v'].values

    # Initialize NCF model, loss function, and optimizer
    model = NCF(num_user=len(unique_row), num_item=len(unique_col), k_gmf=args.k_gmf, k_mlp=args.k_mlp).to(args.device)
    loss_fn = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train model
    test_set = NCFDataset(rows.values, cols.values, args.negative_sample)
    test_loader = DataLoader(test_set, batch_size=args.batch_size)

    train(model=model, loss_fn=loss_fn, optimizer=optimizer,
          train_u=train_u, train_v=train_v, neg_sample=args.negative_sample,
          epochs=args.num_epochs, batch_size=args.batch_size, device=args.device, 
          test_loader=test_loader)

    print(f"auc_score={evaluate_auc(model, loss_fn, data, args.device):.3f}")
