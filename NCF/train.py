import numpy as np
import pandas as pd
import torch

from dataset import NCFDataset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score


def train_test_split_last(u, v):
    df = pd.DataFrame({'u': u, 'v': v})
    test_idx = [idx.index[-1] for _, idx in df.groupby('u')]
    train, test = df.drop(test_idx), df.loc[test_idx]
    return train, test


def train_test_split_random(u, v, ratio=0.2, random_seed=0):
    import random
    df = pd.DataFrame({'u': u, 'v': v})
    test_idx = []
    for _, idx in df.groupby('u'):
        sample_len = int(np.ceil(len(idx) * ratio))
        test_idx += random.sample(list(idx.index), sample_len)
    train, test = df.drop(test_idx), df.loc[test_idx]
    return train, test


def train(model, loss_fn, optimizer, train_u, train_v, neg_sample, epochs, batch_size, device, test_loader=None):
    for epoch in range(epochs):
        train_set = NCFDataset(train_u, train_v, neg_sample)
        train_loader = DataLoader(train_set, batch_size=1024, shuffle=True)
        
        model.train()
        total_loss = 0
        for (u, v), target in train_loader:
            u, v, target = u.to(device), v.to(device), target.to(device)
            pred = model(u.long(), v.long())
            loss = loss_fn(pred, target.view(-1, 1).to(torch.float32))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(train_loader)
        if test_loader is not None:
            test_loss = evaluate_loss(model, loss_fn, test_loader, device=device)
            print(f"Epoch {epoch}: train_loss={train_loss:.6f} test_loss={test_loss:.6f}")
        else:
            print(f"Epoch {epoch}: train_loss={train_loss:.6f}")

            
def evaluate_loss(model, loss_fn, test_loader, device='cpu'):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for (u, v), target in test_loader:
            u, v, target = u.to(device), v.to(device), target.to(device)
            pred = model(u.long(), v.long())
            loss = loss_fn(pred, target.view(-1, 1).to(torch.float32))
            test_loss += loss.item()
    return test_loss / len(test_loader)


def evaluate_auc(model, loss_fn, data, device='cpu'):
    from scipy.sparse import csr_matrix
    
    unique_row = np.sort(data['row'].unique())
    unique_col = np.sort(data['col'].unique())

    rows = data['row'].astype('category').cat.codes
    cols = data['col'].astype('category').cat.codes

    test_matrix = csr_matrix((data['rating'], (rows, cols)), shape=(len(unique_row), len(unique_col)))
    ncf_auc = []
    model.eval()
    with torch.no_grad():
        v = torch.tensor(range(len(unique_col))).to(device)
        for i in range(len(unique_row)):
            target = test_matrix[i].toarray()

            u = torch.tensor([i] * len(unique_col)).to(device)
            pred = model(u, v).cpu().detach().numpy()

            ncf_auc.append(roc_auc_score(target.reshape(-1), pred.reshape(-1)))
    return np.mean(ncf_auc)
