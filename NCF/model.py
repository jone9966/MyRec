import torch
import torch.nn as nn


class GMF(nn.Module):
    def __init__(self, num_user, num_item, k):
        super().__init__()
        self.u_embedding = nn.Embedding(num_user, k)
        self.v_embedding = nn.Embedding(num_item, k)
        self.h = nn.Linear(k, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, u, v):
        element_wise = self.u_embedding(u) * self.v_embedding(v)
        output = self.h(element_wise)
        output = self.sigmoid(output)
        return output


class MLP(nn.Module):
    def __init__(self, num_user, num_item, k):
        super().__init__()
        self.u_embedding = nn.Embedding(num_user, k)
        self.v_embedding = nn.Embedding(num_item, k)

        self.sequence_model = nn.Sequential(
            nn.Linear(k * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, u, v):
        emb_u = self.u_embedding(u)
        emb_v = self.v_embedding(v)
        input_vec = torch.cat((emb_u, emb_v), dim=1)
        output = self.sequence_model(input_vec)
        return output


class NCF(nn.Module):
    """
    Neural Collaborative Filtering model for rating prediction.

    Args:
    - num_user (int): number of unique users in the input data.
    - num_item (int): number of unique items in the input data.
    - k_gmf (int): dimension of the GMF latent space.
    - k_mlp (int): dimension of the MLP latent space.

    Attributes:
    - u_embedding_gmf (nn.Embedding): user embedding layer for GMF.
    - v_embedding_gmf (nn.Embedding): item embedding layer for GMF.
    - u_embedding_mlp (nn.Embedding): user embedding layer for MLP.
    - v_embedding_mlp (nn.Embedding): item embedding layer for MLP.
    - mlp_model (nn.Sequential): multi-layer perceptron for MLP.
    - h (nn.Linear): final linear layer for the model.
    - sigmoid (nn.Sigmoid): sigmoid activation function.

    Methods:
    - forward(u, v): computes the forward pass of the model given user and item indices.
    """

    def __init__(self, num_user, num_item, k_gmf, k_mlp):
        super().__init__()
        self.u_embedding_gmf = nn.Embedding(num_user, k_gmf)
        self.v_embedding_gmf = nn.Embedding(num_item, k_gmf)

        self.u_embedding_mlp = nn.Embedding(num_user, k_mlp)
        self.v_embedding_mlp = nn.Embedding(num_item, k_mlp)

        self.mlp_model = nn.Sequential(
            nn.Linear(k_mlp * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU()
        )

        self.h = nn.Linear(k_gmf + 32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, u, v):
        """
        Computes the forward pass of the model given user and item indices.

        Args:
        - u (torch.Tensor): a tensor of user indices.
        - v (torch.Tensor): a tensor of item indices.

        Returns:
        - output (torch.Tensor): a tensor of predicted ratings for the input (u, v) pairs.
        """

        gmf_output = self.u_embedding_gmf(u) * self.v_embedding_gmf(v)
        mlp_output = self.mlp_model(torch.cat((self.u_embedding_gmf(u), self.v_embedding_gmf(v)), dim=1))

        concat = torch.cat((gmf_output, mlp_output), dim=1)
        output = self.sigmoid(self.h(concat))
        return output
