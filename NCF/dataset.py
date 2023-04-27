import pandas as pd
from torch.utils.data import Dataset


class NCFDataset(Dataset):
    """
    A PyTorch dataset that represents a neural collaborative filtering (NCF) dataset.
    The dataset contains positive and negative samples of user-item interactions.

    Args:
    - u (list or array): user IDs.
    - v (list or array): item IDs.
    - sample_len (int): the number of negative samples to generate per positive sample.

    Attributes:
    - token_df (pd.DataFrame): a pandas DataFrame that contains the positive samples.
    - negative_df (pd.DataFrame): a pandas DataFrame that contains the negative samples.
    - dataset (pd.DataFrame): a pandas DataFrame that contains both positive and negative samples.
    - u (pd.Series): a pandas Series that contains user IDs from the dataset.
    - v (pd.Series): a pandas Series that contains item IDs from the dataset.
    - r (pd.Series): a pandas Series that contains the ratings (1 for positive samples, 0 for negative samples) from the dataset.

    Methods:
    - negative_sampling(df, sample_len): generates negative samples using unigram distribution.
    - __len__(): returns the number of samples in the dataset.
    - __getitem__(idx): returns a tuple of (user, item) and rating for a given index in the dataset.
    """

    def __init__(self, u, v, sample_len):
        self.token_df = pd.DataFrame({'u': u, 'v': v})
        self.token_df['r'] = 1
        self.negative_df = self.negative_sampling(self.token_df, sample_len)
        self.dataset = pd.concat([self.token_df, self.negative_df], ignore_index=True)

        self.u = self.dataset['u']
        self.v = self.dataset['v']
        self.r = self.dataset['r']

    def negative_sampling(self, df, sample_len):
        """Generate negative samples for each positive sample in the dataset.

        Args:
        - df (pd.DataFrame): Pandas DataFrame containing user IDs, item IDs, and ratings.
        - sample_len (int): Number of negative samples to generate for each positive sample.
            
        Returns:
        - negative_df (pd.DataFrame): Pandas DataFrame containing negative samples.
        """
        import random
        negative_u, negative_v = [], []

        distribution = df['v'].value_counts() ** 0.75
        for user, item_list in df.groupby('u')['v'].apply(list).items():
            negative_table = []
            for item, cnt in distribution.astype(int).items():
                if item not in item_list:
                    negative_table += [item] * cnt

            negative_u += [user] * (len(item_list) * sample_len)
            negative_v += random.sample(negative_table, len(item_list) * sample_len)

        negative_df = pd.DataFrame({'u': negative_u, 'v': negative_v})
        negative_df['r'] = 0
        return negative_df

    def __len__(self):
        return len(self.u)

    def __getitem__(self,  idx: int):
        return (self.u[idx], self.v[idx]), self.r[idx]
