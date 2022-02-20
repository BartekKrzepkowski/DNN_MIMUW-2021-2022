import numpy as np
import pandas as pd
import torch
from torchvision import datasets, transforms
from collections import Counter


class NaiveVectorizer:
    def __init__(self, tokenized_data, **kwargs):
        """Converts data from string to vector of ints that represent words. 
        Prepare lookup dict (self.wv) that maps token to int. Reserve index 0 for padding.
        """
        tokenized_data = [seq.split() for seq in tokenized_data]
        tokenized_data = [token for sublist in tokenized_data for token in sublist]
        ### Your code goes here ###
        ind2word = dict(enumerate(np.unique(np.array(tokenized_data))))
        self.wv = {word: idx + 1 for idx, word in ind2word.items()}

    def vectorize(self, tokenized_seq):
        """Converts sequence of tokens into sequence of indices.
        If the token does not appear in the vocabulary(self.wv) it is ommited
        Returns torch tensor of shape (seq_len,) and type long."""
        ### Your code goes here ###
        return [self.wv[token] for token in tokenized_seq if token in self.wv]

    
class ImdbDataset(torch.utils.data.Dataset):
    def __init__(self, data, preprocess_fn, split="train"):
        SPLIT_TYPES = ["train", "test", "unsup"]
        super(ImdbDataset, self).__init__()
        if split not in SPLIT_TYPES:
            raise AttributeError(f"No such split type: {split}")

        self.split = split
        self.label = [i for i, c in enumerate(data.columns) if c == "sentiment"][0]
        self.data_col = [i for i, c in enumerate(data.columns) if c == "tokenized"][0]
        self.data = data[data["split"] == self.split]
        self.preprocess_fn = preprocess_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.preprocess_fn(self.data.iloc[idx, self.data_col].split())
        label = self.data.iloc[idx, self.label]
        seq = torch.tensor(seq, dtype=torch.long)
        label = torch.tensor(label)
        return (seq, label)
    
def custom_collate_fn(pairs):
    """This function is supposed to be used by dataloader to prepare batches
    Input: list of tuples (sequence, label)
    Output: sequences_padded_to_the_same_lenths, original_lenghts_of_sequences, lables.
    torch.nn.utils.rnn.pad_sequence might be usefull here
    """
    ### Your code goes here ###
    seqs, labels = zip(*pairs)
    lengths = torch.tensor([len(seq) for seq in seqs])
    seqcs = torch.nn.utils.rnn.pad_sequence(seqs)
    labels = torch.stack(labels)
    #################################
    return seqcs, lengths, labels

def get_loaders(path_dataset, batch_size):
    """
    Return proper loaders of train and test dataset, according to the chosen experiment.

    Args:
        exp_no (int): Number of chosen experiment
        p1 (float): Percent of replaced classes in the train dataset
        p2 (float): Percent of replaced pixels in a batch
        batch_size (int): Size of batch
    Return:
        loaders (dict[str, torch.utils.data.DataLoader])
    """
    dataset = pd.read_csv(path_dataset)
    naive_vectorizer = NaiveVectorizer(dataset.loc[dataset["split"] == "train", "tokenized"])
    vocab_size = len(naive_vectorizer.wv)
    
    train_dataset = ImdbDataset(dataset, naive_vectorizer.vectorize)
    test_dataset = ImdbDataset(dataset, naive_vectorizer.vectorize, split="test")

    loaders = {}
    loaders['train'] = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn, num_workers=8, pin_memory=True)
    loaders['test'] = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn, num_workers=8, pin_memory=True)
    return loaders, vocab_size