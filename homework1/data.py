import numpy as np
import torch
from torchvision import datasets, transforms
from collections import Counter


def loaders(exp_no, batch_size, p1=0, p2=0):
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
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)
    # if exp_no in [2, 3]:
    #     intensity, distribution, _ = pre_exp2()
    #     transform.transforms.insert(1, Exp2(p2, intensity, distribution))
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    if exp_no in [1, 3]:
        dataset1.targets = exp1(dataset1.targets.numpy(), p1)

    loaders = {}
    loaders['train'] = torch.utils.data.DataLoader(dataset1, batch_size=batch_size, num_workers=4,
                                                   pin_memory=True, shuffle=True)
    loaders['test'] = torch.utils.data.DataLoader(dataset2, batch_size=batch_size, num_workers=4,
                                                  pin_memory=True, shuffle=False)
    return loaders


def exp1(y, p):
    """The distribution of classes in the training set is almost uniform,
    therefore the labels are drawn uniformly.
    
    Args:
        y (np.array): Array of classes
        p (float): Percent of replaced classes in the train dataset
    Return:
        y (torch.tensor): Tensor of classes
    """
    idxs = np.random.choice(np.arange(y.shape[0]), size=int(y.shape[0] * p), replace=False)
    y[idxs] = np.random.choice(np.unique(y), size=idxs.shape[0], replace=True)
    y = torch.tensor(y)
    return y


def pre_exp2():
    """This method returns the distribution for selected pixel values based on the training set."""
    x_train = datasets.MNIST('../data', train=True, download=True,
                             transform=transforms.ToTensor()).data.numpy()
    x_rounded = ((x_train.flatten() * 10).round() / 10)
    intensity, pre_dist = zip(*Counter(x_rounded).items())
    distribution = np.array(pre_dist) / x_rounded.shape[0]
    return intensity, distribution, x_rounded


class Exp2(object):
    """Transform class that replaces random values, in a given batch,
     from the distribution returned by the method pre_exp2"""
    def __init__(self, p, intensity, distribution):
        '''
        Args:
            p (float): Percent of replaced pixels in a batch
            intensity (list[int]): Values of pixel distribution
            distribution (list[int]): Probabilities of pixel distribution
        '''
        self.p = p
        self.intensity = intensity
        self.distribution = distribution

    def __call__(self, x):
        x = x.numpy()
        nb_pixels = np.array(x.shape).prod()
        idxs = np.random.choice(np.arange(nb_pixels), int(nb_pixels * self.p), replace=False)
        x_flatten = x.flatten()
        x_flatten[idxs] = np.random.choice(self.intensity, idxs.shape[0], replace=True, p=self.distribution)
        xs = torch.tensor(x_flatten.reshape(x.shape))
        # print(np.allclose(x,xs))
        return xs
