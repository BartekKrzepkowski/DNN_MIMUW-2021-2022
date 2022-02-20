import gc
import os
import datetime
import copy
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import ChainMap
from itertools import product
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from IPython.display import clear_output


class Trainer:
    def __init__(self, model, loaders, criterion, optim):
        """
        Trainer initializer. Every argument is a shell of class or method that is initialized in init_run.

        Args:
            model (Net): Custom neural network
            loaders (dict[str, torch.utils.data.DataLoader]): Dict of train, test loaders
            criterion (torch.nn.NLLLoss): The negative log likelihood loss
            optim (torch.optim.Optimizer): Chosen optimizer
        """
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model_ = model
        self.loaders_ = loaders
        self.criterion_ = criterion
        self.optim_ = optim

    def run_trainer(self, iter_params, epochs, exp_name, key_params, device=None):
        """
        Main method of trainer.
        Init df -> [Pick run  -> Init Run -> [Run Epoch]_{IL} -> Update df]_{IL} -> Save df -> Plot Results
        {IL - In Loop}

        iter_params (IteratorParams): Iterator which yield run parameters
        epochs (int): Number of epochs
        exp_name (str): Name of experiment
        key_params (list(str)): List of parameters whose values differ between iterations
        device (torch.device): Specify if use gpu or cpu
        """
        self.is_p2 = False
        if key_params and 'p2' in key_params:
            self.pre_exp2()
            self.is_p2 = True
        self.device = device if device else self.device
        base_path = os.path.join(os.getcwd(), f'data/{exp_name}'
                                              f'_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')
        os.mkdir(base_path)
        df_runs = pd.DataFrame()
        for run, params_run in enumerate(iter_params):
            self.init_run(params_run)
            params_pooled = self.params_adjust(copy.deepcopy(params_run))
            fig_path = self.adjust_fig_path(base_path, key_params, params_pooled)
            liveloss = PlotLosses(outputs=[MatplotlibPlot(figpath=fig_path)])
            if self.is_p2: self.p2 = params_pooled['p2']
            for epoch in range(epochs):
                self.logs = {}

                self.model.train()
                self.run_epoch('train')

                self.model.eval()
                with torch.no_grad():
                    self.run_epoch('test')

                liveloss.update(self.logs)
                liveloss.send()
                gc.collect()

            df_runs = pd.concat([df_runs, pd.DataFrame({'log loss': self.logs['log loss'],
                                                        'accuracy': self.logs['accuracy'],
                                                        'val_log loss': self.logs['val_log loss'],
                                                        'val_accuracy': self.logs['val_accuracy'],
                                                        **params_pooled}, index=[run])], axis=0)
        df_runs.to_csv(f'{base_path}/{exp_name}.csv')
        if key_params:
            clear_output(wait=True)
            self.plot_results(key_params, df_runs, base_path)

    def init_run(self, params):
        """Initiate run."""
        self.model = self.model_(**params['model']).to(self.device)
        self.criterion = self.criterion_(**params['criterion']).to(self.device)
        self.optim = self.optim_(self.model.parameters(), **params['optim'])
        self.loaders = self.loaders_(**params['loaders'])

    def params_adjust(self, params):
        """Group run parameters from different dicts into one dict."""
        params['model']['dims'] = str(params['model']['dims'])
        return dict(ChainMap(*params.values()))

    def adjust_fig_path(self, base_path, key_features, params_pooled):
        key_value = '&'.join([f'{kf}={params_pooled[kf]}' for kf in key_features]) if key_features else ''
        return f'{base_path}/{key_value}.png'

    def run_epoch(self, phase):
        """Run whole epoch."""
        running_acc = 0.0
        running_loss = 0.0
        for i, (x_true, y_true) in enumerate(self.loaders[phase]):
            if self.is_p2:
                # x_true  = x_true / 0.3081 + 0.1307
                x_true = self.exp2(x_true.numpy())
                x_true = (x_true-0.1307)/0.3081
            x_true, y_true = x_true.to(self.device), y_true.to(self.device)
            y_pred = self.model(x_true)
            loss = self.criterion(y_pred, y_true)
            if phase == 'train':
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

            running_acc += (torch.argmax(y_pred.data, dim=1) == y_true).sum().item()
            running_loss += loss.item() * x_true.size(0)

        epoch_acc = running_acc / len(self.loaders[phase].dataset)
        epoch_loss = running_loss / len(self.loaders[phase].dataset)

        prefix = ''
        if phase == 'test': prefix = 'val_'
        self.logs[f'{prefix}accuracy'] = round(epoch_acc, 4)
        self.logs[f'{prefix}log loss'] = round(epoch_loss, 4)

    def pre_exp2(self, plot=False):
        from collections import Counter
        from torchvision import datasets, transforms
        """This method returns the distribution for selected pixel values based on the training set."""
        x_train = datasets.MNIST('../data', train=True, download=True,
                                 transform=transforms.ToTensor()).data.numpy()
        x_train = x_train / 255
        x_rounded = (x_train.flatten() * 10).round() * 255 / 10
        if plot: sns.histplot(x_rounded)
        self.intensity, pre_dist = zip(*Counter(x_rounded).items())
        self.distribution = np.array(pre_dist) / x_rounded.shape[0]

    def exp2(self, x):
        nb_pixels = np.array(x.shape).prod()
        idxs = np.random.choice(np.arange(nb_pixels), int(nb_pixels * self.p2), replace=False)
        x_flatten = x.flatten()
        x_flatten[idxs] = np.random.choice(self.intensity, idxs.shape[0], replace=True, p=self.distribution)
        return torch.tensor(x_flatten.reshape(x.shape))

    def plot_results(self, key_params, df, base_path):
        """
        Depending on the number of key parameters:
        (1) Line Plot with respect to key_params
        (2) Heatmap with respect to key_params
        (>2) Not available
        """
        if len(key_params) == 1:
            fig, axes = plt.subplots(nrows=1, ncols=2)
            df.plot(x=key_params[0], y=['log loss', 'val_log loss'],
                    title=f'train', figsize=(14, 7), ax=axes[0])
            df.plot(x=key_params[0], y=['accuracy', 'val_accuracy'],
                    title=f'_test', figsize=(14, 7), ax=axes[1])
            fig.savefig(f'{base_path}/plot.png')
        elif len(key_params) == 2:
            wide_format_df = df.pivot(index=key_params[0], columns=key_params[1])
            fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
            g1 = sns.heatmap(wide_format_df['log loss'], annot=True, ax=axes[0][0])
            g1.set_title('log loss_train')
            g2 = sns.heatmap(wide_format_df['val_log loss'], annot=True, ax=axes[1][0])
            g2.set_title('log loss_test')
            g3 = sns.heatmap(wide_format_df['accuracy'], annot=True, ax=axes[0][1])
            g3.set_title('accuracy_train')
            g4 = sns.heatmap(wide_format_df['val_accuracy'], annot=True, ax=axes[1][1])
            g4.set_title('accuracy_test')
            fig.savefig(f'{base_path}/heatmap.png')
        else:
            print(f'You selected {len(key_params)} parameters to choose the best value for you. '
                  f'Currently, a graphical option for this approach is not available. '
                  f'Check the result manually in the saved csv file.')


class IteratorParams(object):
    """Iterate over all given values of parameters."""
    def __init__(self, model_ls, loaders_ls, criterion_ls, optim_ls):
        self.product = list(product(model_ls, loaders_ls, criterion_ls, optim_ls))
        self.no_run = -1

    def __iter__(self):
        return self

    def __next__(self):
        self.no_run += 1
        if self.no_run < len(self.product):
            tuple_run = self.product[self.no_run]
            return {
                'model': tuple_run[0],
                'loaders': tuple_run[1],
                'criterion': tuple_run[2],
                'optim': tuple_run[3],
            }
        raise StopIteration
