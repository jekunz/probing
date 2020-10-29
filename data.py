import config
import torch
from torch.utils.data import DataLoader
import torch.utils.data as utils


class Dataset:
    """
    create training and dev data
    either for label or for pairs task

    init input: list of sentences

    access points:
    get_dataset(): torch dataloader
    get_dev_data(): tensors X,y
     ... of dim (#training examples x state size) resp. (#training examples x #targets)
    """

    def __init__(self, sentences, sentences_dev = None):

        X,y = self.create_data(sentences)

        if sentences_dev is None:
            self.dev_data = (X[config.split:], y[config.split:])
        else:
            devX, devy = self.create_data(sentences_dev)
            self.dev_data = (devX, devy)

        dset = utils.TensorDataset(X[:config.split], y[:config.split])
        self.data_loader = DataLoader(dset, batch_size=64, shuffle=True)

    def create_data(self, sentences):
        if config.task == 'pairs':
            X, y = Dataset.dep_data(sentences)
        elif config.task == 'labels':
            X, y = Dataset.label_data(sentences)
        elif config.task == 'pos':
            X, y = Dataset.pos_data(sentences)
        if config.inflate:
            X = Dataset.inflate(X)
        return X,y

    @staticmethod
    def inflate(old_tensor, new_size=2048):
        length = list(old_tensor.shape)[1]
        new_tensor = old_tensor
        while length < new_size:
            if new_size - length < 0:
                new_tensor = torch.cat((old_tensor, new_tensor), 1)
            else:
                new_tensor = torch.cat((new_tensor, old_tensor[:, :(new_size - length)]), 1)
            length = list(new_tensor.shape)[1]
        return new_tensor

    @staticmethod
    def dep_data(sentences):
        # prepare data for dependency edge prediction
        # input: list of sentence objects
        X_train = torch.FloatTensor()
        y_train = torch.LongTensor()

        for s in sentences:
            pos = s.positive()
            neg = s.negative()

            try:
                balance = int(len(neg) / len(pos))
            except ZeroDivisionError:
                continue

            pos = pos * balance
            pos_y = torch.ones([len(pos)], dtype=torch.long)
            neg_y = torch.zeros([len(neg)], dtype=torch.long)

            X = torch.stack(neg + pos)
            y = torch.cat((neg_y, pos_y), 0)
            X_train = torch.cat((X_train, X), 0)
            y_train = torch.cat((y_train, y), 0)

        return X_train, y_train

    @staticmethod
    def label_data(sentences):
        # prepare data for label prediction
        # input: list of sentence objects
        X_train = torch.FloatTensor()
        y_train = torch.LongTensor()

        for s in sentences:
            if len(s.dependencies) > 1:
                if config.add_context_features:
                    pairs, labels = s.labels_wcf()
                else: pairs, labels = s.labels()
                X_train = torch.cat((X_train, pairs), 0)
                y_train = torch.cat((y_train, labels), 0)

        # cheater version:
        # y_new = y_train.float()
        # X_train = torch.cat((X_train, y_new.view(-1, 1)), 1)
        return X_train, y_train

    @staticmethod
    def pos_data(sentences):
        # prepare data for pos tag prediction
        # input: list of sentence objects
        X_train = torch.FloatTensor()
        y_train = torch.LongTensor()

        for s in sentences:
            if len(s.tokens) > 1:
                words, tags = s.pos_tagging()
                X_train = torch.cat((X_train, words), 0)
                y_train = torch.cat((y_train, tags), 0)

        return X_train, y_train

    def get_dataset(self):
        return self.data_loader

    def get_dev_data(self):
        return self.dev_data