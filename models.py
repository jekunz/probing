import metrics

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import time


class ProbingModule(nn.Module):
    def __init__(
            self,
            input_dim,
            hidden_dim,
            output_dim,
    ):
        super(ProbingModule, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, X, **kwargs):
        X = F.relu(self.hidden(X))
        X = self.output(X)
        return X


class ProbeTrainer:
    def __init__(self, inp_dim, hidden_dim, out_dim):
        self.model = ProbingModule(inp_dim, hidden_dim, out_dim)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0.0)
        self.pre_trained = False

    # general training or pre-training for TL probe
    def train_probe(self, num_epochs, data_loader, devX, devy, manual_seed=1337):
        evaluator = metrics.Evaluator(self.model)
        n_steps = 0
        epoch_accuracies = []
        if not self.pre_trained:
            torch.manual_seed(manual_seed)
        self.pre_trained = True

        for epoch in range(num_epochs):
            start_time = time.time()
            for _, (state, target) in enumerate(data_loader):
                self.optimizer.zero_grad()
                tag_score = self.model(state)
                loss = self.criterion(tag_score, target)
                loss.backward()
                self.optimizer.step()

                if n_steps in evaluator.steps:
                    with torch.no_grad():
                        print("n_steps: {0}, accuracy: {1} ".format(n_steps, evaluator.accuracy_for_ws(devX, devy)))
                n_steps += 64

            with torch.no_grad():
                epoch_acc = evaluator.accuracy(devX, devy, self.model)
                epoch_accuracies.append(epoch_acc)
                print("Epoch: {0}, Seconds: {1}, Loss {2}, Acc Dev: {3}".format(epoch, (time.time() - start_time), loss,
                                                                                epoch_acc))

        print('WS: {}'.format(evaluator.calc_ws()))
        print('Best accuracy: {}'.format(max(epoch_accuracies)))
        # evaluator.plot_accs()
        return evaluator
