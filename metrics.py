import torch
import math
import matplotlib.pyplot as plt


class Evaluator:
    """includes best epoch's final accuracy and WS metric as in Talmor et al. 2019"""

    def __init__(self, model):
        self.steps = [0, 64, 128, 256, 512, 1024, 2048, 4096]
        self.weights = [0, 0.23, 0.2, 0.17, 0.14, 0.11, 0.08, 0.07]
        self.model = model

        self.acc_list = []
        self.final_acc = 0

    def accuracy(self, X, y, model):
        correct = 0
        total = 0
        for state, target in zip(X, y):
            outputs = model(state)
            max_value = torch.max(outputs.data)
            pred = (outputs.data == max_value).nonzero()
            total += 1
            correct += (pred == target).sum().item()
        try:
            acc = 100 * correct / total
            # save accuracy of best epoch only
            if acc > self.final_acc:
                self.final_acc = acc
            return acc
        except ZeroDivisionError:
            print('The dev set seems to be empty.')
            raise ZeroDivisionError

    # accuracy function for use during training; result is appended to self.accs
    def accuracy_for_ws(self, X, y):
        correct = 0
        total = 0
        for state, target in zip(X, y):
            outputs = self.model(state)
            max_value = torch.max(outputs.data)
            pred = (outputs.data == max_value).nonzero()
            total += 1
            correct += (pred == target).sum().item()
        try:
            acc = 100 * correct / total
            self.acc_list.append(acc)
            return (acc)
        except ZeroDivisionError:
            print('ZeroDivisionError: The dev set seems to be empty.')

    # the WS metric can be calculated after training if self.accs has been build
    def calc_ws(self):
        ws = 0
        for i in range(len(self.weights)):
            ws += self.weights[i] * self.acc_list[i]
        return ws

    # plot the learning curve
    def plot_accs(self):
        plt.plot(self.steps, self.acc_list[0:len(self.steps)], 'ro')
        plt.ylabel('Accuracy')
        plt.xlabel('Training examples')
        plt.xscale('log')
        plt.xticks([64, 128, 256, 512, 1024, 2048, 4096], [64, 128, 256, 512, 1024, 2048, 4096])
        plt.show()


class MeanSD:
    """Calculate mean and standart deviation for objects of the class WS"""

    def __init__(self, evaluators):
        self.evaluators = evaluators

    # calculate mean of a list
    @staticmethod
    def mean(acc_list):
        return sum(acc_list) / len(acc_list)

    # calculate standard deviation of the mean of a list
    @staticmethod
    def std_deviation(acc_list):
        deviations = [math.pow(MeanSD.mean(acc_list) - r, 2) for r in acc_list]
        std_dev = math.sqrt((1 / (len(acc_list) - 1)) * sum(deviations))  # with Bessel's correction
        return std_dev

    # print mean and sd for WS scores
    def print_ws(self):
        ws_list = [e.calc_ws() for e in self.evaluators]
        ws_mean = MeanSD.mean(ws_list)
        std_dev = MeanSD.std_deviation(ws_list)
        print('WS:    Mean: {0}    Standard Deviation: {1}'.format(ws_mean, std_dev))

    # print mean and sd for accuracies of the best epoch
    def print_accs(self):
        acc_list = [e.final_acc for e in self.evaluators]
        acc_mean = MeanSD.mean(acc_list)
        std_dev = MeanSD.std_deviation(acc_list)
        print('Best epoch:    Mean: {0}    Standard Deviation: {1}'.format(acc_mean, std_dev))

    # print mean and sd for all intermediate accuracies used for WS
    def print_accuracies(self):
        for i in range(1, 8):
            acc_list = [e.acc_list[i] for e in self.evaluators]
            acc_mean = MeanSD.mean(acc_list)
            std_dev = MeanSD.std_deviation(acc_list)
            print('Step: {0}    Mean: {1}    Standard Deviation: {2}'.format(self.evaluators[0].steps[i], acc_mean,
                                                                             std_dev))

    # print mean and sd for all metrics
    def print_all(self):
        self.print_accs()
        self.print_ws()
        self.print_accuracies()