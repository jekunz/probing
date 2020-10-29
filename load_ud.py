import torch
import config, word_representations

'''

for original UD files in conllu format, e.g. 'en_ewt-ud-train.conllu'.

'''


class Sentence:
    def __init__(self, tokens, pos, states, dependencies):
        self.tokens = tokens
        self.pos = pos
        self.states = states
        self.dependencies = dependencies

    def __str__(self):
        return ' '.join(t for t in self.tokens)

    # states of true dependencies for pairs task
    def positive(self):
        return [torch.cat((self.states[d[0]-1], self.states[d[1]-1]), 0) for d in self.dependencies]

    # negative sampling for pairs task
    def negative(self):
        tups = [elem[0:2] for elem in self.dependencies]
        non_deps = []
        for i in range(len(self.tokens)):
            for j in range(len(self.tokens)):
                if (i, j) not in tups and i != j:
                    non_deps.append(torch.cat((self.states[i-1], self.states[j-1]), 0))
        return non_deps

    # states and labels for labels task
    def labels(self):
        dep_pairs = []
        labels = []
        label_ind = []
        for d in self.dependencies:
            # print(len(self.states))
            # print(d)
            dep_pairs.append(torch.cat((self.states[d[0]-1], self.states[d[1]-1]), 0))
            labels.append(d[2])
        for l in labels:
            if l in config.label2ix:
                label_ind.append(config.label2ix[l])
            else:
                config.label2ix[l] = config.index
                config.index = config.index + 1
                label_ind.append(config.label2ix[l])
        return torch.stack(dep_pairs), torch.LongTensor(label_ind)

    # states and tags for pos task
    def pos_tagging(self):
        return torch.stack(self.states[0:len(self.pos)]), torch.LongTensor(self.pos)


def pos_to_ix(tag):
    tags = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PART', 'PRON', 'PROPN', 'PUNCT',
            'SCONJ', 'SYM', 'VERB', 'X']
    tag_to_ix = dict(zip(tags, list(range(len(tags)))))
    tag_index = tag_to_ix[tag]
    return torch.tensor(tag_index, dtype=torch.long)


if config.mode == 'bert':
    word_repr = word_representations.Bert()
elif config.mode == 'elmo':
    word_repr = word_representations.Elmo()
elif config.mode == 'glove':
    word_repr = word_representations.GloVe()


def read_conllu(mode):
    if mode == 'train':
        path = config.path_ud
    elif mode == 'test':
        path = config.path_dev_ud
    with open(path, 'r') as file:
        sentences = []
        pos = []
        deps = []
        tokens = []
        for i, line in enumerate(file):
            if i > config.train_size:
                break
            if line == '\n':
                text = '[CLS]' + ' '.join(tokens) + '[SEP]'
                pos = [pos_to_ix(tag) for tag in pos]
                if config.mode == 'bert':
                    sentences.append(Sentence(tokens, pos, word_repr.get_bert(text), deps))
                elif config.mode == 'elmo':
                    sentences.append(Sentence(tokens, pos, word_repr.get_elmo(' '.join(tokens)), deps))
                elif config.mode == 'glove':
                    sentences.append(Sentence(tokens, pos, word_repr.get_glove(' '.join(tokens)), deps))
                pos = []
                tokens = []
                deps = []
                continue
            if line[0] == '#':
                continue
            line = line.rstrip('\n')
            line = line.split('\t')
            symbols = ['.', ',', '<', '>', ':', ';', '\'', '/', '-', '_', '%', '@', '#', '$', '^', '*', '?', '!', "‘",
                       "’", "'", "+", '=', '|', '\’']
            if len(line[1]) > 1:
                for sym in symbols:
                    line[1] = line[1].replace(sym, '')
            if line[1] == '':
                line[1] = 'unk'
            tokens.append(line[1])
            pos.append(line[3])
            try:
                if int(line[6]) != 0:
                    deps.append((int(line[0]), int(line[6]), line[7]))
            except ValueError:
                # print("value error ; the following dependency was not appended:", line[0], line[6], line[7])
                # occurs with index of type '5.1'; rare ; can be ignored
                pass

        return sentences
