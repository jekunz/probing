import config
import numpy as np
import torch
import tensorflow as tf
import tensorflow_hub as hub
from transformers import BertTokenizer, BertModel, BertConfig

"""

classes Bert, Elmo, GloVe

methods get_[bert|elmo|glove] 
input: sentence represented as a string
return list with torch states 

"""


class Bert:
    def __init__(self):
        self.config = BertConfig.from_pretrained("bert-base-uncased", output_hidden_states=True)
        self.model = BertModel.from_pretrained('bert-base-uncased', config=self.config)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_basic_tokenize=True)
        self.bert_layer = config.layer
        self.model.eval()

    # align word pieces with words
    @staticmethod
    def collect_pieces(tokenized_text):
        output = []
        curr_token = []
        seq_length = len(tokenized_text)

        for i in range(seq_length):
            curr_piece = tokenized_text[i]
            curr_token.append((i, curr_piece))

            if i < seq_length - 1:
                next_piece = tokenized_text[i + 1]
                if not next_piece.startswith('##'):
                    output.append(curr_token)
                    curr_token = []

        output.append(curr_token)
        return output

    def get_bert(self, text):
        tokenized_text = self.tokenizer.tokenize(text)
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])

        with torch.no_grad():
            outputs = self.model(tokens_tensor)
            target_layer = outputs[2][self.bert_layer]

        collected_pieces = Bert.collect_pieces(tokenized_text)
        token_states = []
        for t in collected_pieces:
            token_index = t[-1][0]  # taking last word piece
            token_states.append(target_layer[0, token_index])
        return token_states[1:len(token_states)]    # -1 ???


class Elmo:
    def __init__(self):
        # layer is string from ['word_emb', 'lstm_outputs1', 'lstm_outputs2', 'elmo']
        self.elmo_model = hub.load("https://tfhub.dev/google/elmo/3")
        self.elmo_layer = config.layer

    def get_elmo(self, sentence):
        t = tf.constant([sentence])
        u = self.elmo_model.signatures['default'](t)
        state_lst = u[self.elmo_layer][0]
        return [torch.from_numpy(state.numpy()).float() for state in state_lst]


class GloVe:
    def __init__(self):
        self.embeddings_dict = {}
        with open("glove.42B.300d.txt", 'r') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], "float32")
                self. embeddings_dict[word] = vector

    def get_glove(self, sentence):
        states = []
        for w in sentence.split(' '):
            if w in self.embeddings_dict:
                states.append(torch.from_numpy(self.embeddings_dict[w]))
            else:
                states.append(torch.from_numpy(self.embeddings_dict['unk']))
        return states
