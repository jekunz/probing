train_size = 10000

task = 'labels'      # pairs, labels, pos
mode = 'elmo'      # bert, elmo, glove

bert_layer = 0         # 0-12
elmo_layer = 'lstm_outputs1'     # word_emb, lstm_outputs1, lstm_outputs2, elmo

path_ud = 'en_ewt-ud-train.conllu'
path_dev_ud = 'en_ewt-ud-dev.conllu'

inflate = False      # adjust BERT/GloVE dims (default: 1024 f.e. word as in ELMo)

# random seeds for classifier
seeds = [1,2,3,4,5,6,7,8,9,10] #,11,12,13,14,15,16,17,18,19,20]


################################################################################
################################################################################

# end of manual settings

# adjust input & output shapes

if mode == 'elmo':
    layer = elmo_layer
elif mode == 'bert':
    layer = bert_layer
else: layer = None

# get parameters for classifier --> UPPER CASE
if mode == 'bert':
    if inflate:
        inp_dim = 2048
    else: inp_dim = 1536
elif mode == 'elmo' and layer == 'word_emb':
    if inflate:
        inp_dim = 2048
    else: inp_dim = 1024
elif mode == 'elmo':
    inp_dim = 2048
elif mode == 'glove':
    if inflate:
        inp_dim = 2048
    else: inp_dim = 600
else: mode = None

if task == 'pos':
    if mode == 'bert':
        if inflate:
            inp_dim = 1024
        else:
            inp_dim = 768
    elif mode == 'elmo' and layer == 'word_emb':
        inp_dim = 512
    elif mode == 'elmo':
        inp_dim = 1024
    elif mode == 'glove':
        if inflate:
            inp_dim = 1024
        else:
            inp_dim = 300
    else:
        mode = None


if task == 'labels':
    out_dim = 65
    num_epochs = 10
    split = 5000
elif task == 'pairs':
    out_dim = 2
    num_epochs = 1
    split = 300000
elif task == 'pos':
    out_dim = 17
    num_epochs = 10
    split = -1

# container for data
label2ix = {}
index = 0