# Code for COLING 2020 Paper "Classifier Probes May Just Learn from Linear Context Features"

To reproduce results: 

## For experiments on syntactic dependency parsing: 

Run main.py. 
By default, the program expects the files 'en_ewt-ud-train.conllu' and 'en_ewt-ud-dev.conllu' in the same directory. 

The following parameters can be changed in config.py:
* train_size: integer; number of lines to be used from the train file
* task: string; ’pairs’ or ’labels’ for dependency pair / label prediction
* mode: string; representation to be used. Options: ’bert’, ’elmo’ or ’glove’
* bert_layer: integer; between 0 and 12
* elmo_layer: string; ’word_emb’, ’lstm_outputs1’, ’lstm_outputs2’, ’elmo’
* path_ud: string; name of conllu file for training
* path_ud_dev: string; name of conllu file for testing
* inflate: boolean; True if BERT/GloVE representation should be inflated to 1024 dims
* seeds: list with random seeds (integers) for the classifier

## For word identity prediction task:

Run word_identity.ipynb. 
The parameters are by default also changed in config.py (see above; except 'tasks' that will not have any effect).
