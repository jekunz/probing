import config
import models, load_ud, data, metrics


# create data
sentences = load_ud.read_conllu('train')
sentences_val = load_ud.read_conllu('test')
dset = data.Dataset(sentences, sentences_val)

devX, devy = dset.get_dev_data()

print("data loaded, contains {0} sentences and {1} training examples.".format(len(sentences), len(dset.data_loader)*64))

# train with different hidden dimensions
for hidden_dim in [1,2,4,6,16,32,64]:
    print("HIDDEN DIM: ", hidden_dim)
    # run models and save results
    evaluator_saver = []
    model_saver = []

    # train with different seeds
    # (as for low dimensionalities, the seed matters a lot)
    for s in config.seeds:
        trainer = models.ProbeTrainer(config.inp_dim, hidden_dim, config.out_dim)
        evaluator = trainer.train_probe(config.num_epochs, dset.data_loader, devX, devy, s)
        evaluator_saver.append(evaluator)
        model_saver.append(trainer)

    # calculate and print mean & sd
    mean_sd = metrics.MeanSD(evaluator_saver)
    mean_sd.print_all()