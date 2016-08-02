from autoencoder.Autoencoder import Autoencoder
from autoencoder.Import import Import


autoencoder_parameters = {'hidden1_units': 700,
                          'regularisation': 0.02,
                          'learning_rate0': 0.001,
                          'learning_decay': 0.9,
                          'batch_size_evaluate': 100,
                          'batch_size_train': 35,
                          'nb_epoch': 15}

sets_parameters = {'database_id': 1,
                   'test_ratio': 0.,
                   'validation_ratio': 0.1}

Import = Import(sets_parameters=sets_parameters)

autoencoder_sets = Import.run()

Autoencoder0 = Autoencoder(autoencoder_parameters=autoencoder_parameters,
                           autoencoder_sets=autoencoder_sets,
                           sets_parameters=sets_parameters)

Autoencoder0.run_training()


"""""
database = 1
hidden1_units = 700
regularisation = 0.02
# learning_rate0 = 0.01
learning_rate0 = 0.001
learning_decay = 0.9
# learning_decay = 0.2
batch_size_evaluate = 100
batch_size_train = 35
nb_epoch = 15

Autoencoder0 = Autoencoder(database_id=1,
                          hidden1_units=700,
                          regularisation=0.02,
                          learning_rate0=0.001,
                          learning_decay=0.9,
                          batch_size_evaluate=100,
                          batch_size_train=15,
                          nb_epoch=15)


"""""