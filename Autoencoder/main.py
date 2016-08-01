from Autoencoder import Autoencoder


# Autoencoder0 = Autoencoder(database_id=1,
#                           hidden1_units=700,
#                           regularisation=0.025113243590272,
#                           learning_rate0=0.090054066465632,
#                           learning_decay=0.17623840298814,
#                           batch_size_evaluate=100,
#                           batch_size_train=35,
#                           nb_epoch=15)

Autoencoder0 = Autoencoder(database_id=1,
                           hidden1_units=700,
                           regularisation=0.02,
                           learning_rate0=0.001,
                           learning_decay=0.9,
                           batch_size_evaluate=100,
                           batch_size_train=35,
                           nb_epoch=15)

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