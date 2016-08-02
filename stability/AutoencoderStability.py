import tensorflow as tf
from Evaluation import Evaluation
from Loss import Loss
from Train import Train

from Autoencoder.Import import Import
from Autoencoder.Autoencoder import Autoencoder

class AutoencoderStability(Autoencoder):
    def __init__(self, database_id, hidden1_units, regularisation, learning_rate0, learning_decay, batch_size_evaluate,
                 batch_size_train, nb_epoch):
        super().__init__(database_id, hidden1_units, regularisation, learning_rate0, learning_decay, batch_size_evaluate,
                 batch_size_train, nb_epoch)

        self.Import = Import(database=self.database,
                             test_ratio=0.,
                             validation_ratio=0.1)

        self.Train_set, self.Validation_set, self.Test_set = self.Import.run()

        self.Loss = Loss()

        self.Train = Train(database=self.database,
                           Train_set=self.Train_set,
                           batch_size=self.batch_size_train,
                           learning_decay=self.learning_decay,
                           learning_rate0=self.learning_rate0)

        self.Evaluation = Evaluation(database_id=self.database,
                                     batch_size_evaluate=self.batch_size_evaluate,
                                     Train_set=self.Train_set)

