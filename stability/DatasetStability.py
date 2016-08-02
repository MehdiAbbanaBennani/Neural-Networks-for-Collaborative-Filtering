from autoencoder.Dataset import Dataset
from scipy.sparse import csr_matrix

import numpy as np
from tools import global_parameters


# TODO square differences instead of rmse
class DatasetStability(Dataset):
    def __init__(self, dataset, sets_parameters, stability_parameters):

        super().__init__(dataset)

        self.nb_users, self.nb_movies = global_parameters(database=sets_parameters['database_id'])[0:2]

        self.differences = stability_parameters['differences']
        self.probability = stability_parameters['probability']
        self.rmse = stability_parameters['rmse']
        self.subsets_number = stability_parameters['subsets_number']
        self.landa_array = stability_parameters['landa_array']

        assert np.sum(self.landa_array) == 1
        assert np.size(self.landa_array) == self.subsets_number + 1

        self.coefficients = self.run()
        
        self.category_indices = {'user': 0, 'rmse': 1, 'coefficients': 2}
        self.category_matrix = {'user': self.ratings,
                                'rmse': self.ratings, 
                                'coefficients': self.coefficients}
        self.category_permute = {'user': self.permute,
                                 'rmse': self.identity,
                                 'coefficients': self.permute}

    def omega_p_build(self):
        probabilities = np.random.rand(self.nb_elements)
        differences = np.absolute(self.differences.data)
        print('check array size to optimize parameters')
        set1 = np.intersect1d(np.where(differences <= self.rmse),
                              np.where(probabilities <= self.probability))
        set2 = np.intersect1d(np.where(differences > self.rmse),
                              np.where(probabilities <= (1 - self.probability)))
        omega_p_set = np.union1d(set1, set2)
        return omega_p_set

    def random_division(self, set):
        coefficients = np.random.choice(a=self.landa_array[1:(self.nb_elements+1)],
                                        size=np.size(set))
        return coefficients

    def loss_coefficients(self, omega_p_set, omega_p_coefficients):
        loss_coefficients = np.zeros(self.nb_elements)

        # Omega0
        loss_coefficients += self.landa_array[0]

        # Omega prime
        loss_coefficients[omega_p_set] += (1 - self.landa_array[0])

        # Removing omega_k
        loss_coefficients[omega_p_set] = loss_coefficients[omega_p_set] - omega_p_coefficients

        return loss_coefficients

    def coefficients(self, loss_coefficients):
        indices_cols = self.ratings.indices
        indices_rows = self.sparse_indices(self.ratings)
        return csr_matrix((loss_coefficients, (indices_rows, indices_cols)),
                          shape=[self.nb_users, self.nb_movies])

    def run(self):
        omega_p_set = self.omega_p_build()
        omega_p_coefficients = self.random_division(omega_p_set)
        loss_coefficients = self.loss_coefficients(omega_p_set=omega_p_set,
                                                   omega_p_coefficients=omega_p_coefficients)
        coefficients = self.coefficients(loss_coefficients=loss_coefficients)
        return coefficients
    
    def next_batch_train_stability(self, batch_size):
        start, end = self.next_range(batch_size, 'user')
        indices_user, values_user = self.subset(start, end, 'user')
        indices_coefficients, values_coefficients = self.subset(start, end, 'coefficients')
        return indices_user, values_user, indices_coefficients, values_coefficients
