from autoencoder.Dataset import Dataset
from scipy.sparse import csr_matrix

import numpy as np
from tools.tools import global_parameters
from tools.tools import sparse_indices


class DatasetStability(Dataset):
    def __init__(self, dataset, sets_parameters, stability_parameters):
        super().__init__(dataset)

        self.nb_users, self.nb_movies = global_parameters(sets_parameters=sets_parameters)[0:2]

        self.differences = stability_parameters['differences']
        self.probability = stability_parameters['probability']
        self.rmse = stability_parameters['rmse']
        print('check self.rmse different from zero:')
        print(self.rmse)
        self.subsets_number = stability_parameters['subsets_number']
        self.landa_array = stability_parameters['landa_array']

        print(self.landa_array)
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
        # TODO Check array size to optimize parameters
        print('check array size to optimize parameters')
        set1 = np.intersect1d(np.where(differences <= self.rmse),
                              np.where(probabilities <= self.probability))
        set2 = np.intersect1d(np.where(differences > self.rmse),
                              np.where(probabilities <= (1 - self.probability)))
        omega_p_set = np.union1d(set1, set2)
        return omega_p_set

    def belongings_update(self, set):
        belongings = np.random.rand(set.size)
        belongings *= self.subsets_number
        belongings += 1
        belongings = belongings.astype(int)
        return belongings

    def subsets_sizes_compute(self, belongings):
        subsets_sizes = np.zeros(self.subsets_number)
        for i in range(self.subsets_number):
            subsets_sizes[i] = np.size(np.where(belongings == i + 1))
        return subsets_sizes

    def eta_compute(self, subsets_sizes):
        eta = self.landa_array[0] / self.nb_elements
        eta += sum([x / y for x, y in zip(self.landa_array, subsets_sizes)])
        return eta

    def omega_p_coefficients_update(self, belongings, omega_p, eta, subsets_sizes):
        omega_p_size = np.size(omega_p)
        omega_p_coefficients = [eta - self.landa_array[belongings[i]] / subsets_sizes[belongings[i]]
                                for i in range(omega_p_size)]
        omega_p_coefficients = np.asarray(omega_p_coefficients)
        return omega_p_coefficients

    def loss_coefficients(self, omega_p_set, omega_p_coefficients, eta):
        loss_coefficients = np.zeros(self.nb_elements)
        loss_coefficients += eta
        loss_coefficients[omega_p_set] = omega_p_coefficients
        print('check that the value changed')
        return loss_coefficients

    def coefficients(self, loss_coefficients):
        indices_cols = self.ratings.indices
        indices_rows = sparse_indices(self.ratings)
        return csr_matrix((loss_coefficients, (indices_rows, indices_cols)),
                          shape=[self.nb_users, self.nb_movies])

    def run(self):
        omega_p_set = self.omega_p_build()
        belongings = self.belongings_update(omega_p_set)
        subsets_sizes = self.subsets_sizes_compute(belongings)
        eta = self.eta_compute(subsets_sizes)
        omega_p_coefficients = self.omega_p_coefficients_update(belongings=belongings,
                                                                omega_p=omega_p_set,
                                                                eta=eta,
                                                                subsets_sizes=subsets_sizes)
        loss_coefficients = self.loss_coefficients(omega_p_set=omega_p_set,
                                                   omega_p_coefficients=omega_p_coefficients)
        coefficients = self.coefficients(loss_coefficients=loss_coefficients)
        return coefficients

    def next_batch_train_stability(self, batch_size):
        start, end = self.next_range(batch_size, 'user')
        indices_user, values_user = self.subset(start, end, 'user')
        indices_coefficients, values_coefficients = self.subset(start, end, 'coefficients')
        return indices_user, values_user, indices_coefficients, values_coefficients
