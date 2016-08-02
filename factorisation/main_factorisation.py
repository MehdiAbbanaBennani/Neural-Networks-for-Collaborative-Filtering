from factorisation.Factorisation import Factorisation
from stability.ImportStability import ImportStability


factorisation_parameters = {'landa': 3,
                            'iterations': 10,
                            'dimension': 10}

sets_parameters = {'database_id': 1,
                   'test_ratio': 0.,
                   'validation_ratio': 0.1}

Import = ImportStability(sets_parameters=sets_parameters)
factorisation_sets, autoencoder_sets = Import.run()

Factorization = Factorisation(factorisation_sets=factorisation_sets,
                              factorisation_parameters=factorisation_parameters,
                              sets_parameters=sets_parameters)

print(Factorization.rmse)