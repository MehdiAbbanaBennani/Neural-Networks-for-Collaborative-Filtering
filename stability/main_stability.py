from factorisation.Factorisation import Factorisation
from factorisation.ImportFactorisation import ImportFactorisation
from Stability.DatasetStability import DatasetStability

database_id = 0

Import = ImportFactorisation(validation_ratio=0.1,
                             test_ratio=0,
                             database_id=database_id)
TrainSet, ValidationSet, TestSet = Import.run()
Factorization = Factorisation(TrainSet=TrainSet,
                              ValidationSet=ValidationSet,
                              database_id=database_id,
                              landa=3,
                              iterations=5,
                              dimension=10)

Dataset = DatasetStability(dataset=[TrainSet],
                           differences=Factorization.difference_matrix,
                           probability=0.9,
                           rmse=Factorization.rmse,
                           subsets_number=3,
                           landa_array=[0.5, 0.3, 0.15, 0.05],
                           database_id=database_id)
print(1)
