from factorisation.Factorisation import Factorisation
from factorisation.ImportFactorisation import ImportFactorisation

database_id = 1

Import = ImportFactorisation(validation_ratio=0.1,
                             test_ratio=0,
                             database_id=database_id)
TrainSet, ValidationSet, TestSet = Import.run()
Factorization = Factorisation(TrainSet=TrainSet,
                              ValidationSet=ValidationSet,
                              database_id=database_id,
                              landa=3,
                              iterations=10,
                              dimension=10)
print(Factorization.rmse)