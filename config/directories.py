from os.path import dirname
from os.path import join

root = dirname(dirname(__file__))

global_parameters_directories = {'0': join(root, 'Databases/ratings100k.csv'),
                                 '1': join(root, 'Databases/ratings1M.csv'),
                                 '2': join(root, 'Databases/ratings10M.csv')}

summary_folder_directories = {'0': join(root, 'tmp/')}

log_folder_directories = {'0': join(root, 'tmp/experiment_logs/')}