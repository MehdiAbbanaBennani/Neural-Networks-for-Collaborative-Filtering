from os.path import dirname
from os.path import join

root = dirname(dirname(__file__))

global_parameters_directories = {'0': join(root, 'Databases/ratings100K.csv'),
                                 '1': join(root, 'Databases/ratings1M.csv')}

summary_folder_directories = {'0': join(root, 'tmp/')}

log_folder_directories = {'0': join(root, 'tmp/experiment_logs/')}