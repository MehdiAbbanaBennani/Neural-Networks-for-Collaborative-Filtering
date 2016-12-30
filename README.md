# Neural networks for Collaborative Filtering

Required packages :
Tensorflow 0.10
Python 3.5
Scikit-learn 0.17
Numpy 
Scipy
The code was developped under this versions of the packages, it might be incompatible with some newer versions due to the depreciation of some functions.

You can download the datasets by using the import.sh script from the bash subfloder
In order to specify the Dataset, you should add an option next to the script
0 : 100K
1 : 1M
2 : 10M
For example ./import.sh 1

Then you can choose the configuration by changing the first line on the main.py file.
For example:
from config.autoencoder.U_1M import Experiment
This runs the Autoencoder for 1M dataset under U setting (user setting)

from config.stability.V_1M import Experiment
This runs the Stability for 1M dataset under V setting (movie setting)

You can change the experiment parameters under the config/ subfolder.
