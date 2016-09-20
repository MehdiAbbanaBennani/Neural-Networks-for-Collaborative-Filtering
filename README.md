# Neural networks for Collaborative Filtering

In order to run the code you need Tensorflow 0.10 and up, and Python3

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

A setup script will be added soon.
