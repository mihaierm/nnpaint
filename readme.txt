1. Files in this package:

AI-CW1-Report.pdf - the final report
src/neural_networks.py - code for training various neural networks
src/search_py - code for performing the search
img/* - various search results
All the other files in the main folder are saved data of neural networks. See below how to use them.


2. How to use the program

Python version: 3.6.2
Required libraries: TensorFlow, numpy, OpenCV, os


2.1. To train a neural network: 

- Edit neural_networks.py, modify the last line of code if necessary (the building function and the saving location)
- From a command prompt inside src, execute: "python neural_networks.py"

2.2. To perform a search:

- Edit search.py (final section) to modify the neural network being used or other parameters. To use simulated annealing, set annealing_schedule_fn=annealing1 in the generic_search call
- From a command prompt inside src, execute: "python search.py"

The default output is in img/2layers5-fchc-1