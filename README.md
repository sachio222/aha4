# Artificial Hippocampal Algorithm in Pytorch

ATTN: This is a work in progress.

Pytorch implementation of AHA! an ‘Artificial Hippocampal Algorithm’ for Episodic Machine Learning by Kowadlo, Rawlinson and Ahmed (2019). 

## Getting Started

Use Pipenv to install dependencies and create compatible virtual environment. (https://thoughtbot.com/blog/how-to-manage-your-python-projects-with-pipenv)

 - Requires Python version >= 3.6


### Running
Weights should already be trained for the VC module, so to run predictions, do the following:

1. run /eval.py
2. Save training set image.
3. run /test.py
4. Save test set image, save predictions image.

In /experiments/train/params.json:
 - modify test_seed or test_shift values to change the test set variations.


## Built With

* [Pytorch](https://pytorch.org/) - Artificial neural network framework
* [Pipenv](https://pypi.org/project/pipenv/) - Dependency Management


## Authors

* **Jacob Krajewski** - *Pytorch implementation*


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Thanks to **Gideon Kowadlo** and **David Rawlinson** for guiding me through some of the more difficult elements of the ANN. 
* Thanks to @ptrblk on the Pytorch forums for walking me through some of the more confusing aspects of Pytorch. 

