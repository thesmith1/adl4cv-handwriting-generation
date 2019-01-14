# adl4cv-handwriting-generation
Repository for Project in Advanced Deep Learning for Computer Vision at TUM

## Installation

To create a new virtual environment with the required dependencies:

```conda env create -f adl4cv-env.yml```

To log into Tensorboard and check the training procedure live, an environment with Tensorflow (and Tensorboard)
is required. From that environment:

```tensorboard --logdir model/runs```


## Project Structure
The folder `model/` contains the GAN classes and the models of all discriminators and generators.
It is also the base folder to store Tensorboard runs.

The folder `preprocessing` contains the code required to process the raw crops of characters and
the label files

The folder `data_management/` contains the class CharacterDataset.

The folder `utils` contains common definitions and frequently used functions, the crop utility
(required to crop the scanned pages of raw handwritten text) and the script used to compute the
content of the dataset.

Finally, the script `train.py` is the entry point of the training procedure.